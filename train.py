# train.py
import os, time, argparse, csv
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from models import CNNFromScratch, get_transfer_model
from utils import set_seed, eval_epoch, plot_metrics, plot_confusion_matrix

def build_dataloaders(
    data_root="data", img_size=224, batch_size=32,
    val_split=0.2, augment=True, workers=0, subset=1.0
):
    """
    Charge data/train et fait un split train/val. data/test optionnel.
    subset: fraction du dataset à utiliser (0<subset<=1).
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    ds_full = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=train_tf)

    # Sous-échantillonnage pour accélérer en CPU
    if subset < 1.0:
        assert 0 < subset <= 1.0, "--subset doit être dans (0,1]"
        sub_len = max(2, int(len(ds_full) * subset))
        rest = len(ds_full) - sub_len
        ds_full, _ = random_split(ds_full, [sub_len, rest], generator=torch.Generator().manual_seed(42))

    # Split 80/20
    n_val = int(val_split * len(ds_full))
    n_train = len(ds_full) - n_val
    train_ds, val_ds = random_split(ds_full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    val_ds.dataset.transform = val_tf  # pas d'augmentation en val

    # test (optionnel)
    test_loader = None
    test_dir = os.path.join(data_root, "test")
    if os.path.isdir(test_dir) and any(os.scandir(test_dir)):
        test_ds = datasets.ImageFolder(root=test_dir, transform=val_tf)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=workers, pin_memory=False)

    # DataLoaders (Windows/CPU-friendly)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=False)

    return train_loader, val_loader, test_loader, ds_full.dataset.classes if hasattr(ds_full, "dataset") else ds_full.classes


def train_one_epoch(model, loader, device, criterion, optimizer, grad_clip=0.0, max_batches=0):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for i, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        if grad_clip and grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if max_batches and i >= max_batches:
            break
    train_loss = running_loss / total if total else 0.0
    train_acc = correct / total if total else 0.0
    return train_loss, train_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["A","B"], default="A", help="A: CNN from scratch, B: transfer learning")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam","sgd"], default="adam")
    parser.add_argument("--scheduler", choices=["step","cosine",None], default="step")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)  # Windows/CPU safe
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--outdir", type=str, default="runs")
    parser.add_argument("--ckptdir", type=str, default="checkpoints")
    # QoL / CPU
    parser.add_argument("--fast_debug", action="store_true", help="test rapide: epochs/img/batch réduits")
    parser.add_argument("--clip", type=float, default=0.0, help="gradient clipping (0=off)")
    parser.add_argument("--patience", type=int, default=0, help="early stopping sur val_loss (0=off)")
    parser.add_argument("--subset", type=float, default=1.0, help="fraction du dataset train/val à utiliser (0<subset<=1)")
    parser.add_argument("--max_train_batches", type=int, default=0, help="limite de batches par époque (0=pas de limite)")
    # Transfer learning
    parser.add_argument("--pretrained", action="store_true", help="backbone pré-entraîné (B)")
    parser.add_argument("--no_freeze", action="store_true", help="ne pas geler le backbone (B)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} (GPU dispo: {torch.cuda.is_available()})")

    # Limiter les threads CPU (Windows)
    try:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    except Exception:
        pass

    # Fast debug
    if args.fast_debug:
        args.epochs = 1
        args.batch_size = 4
        args.img_size = 128
        args.augment = False
        args.subset = min(args.subset, 0.1)
        args.max_train_batches = max(args.max_train_batches, 50)
        print("[fast_debug] epochs=1, batch=4, img=128, augment=False, subset<=0.1, max_train_batches>=50")

    os.makedirs(args.ckptdir, exist_ok=True)
    os.makedirs(args.outdir,  exist_ok=True)

    # Loaders
    train_loader, val_loader, test_loader, classes = build_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=0.2,
        augment=args.augment,
        workers=args.workers,
        subset=args.subset
    )
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader .dataset)
    n_test  = len(test_loader.dataset) if test_loader is not None else 0
    print(f"Datasets -> train: {n_train} | val: {n_val} | test: {n_test}")

    # Modèle
    if args.exp == "A":
        try:
            model = CNNFromScratch(num_classes=len(classes), dropout_p=args.dropout).to(device)
        except TypeError:
            model = CNNFromScratch().to(device)
    else:
        model = get_transfer_model(
            num_classes=len(classes),
            backbone="resnet18",
            pretrained=args.pretrained,
            freeze_backbone=(not args.no_freeze),
            dropout_p=args.dropout
        ).to(device)

    # Optimiseur
    params = (p for p in model.parameters() if p.requires_grad)
    if args.optimizer == "adam":
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # Scheduler
    if args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=max(1, args.epochs//3), gamma=0.1)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()

    # Historique
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_precision": [], "val_recall": []}

    best_val_acc = -1.0
    best_val_loss = float("inf")
    no_improve = 0
    run_name = f"{'fromscratch' if args.exp=='A' else 'transfer'}_{args.optimizer}_lr{args.lr}"
    best_ckpt = os.path.join(args.ckptdir, f"best_{run_name}.pth")
    last_val_cm = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, device, criterion, optimizer,
            grad_clip=args.clip, max_batches=args.max_train_batches
        )

        # Validation loss
        model.eval()
        val_running_loss, val_total = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_running_loss += criterion(logits, y).item() * y.size(0)
                val_total += y.size(0)
        val_loss = (val_running_loss / val_total) if val_total else 0.0

        # Validation metrics
        val_acc, val_prec, val_rec, val_f1, val_cm = eval_epoch(model, val_loader, device)
        last_val_cm = val_cm

        # Historique
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)

        # Meilleur checkpoint (sur val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "classes": classes,
                        "exp": args.exp}, best_ckpt)

        # Early stopping sur val_loss
        if args.patience > 0:
            if val_loss + 1e-8 < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"Early stopping: pas d'amélioration de val_loss depuis {args.patience} époque(s).")
                    break

        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        print(f"[{epoch}/{args.epochs}] "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} prec={val_prec:.3f} rec={val_rec:.3f} | "
              f"{dt:.1f}s")

    # Sorties: figures + matrices + CSV
    out_run = os.path.join(args.outdir, run_name)
    os.makedirs(out_run, exist_ok=True)
    plot_metrics(history, out_run)
    if last_val_cm is not None:
        plot_confusion_matrix(last_val_cm, os.path.join(out_run, "confusion_val.png"), class_names=classes)

    # CSV des métriques
    csv_path = os.path.join(out_run, "history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","train_acc","val_loss","val_acc","val_precision","val_recall"])
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i+1,
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i],
                history["val_precision"][i],
                history["val_recall"][i],
            ])
    print(f"Métriques enregistrées dans {csv_path}")

    # Test final (si présent) en rechargeant le meilleur modèle
    if test_loader is not None and os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_acc, test_prec, test_rec, test_f1, test_cm = eval_epoch(model, test_loader, device)
        print(f"[TEST] acc={test_acc:.3f} prec={test_prec:.3f} rec={test_rec:.3f} f1={test_f1:.3f}")
        plot_confusion_matrix(test_cm, os.path.join(out_run, "confusion_test.png"), class_names=classes)

if __name__ == "__main__":
    main()
