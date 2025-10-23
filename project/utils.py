#utils.py
import random, numpy as np, torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def eval_epoch(model, loader, device):
    """Retourne: acc, precision, recall, f1, confusion_matrix"""
    model.eval()
    total, correct = 0, 0
    preds_all, targets_all = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        preds_all.append(preds.cpu())
        targets_all.append(y.cpu())
    acc = correct / total if total else 0.0
    y_true = torch.cat(targets_all).numpy()
    y_pred = torch.cat(preds_all).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return acc, precision, recall, f1, cm

def plot_metrics(history, outdir):
    """
    history: dict avec clés
    - train_loss, train_acc, val_loss, val_acc, val_precision, val_recall
    Sauvegarde loss.png, accuracy.png, precision_recall.png
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Loss
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"],   label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss"); plt.legend()
    plt.savefig(outdir / "loss.png", dpi=160); plt.close()

    # Accuracy
    plt.figure()
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"],   label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy"); plt.legend()
    plt.savefig(outdir / "accuracy.png", dpi=160); plt.close()

    # Precision & Recall
    plt.figure()
    plt.plot(history["val_precision"], label="val_precision")
    plt.plot(history["val_recall"],   label="val_recall")
    plt.xlabel("epoch"); plt.ylabel("score"); plt.title("Precision & Recall"); plt.legend()
    plt.savefig(outdir / "precision_recall.png", dpi=160); plt.close()

def plot_confusion_matrix(cm, outpath, class_names=("cat","dog")):
    import numpy as np
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matrice de confusion"); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45); plt.yticks(ticks, class_names)
    plt.tight_layout(); plt.ylabel("Vrai"); plt.xlabel("Prédit")
    plt.savefig(outpath, dpi=160); plt.close()
