# eval.py
import argparse, os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import CNNFromScratch, get_transfer_model
from utils import eval_epoch, plot_confusion_matrix

def build_loader(data_dir, img_size=224, batch_size=32, workers=0):
    mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
    tf = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    ds = datasets.ImageFolder(root=data_dir, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False), ds.classes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="chemin du .pth")
    ap.add_argument("--data_dir", required=True, help="dossier à évaluer (ex: data/test)")
    ap.add_argument("--exp", choices=["A","B"], required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out_png", type=str, default="confusion_eval.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, classes = build_loader(args.data_dir, img_size=args.img_size, batch_size=args.batch_size, workers=args.workers)

    if args.exp == "A":
        model = CNNFromScratch(num_classes=len(classes))
    else:
        model = get_transfer_model(num_classes=len(classes), pretrained=False, freeze_backbone=True)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    acc, prec, rec, f1, cm = eval_epoch(model, loader, device)
    print(f"[EVAL] acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")
    plot_confusion_matrix(cm, args.out_png, class_names=classes)
    print(f"Confusion matrix saved to {args.out_png}")

if __name__ == "__main__":
    main()
