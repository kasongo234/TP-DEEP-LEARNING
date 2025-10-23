# collect_results.py
import os, csv, glob
def last_row(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    hdr, data = rows[0], rows[1:]
    if not data: return None
    e = data[-1]
    return {
        "epoch": int(e[0]),
        "train_loss": float(e[1]),
        "train_acc": float(e[2]),
        "val_loss": float(e[3]),
        "val_acc": float(e[4]),
        "val_precision": float(e[5]),
        "val_recall": float(e[6]),
    }

def main():
    runs = sorted(glob.glob("runs/*"))
    print("| Run | Epoch | Val Acc | Val Prec | Val Rec | Train Loss | Val Loss |")
    print("|-----|------:|--------:|---------:|--------:|-----------:|---------:|")
    for r in runs:
        csv_path = os.path.join(r, "history.csv")
        if not os.path.exists(csv_path): continue
        m = last_row(csv_path)
        if not m: continue
        name = os.path.basename(r)
        print(f"| {name} | {m['epoch']} | {m['val_acc']:.3f} | {m['val_precision']:.3f} | {m['val_recall']:.3f} | {m['train_loss']:.3f} | {m['val_loss']:.3f} |")

if __name__ == "__main__":
    import csv
    main()
