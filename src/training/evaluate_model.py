# src/training/evaluate_model.py
import os, json, argparse, itertools, time, math
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


# ------------------ CLI ------------------
def get_args():
    p = argparse.ArgumentParser(description="Evaluate coin classifier")
    p.add_argument("--model", default="models/coin_classifier_v2.h5")
    p.add_argument("--data_dir", default="data/StressTest",
                   help="Directory with class subfolders")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--labels", default="class_labels.json")
    p.add_argument("--out_dir", default="runs/StressTest")
    p.add_argument("--train_dir", default="data/Train",
                   help="Used only for duplicate-leakage scan")
    p.add_argument("--dup_scan", action="store_true",
                   help="Enable simple perceptual-hash duplicate scan against Train/")
    return p.parse_args()


# ------------------ Utilities ------------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_class_names(labels_path: str, class_indices: dict | None) -> list[str]:
    """
    Supports:
      - ["label0", "label1", ...]
      - {"0": "label0", "1": "label1", ...}
    Falls back to generator's class_indices order if provided.
    """
    lp = Path(labels_path)
    if lp.exists():
        try:
            with open(lp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
            if isinstance(data, dict):
                pairs = sorted(((int(k), v) for k, v in data.items()), key=lambda t: t[0])
                return [v for _, v in pairs]
        except Exception:
            pass
    if class_indices:
        pairs = sorted(class_indices.items(), key=lambda kv: kv[1])
        return [k for k, _ in pairs]
    return []


def plot_confusion(cm: np.ndarray, class_names: list[str], out_path: Path, figsize=(12, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix — StressTest"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        v = cm[i, j]
        if v:
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > thresh else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc_micro_macro(y_true_onehot: np.ndarray,
                         y_score: np.ndarray,
                         out_path: Path):
    """
    Saves a plot with micro & macro ROC curves and prints AUCs.
    """
    n_classes = y_true_onehot.shape[1]

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), y_score.ravel())
    auc_micro = roc_auc_score(y_true_onehot, y_score, average="micro", multi_class="ovr")

    # Macro-average
    fprs, tprs, aucs = [], [], []
    for c in range(n_classes):
        fpr_c, tpr_c, _ = roc_curve(y_true_onehot[:, c], y_score[:, c])
        fprs.append(fpr_c); tprs.append(tpr_c)
        aucs.append(roc_auc_score(y_true_onehot[:, c], y_score[:, c]))

    # Interpolate macro curve over a common grid
    all_fpr = np.unique(np.concatenate([f for f in fprs]))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr_c, tpr_c in zip(fprs, tprs):
        mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
    mean_tpr /= n_classes
    auc_macro = roc_auc_score(y_true_onehot, y_score, average="macro", multi_class="ovr")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(fpr_micro, tpr_micro, label=f"micro-average (AUC = {auc_micro:.4f})")
    ax.plot(all_fpr, mean_tpr, label=f"macro-average (AUC = {auc_macro:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves — Micro & Macro (StressTest)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"AUC (macro): {auc_macro:.4f} | AUC (micro): {auc_micro:.4f}")


# Simple perceptual hash (aHash) without extra deps
def compute_ahash(pil_img: Image.Image, hash_size: int = 8) -> int:
    img = pil_img.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    avg = arr.mean()
    bits = (arr > avg).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def scan_duplicate_hashes(train_dir: Path, eval_dir: Path, img_exts=(".jpg", ".jpeg", ".png", ".webp")):
    def iter_imgs(root: Path):
        for p in root.rglob("*"):
            if p.suffix.lower() in img_exts:
                yield p

    print("\n🔎 Duplicate/leakage scan (aHash)…")
    t0 = time.time()
    train_hashes = set()
    n_train = 0
    for p in iter_imgs(train_dir):
        try:
            h = compute_ahash(Image.open(p))
            train_hashes.add(h); n_train += 1
        except Exception:
            pass

    n_eval, dup = 0, 0
    dup_paths = []
    for p in iter_imgs(eval_dir):
        try:
            h = compute_ahash(Image.open(p)); n_eval += 1
            if h in train_hashes:
                dup += 1
                if len(dup_paths) < 10:
                    dup_paths.append(str(p))
        except Exception:
            pass

    elapsed = time.time() - t0
    pct = (dup / max(1, n_eval)) * 100.0
    print(f"  Train images hashed: {n_train}")
    print(f"  Eval images checked: {n_eval}")
    print(f"  Possible near-duplicates: {dup}  ({pct:.2f}%)")
    if dup_paths:
        print("  Sample dup paths:")
        for s in dup_paths:
            print("   •", s)
    print(f"  Scan time: {elapsed:.1f}s\n")


# ------------------ Main ------------------
def main():
    args = get_args()

    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    # --- Load model
    print(f"📦 Loading model: {args.model}")
    model = load_model(args.model)

    # --- Data generator (no augmentation for eval!)
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = datagen.flow_from_directory(
        args.data_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False
    )

    # --- Resolve class names
    class_names = load_class_names(args.labels, generator.class_indices)
    if not class_names:
        class_names = list(generator.class_indices.keys())

    # --- Predict
    print("🔍 Predicting on evaluation set…")
    t0 = time.time()
    y_prob = model.predict(generator, verbose=1)
    elapsed = time.time() - t0
    n_images = generator.samples
    print(f"⏱️  Inference time: {elapsed:.2f}s for {n_images} imgs "
          f"(avg {elapsed/max(1,n_images):.4f}s/img)")

    y_pred = np.argmax(y_prob, axis=1)
    y_true = generator.classes

    # --- Metrics with more precision
    acc = accuracy_score(y_true, y_pred)
    errors = int((y_true != y_pred).sum())
    print(f"\nOverall accuracy: {acc:.4%}  ({n_images - errors} / {n_images}; errors={errors})")

    # --- Classification report
    report_txt = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print("\n📊 Classification Report:\n")
    print(report_txt)

    (out_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    # --- Confusion matrix (and plot)
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")

    # --- ROC / AUC (micro + macro)
    y_true_onehot = label_binarize(y_true, classes=range(len(class_names)))
    plot_roc_micro_macro(y_true_onehot, y_prob, out_dir / "roc_auc_micro_macro.png")

    # --- Save per-image predictions (optional but handy)
    # Map generator.filenames to predicted labels
    pred_labels = [class_names[i] for i in y_pred]
    with open(out_dir / "predictions.csv", "w", encoding="utf-8") as f:
        f.write("file,true_idx,true_label,pred_idx,pred_label,conf\n")
        for path, ti, pi in zip(generator.filenames, y_true, y_pred):
            conf = float(np.max(y_prob[generator.filenames.index(path)]))
            f.write(f"{path},{ti},{class_names[ti]},{pi},{class_names[pi]},{conf:.6f}\n")

    # --- Optional duplicate scan vs Train/
    if args.dup_scan and Path(args.train_dir).exists():
        scan_duplicate_hashes(Path(args.train_dir), Path(args.data_dir))

    print(f"\n✅ Done. Outputs in: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
