import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from metadata import get_metadata
from gradcam_utils import generate_gradcam
from live_value import get_live_value   # returns dict with estimated_median/range_*/sample_listings

# ---------------- CONFIG ----------------
MODEL_PATH_V1 = "models/coin_classifier.h5"
MODEL_PATH_V2 = "models/coin_classifier_v2.h5"
LABELS_PATH = "class_labels.json"
IMG_SIZE = (224, 224)


# ---------------- I/O utils ----------------
def _load_labels(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if isinstance(labels, dict) and "labels" in labels:
        return labels["labels"]
    return labels


def _load_model(version: str = "v2"):
    version = (version or "v2").lower()
    if version == "v1":
        path = MODEL_PATH_V1
    else:
        path = MODEL_PATH_V2
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    print(f"✅ Loaded model: {path}")
    return load_model(path)


def _preprocess_image(img_path: str, target_size=IMG_SIZE) -> np.ndarray:
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # If your model was trained on MobileNetV2 preprocessing, keep this:
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return x


# ---------------- Inference ----------------
def classify_image(img_path: str, model_version: str = "v2") -> dict:
    labels = _load_labels(LABELS_PATH)
    model = _load_model(model_version)
    x = _preprocess_image(img_path)

    preds = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    pred_label = labels[top_idx] if top_idx < len(labels) else f"label_{top_idx}"

    print("\n== Prediction ==")
    print(f"📌 Label: {pred_label}")
    print(f"🔢 Confidence: {confidence:.3f}")

    # ---- Grad-CAM (optional visual explanation) ----
    try:
        print("\n== Grad-CAM ==")
        generate_gradcam(
            model=model,
            img_path=img_path,
            pred_index=top_idx,
            img_size=IMG_SIZE,
            layer_name="Conv_1",  # last conv in MobileNetV2
            pred_label=pred_label,
            confidence=confidence,
        )
    except Exception as e:
        print("⚠️ Grad-CAM failed (continuing):", e)

    # ---- Live value estimation (eBay first, Numista fallback) ----
    print("\n== Market Value (eBay/Numista) ==")
    value = get_live_value(pred_label)

    if not value:
        print("⚠️ No market value data found.")
        return {"label": pred_label, "confidence": confidence, "value": None}

    # Friendly printout consistent with get_live_value structure
    print(f"Name:   {value.get('name')}")
    print(f"Source: {value.get('source')}")

    med = value.get("estimated_median")
    lo = value.get("range_low")
    hi = value.get("range_high")
    n_used = value.get("n_listings_used")
    q = value.get("query")

    if med is not None:
        print(f"Median: {med:.2f}")
        if lo is not None and hi is not None:
            print(f"Range:  {lo:.2f} – {hi:.2f}")
        if n_used is not None:
            print(f"Used listings: {n_used}")
    else:
        print("(insufficient data for a robust estimate)")

    if q:
        print(f"Query:  {q}")

    sample = value.get("sample_listings", []) or []
    for s in sample[:5]:
        ttl = s.get("title", "")
        pr = s.get("price")
        cur = s.get("currency", "")
        url = s.get("url", "")
        if pr is not None:
            try:
                print(f" - {ttl} | {float(pr):.2f} {cur} | {url}")
            except Exception:
                print(f" - {ttl} | {pr} {cur} | {url}")
        else:
            print(f" - {ttl} | {cur} | {url}")

    return {
        "label": pred_label,
        "confidence": confidence,
        "value": value,
    }


# ---------------- Main ----------------
if __name__ == "__main__":
    img_path = input("🔍 Enter path to coin image: ").strip()
    version = input("🧪 Choose model version ('v1' or 'v2'): ").strip() or "v2"
    classify_image(img_path, model_version=version)
