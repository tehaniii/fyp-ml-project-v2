"""
Grad-CAM utilities for the Coin Analyzer — coin-collector friendly.

Fix in this version:
- OpenCV cannot render curly quotes; all labels are ASCII-only and sanitized.
- `_sanitize_label` normalizes Unicode quotes to ASCII and strips non-ASCII.
- Same explainability features: coin mask, 3×3 % contributions, curated obverse hints,
  optional OCR mapped to curated labels (never raw OCR text), mirrored textual reasons.

`generate_gradcam(..., return_array=True, explain=True)` returns:
    (heatmap01, out_path, explanation_markdown, base_bgr)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------- Optional OCR (graceful if not installed) ---------
try:
    import pytesseract  # type: ignore
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

# --------- Optional metadata lookup (graceful if not available) ---------
try:
    from .metadata import get_metadata  # type: ignore
except Exception:
    get_metadata = None  # graceful fallback


# ============================== Curated hints ==============================

# All-caps keywords -> human-friendly captions (ASCII ONLY)
_OBV_HINTS: List[Tuple[str, str]] = [
    ("CEYLON",       "inscription"),
    ("ONE STIVER",   "inscription"),
    ("STIVER",       "denomination"),
    ("ELEPHANT",     "elephant figure"),
    ("VICTORIA",     "queen's legend"),
    ("GEORGIUS",     "king's legend"),
    ("GEORGE",       "king's legend"),
    ("WREATH",       "wreath"),
    ("COAT OF ARMS", "coat of arms"),
    ("SHIELD",       "shield emblem"),
    ("CROWN",        "crown"),
    ("DATE",         "date area"),
    ("1815",         "date area"),
    ("1821",         "date area"),
    ("NUMERAL",      "large numeral"),
]

# Lower-case keywords to build a short “likely used” list from metadata (ASCII)
_FEATURE_MAP: List[Tuple[str, str]] = [
    ("king",       "king's figure"),
    ("bust",       "portrait"),
    ("crown",      "crown"),
    ("elephant",   "elephant"),
    ("conch",      "conch shell"),
    ("lotus",      "lotus motif"),
    ("wreath",     "wreath"),
    ("script",     "script / legend"),
    ("legend",     "script / legend"),
    ("nagari",     "Nagari legend"),
    ("date",       "date area"),
    ("ceylon",     "'CEYLON' inscription"),
    ("one stiver", "'ONE STIVER' inscription"),
    ("one rixdollar", "'ONE RIXDOLLAR' inscription"),
    ("rixdollar",  "rixdollar legend"),
    ("value",      "value inscription"),
    ("lamp",       "lamp"),
    ("jasmine",    "jasmine flower"),
    ("sri",        "'SRI' legend"),
    ("tusk",       "elephant tusk"),
    ("trident",    "trident emblem"),
    ("shield",     "shield motif"),
    ("sinhala",    "Sinhala legend"),
]


# =============================== Utilities ================================

def _sanitize_label(s: str) -> str:
    """
    Make a label drawable by cv2.putText:
    - Normalize curly quotes to ASCII.
    - Remove any question marks and non-ASCII bytes.
    - Collapse repeated spaces and trim.
    """
    if not s:
        return s
    # normalize quotes
    s = (s.replace("‘", "'")
           .replace("’", "'")
           .replace("“", '"')
           .replace("”", '"'))
    # drop any leftover non-ascii chars
    s = s.encode("ascii", "ignore").decode("ascii")
    # remove '?', collapse whitespace
    s = s.replace("?", " ").replace("  ", " ").strip()
    return s


def _obverse_hints_from_meta(meta: dict, pred_label: str) -> List[str]:
    """
    Build short, human captions using ONLY obverse clues.
    Looks in Obverse + high-level fields; ignores Reverse entirely.
    """
    text = " ".join([
        str(meta.get("Obverse", "")),
        str(meta.get("HistoricalContext", "")),
        str(meta.get("Denomination", "")),
        str(pred_label or ""),
    ]).upper()

    out: List[str] = []
    for key, nice in _OBV_HINTS:
        if key in text and nice not in out:
            out.append(nice)

    if not out:
        out = ["inscription lettering", "central ", "date / value area"]

    return [_sanitize_label(x) for x in out[:4]]


def _ocr_hint_to_curated(bgr: np.ndarray) -> Optional[str]:
    """
    Use OCR to *select* one curated label.
    We never show raw OCR output; we only map to curated captions.
    """
    if not _HAS_OCR:
        return None
    try:
        txt = pytesseract.image_to_string(bgr, config="--psm 7").upper()
    except Exception:
        return None

    for key, nice in _OBV_HINTS:
        if key in txt:
            return _sanitize_label(nice)
    return None


def _features_from_metadata(meta: Optional[Dict]) -> List[str]:
    if not meta:
        return []
    text = " ".join([str(meta.get("Obverse", "")), str(meta.get("Reverse", ""))]).lower()
    if any(ch.isdigit() for ch in text) and "date" not in text:
        text += " date"
    seen: List[str] = []
    for kw, pretty in _FEATURE_MAP:
        if kw in text and pretty not in seen:
            seen.append(pretty)
        if len(seen) >= 5:
            break
    return [_sanitize_label(x) for x in seen]


def _history_snippet(meta: Optional[Dict]) -> str:
    if not meta:
        return ""
    s = str(meta.get("HistoricalContext", "")).strip()
    if not s:
        return ""
    s = s.encode("ascii", "ignore").decode("ascii")
    return (s[:237].rstrip() + "...") if len(s) > 240 else s


# =========================== Image / Model helpers ========================

def _load_and_preprocess(img_path: str, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Load image and return (original_bgr, preprocessed_batch)."""
    pil = image.load_img(img_path, target_size=size)
    arr = image.img_to_array(pil)  # RGB float32
    batch = np.expand_dims(arr, axis=0)
    batch = preprocess_input(batch)
    orig = cv2.imread(img_path)
    if orig is None:
        orig = cv2.cvtColor(np.uint8(arr), cv2.COLOR_RGB2BGR)
    return orig, batch


def _gradcam_map(model, batch: np.ndarray, layer_name: str, class_idx: Optional[int]) -> Tuple[np.ndarray, int, float]:
    """
    Compute a Grad-CAM heatmap (0..1). Works for binary or multi-class heads.
    Returns: (heatmap01, chosen_class_idx, chosen_confidence)
    """
    conv_layer = model.get_layer(layer_name)
    grad_model = Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(batch, training=False)
        preds = tf.squeeze(preds)

        if preds.shape.rank == 0 or int(tf.size(preds)) == 1:
            prob1 = float(preds.numpy())
            idx = int(prob1 >= 0.5) if class_idx is None else int(class_idx)
            conf = prob1 if idx == 1 else (1.0 - prob1)
            loss = preds if idx == 1 else (1.0 - preds)
        else:
            if class_idx is None:
                idx = int(tf.argmax(preds).numpy())
            else:
                idx = int(class_idx)
            conf = float(preds[idx].numpy())
            loss = preds[idx]

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))  # GAP over H,W
    cam = tf.reduce_sum(conv_out * tf.reshape(weights, (-1, 1, 1, weights.shape[-1])), axis=-1)
    cam = tf.nn.relu(cam)[0]
    cam = cam / (tf.reduce_max(cam) + tf.keras.backend.epsilon())
    return cam.numpy().astype("float32"), idx, conf


def _overlay_heatmap(base_bgr: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    """Resize heatmap, colorize and overlay. (JET for saved JPG; UI may use TURBO.)"""
    h, w = base_bgr.shape[:2]
    hm = cv2.resize(heatmap01, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_u8 = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)  # BGR
    out = np.clip(hm_color * alpha + base_bgr, 0, 255).astype(np.uint8)
    return out


# ============================ Regions / Mask ==============================

def _hot_mask(heatmap01: np.ndarray, quantile: float = 0.90) -> np.ndarray:
    th = float(np.quantile(heatmap01, quantile))
    return heatmap01 >= th


def _region_sentence(mask: np.ndarray) -> str:
    H, W = mask.shape
    if H == 0 or W == 0:
        return "the system did not find a strong focal area"

    regions = {
        "top": mask[: H // 2, :].mean(),
        "bottom": mask[H // 2 :, :].mean(),
        "left": mask[:, : W // 2].mean(),
        "right": mask[:, W // 2 :].mean(),
        "center": mask[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4].mean(),
    }
    top2 = [r for r, _ in sorted(regions.items(), key=lambda kv: kv[1], reverse=True)[:2] if regions[r] > 0]
    if not top2:
        return "attention was spread out over the coin"
    if len(top2) == 1:
        return f"the system focused mostly on the {top2[0]}"
    return f"the system focused on the {top2[0]} and {top2[1]}"


# ============================ Drawing helpers =============================

def _rect(img, x0, y0, x1, y1, *, fill=(0, 0, 0), alpha=0.35, border=(255, 255, 255), thickness=2):
    """Draw a semi-transparent filled rectangle with a crisp border."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), fill, -1)
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.rectangle(out, (x0, y0), (x1, y1), border, thickness)
    return out


# =================== Region-based, curated-legend overlay ==================

def annotate_attention_regions(
    base_bgr: np.ndarray,
    heatmap01: np.ndarray,
    meta: Optional[Dict],
    *,
    pred_label: str = "",
    grid: Tuple[int, int] = (4, 4),
    topk: int = 4,
    hot_quantile: float = 0.90,  # retained for compatibility; ranking uses sum()
) -> np.ndarray:
    """
    Highlight general regions (grid cells) where attention is strongest.
    Clean, aligned, and readable (not pixel-perfect).
    Returns a BGR uint8 image with heatmap + soft region boxes + labels.
    """
    h, w = base_bgr.shape[:2]

    # Base overlay (TURBO is friendlier for mild color blindness)
    hm = cv2.resize(heatmap01, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_u8 = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_TURBO)
    out = np.clip(hm_color * 0.40 + base_bgr, 0, 255).astype(np.uint8)

    # ---- Limit attention to the coin (reduce background) ----
    gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w)//2,
        param1=100, param2=30,
        minRadius=int(min(h, w)*0.35), maxRadius=int(min(h, w)*0.49)
    )
    mask = np.zeros((h, w), dtype=np.float32)
    if circles is not None:
        x, y, r = circles[0][0].astype(int)
        cv2.circle(mask, (x, y), r, 1.0, -1)
    else:
        mask[:] = 1.0

    hm = hm * mask
    hm = hm / (hm.max() + 1e-8)
    total_cam = float(hm.sum() + 1e-8)

    # ---- Score grid cells by SUM of attention (contribution) ----
    gy, gx = grid
    cell_h = h // gy
    cell_w = w // gx

    cells = []
    for r in range(gy):
        for c in range(gx):
            y0, y1 = r * cell_h, h if r == gy - 1 else (r + 1) * cell_h
            x0, x1 = c * cell_w, w if c == gx - 1 else (c + 1) * cell_w
            score = float(hm[y0:y1, x0:x1].sum())
            cells.append((score, (x0, y0, x1, y1), r, c))

    cells.sort(key=lambda t: t[0], reverse=True)
    hot_cells = [c for c in cells[:topk] if c[0] > 0.0]

    # Friendly labels from obverse-only metadata
    labels = _obverse_hints_from_meta(meta or {}, pred_label)

    # ---- Draw boxes + tags (with % contribution) ----
    for i, (score, (x0, y0, x1, y1), r, c) in enumerate(hot_cells):
        out = _rect(out, x0, y0, x1, y1, fill=(0, 0, 0), alpha=0.28, border=(255, 255, 255), thickness=2)

        pct = 100.0 * score / total_cam
        nice = labels[i] if i < len(labels) else "inscription"

        # Optional OCR to override the hint (if a clear word is detected)
        roi = base_bgr[max(0, y0):y1, max(0, x0):x1]
        maybe = _ocr_hint_to_curated(roi)
        if maybe:
            nice = maybe

        tag = _sanitize_label(f"{nice} ({pct:.1f}%)")

        pad = 6
        # scale font size based on image width (so small images don’t get huge labels)
        font_scale = max(0.35, min(0.6, w / 800))   # between 0.35 and 0.6 depending on width
        thickness = 1

        (tw, th_txt), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        bx0, by0 = x0 + 5, max(y0 + th_txt + 6, th_txt + 6)
        bx1, by1 = min(bx0 + tw + 10, w - 1), min(by0 + th_txt + 6, h - 1)

        # draw background + text
        cv2.rectangle(out, (bx0, by0 - th_txt - 4), (bx1, by1), (0, 0, 0), -1)
        cv2.putText(out, tag, (bx0 + 5, by0 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


    return out


# ========================== Legacy contour overlay =========================

def _largest_hot_components(mask: np.ndarray, max_k: int = 3):
    mask_u8 = (mask.astype(np.uint8) * 255)
    num, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    comps = []
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        cy, cx = centroids[lab][1], centroids[lab][0]
        comps.append((area, (int(cx), int(cy)), lab))
    comps.sort(reverse=True, key=lambda t: t[0])
    return comps[:max_k]


def _nice_feature_list(meta: Optional[Dict]) -> List[str]:
    feats = _features_from_metadata(meta)
    return feats[:3] if feats else []


def annotate_heatmap_with_features(
    base_bgr: np.ndarray,
    heatmap01: np.ndarray,
    meta: Optional[Dict],
    *,
    hot_quantile: float = 0.90,
) -> np.ndarray:
    h, w = base_bgr.shape[:2]
    hm = cv2.resize(heatmap01, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_u8 = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_TURBO)
    overlay = np.clip(hm_color * 0.40 + base_bgr, 0, 255).astype(np.uint8)

    th = float(np.quantile(hm, hot_quantile))
    hot = (hm >= th).astype(np.uint8)
    hot_u8 = (hot * 255).astype(np.uint8)

    contours, _ = cv2.findContours(hot_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    comps = _largest_hot_components(hot, max_k=3)
    features = _nice_feature_list(meta)
    for i, (area, (cx, cy), _) in enumerate(comps):
        tag = _sanitize_label(features[i] if i < len(features) else "key detail")
        cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), thickness=-1)
        tx, ty = min(cx + 12, w - 10), max(cy - 8, 20)
        cv2.line(overlay, (cx, cy), (tx, ty), (255, 255, 255), 2)
        (tw, th_txt), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        pad = 6
        box_tl = (tx, max(ty - th_txt - pad, 0))
        box_br = (min(tx + tw + 2 * pad, w - 1), min(ty + pad, h - 1))
        cv2.rectangle(overlay, box_tl, box_br, (0, 0, 0), thickness=-1)
        cv2.putText(overlay, tag, (tx + pad, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return overlay


# ============================ Explanation builder =========================

def _build_explanation(
    heatmap01: np.ndarray,
    pred_label: str,
    confidence: float,
    meta: Optional[Dict],
) -> str:
    mask = _hot_mask(heatmap01, quantile=0.92)
    focus_pct = round(float(mask.mean() * 100.0), 1)
    where = _region_sentence(mask)
    feats = _features_from_metadata(meta)
    hist = _history_snippet(meta)

    # Top reasons via 3×3 grid contributions (matches overlay)
    gy, gx = 3, 3
    H, W = heatmap01.shape
    cell_h = H // gy
    cell_w = W // gx
    contrib = []
    for r in range(gy):
        for c in range(gx):
            y0, y1 = r * cell_h, H if r == gy - 1 else (r + 1) * cell_h
            x0, x1 = c * cell_w, W if c == gx - 1 else (c + 1) * cell_w
            contrib.append(float(heatmap01[y0:y1, x0:x1].sum()))
    total = max(sum(contrib), 1e-8)
    order = np.argsort(contrib)[::-1][:3]
    hints = _obverse_hints_from_meta(meta or {}, pred_label)
    reasons = [
        f"- Reason {i+1}: {_sanitize_label(hints[i] if i < len(hints) else 'inscription')} "
        f"(~{(contrib[idx]/total)*100:.1f}%)"
        for i, idx in enumerate(order)
    ]

    lines: List[str] = [
        "### What the colors mean",
        f"In short: the colors show where the system looked most to decide this is **{_sanitize_label(pred_label)}**.",
        "- Red/orange = strongest attention; yellow/green = some attention.",
        f"- Focus: the system concentrated on about **{focus_pct}%** of the coin's surface.",
        f"- Where: {where}.",
        *reasons,
        (
            f"- What it likely used: notes for this type mention **{', '.join(feats[:3])}**."
            if feats else
            "- What it likely used: overall shapes, lettering, and borders typical of this coin."
        ),
        f"- How sure? About **{round(confidence*100, 2)}%** certain.",
    ]
    if hist:
        lines += ["", f"**Background:** {hist}"]

    if focus_pct < 5:
        lines.append("_Tip: the system focused on a very small spot; try softer light or reduce glare._")
    elif focus_pct > 70:
        lines.append("_Tip: attention is spread widely; try a closer, sharper photo for clearer details._")

    return "\n".join(lines)


# =============================== Public API ===============================

def generate_gradcam(
    model,
    img_path: str,
    pred_index: Optional[int] = None,
    layer_name: str = "",
    img_size: Tuple[int, int] = (224, 224),
    pred_label: str = "",
    confidence: float = 0.0,
    *,
    return_array: bool = False,
    explain: bool = False,
    meta: Optional[Dict] = None,
):
    """
    Generate a Grad-CAM overlay and (optionally) a plain-English explanation.

    Returns when return_array=True:
      (heatmap01, output_path, explanation_markdown, base_bgr)
    """
    # 1) image & forward
    base_bgr, batch = _load_and_preprocess(img_path, img_size)
    heatmap01, _, conf_calc = _gradcam_map(model, batch, layer_name, pred_index)
    conf_for_text = float(confidence) if confidence and confidence > 0 else float(conf_calc)

    # 2) overlay (saved for convenience)
    overlay = _overlay_heatmap(base_bgr, heatmap01, alpha=0.40)
    label_text = _sanitize_label(pred_label or "Prediction")
    cv2.putText(
        overlay, f"{label_text} ({conf_for_text:.2%})",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )

    # 3) save to disk
    out_dir = os.path.join("outputs", "gradcams")
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, f"{base_name}_gradcam.jpg")
    cv2.imwrite(out_path, overlay)
    print(f"✅ Grad-CAM saved to: {out_path}")

    # 4) explanation (optional)
    explanation = None
    if explain:
        if meta is None and get_metadata is not None and pred_label:
            try:
                meta = get_metadata(pred_label)  # type: ignore
            except Exception:
                meta = None
        explanation = _build_explanation(heatmap01, label_text or "this coin", conf_for_text, meta)

    if return_array:
        return heatmap01, out_path, explanation, base_bgr
    return None
