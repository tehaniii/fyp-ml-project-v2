# src/app.py
from __future__ import annotations
"""
Coin Analyzer: classify a coin image, fetch live eBay prices, show Grad-CAM,
and explain results in plain English with metadata-driven context.
Run with:  python -m src.app   (assuming this file is src/app.py)
"""

import os, time, base64, requests, tempfile, json, mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import gradio as gr
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------- local imports --------------------
# LABEL_SYNONYMS may live in different places across your branches; try both.
try:
    from .ebay_module.ebay_pricing import LABEL_SYNONYMS
except Exception:
    try:
        from .ebay_pricing import LABEL_SYNONYMS
    except Exception:
        LABEL_SYNONYMS = {}

from .metadata import get_metadata  # uses normalize_label + prebuilt map

# --- NEW (optional): Numista enrichment import (graceful if unavailable) ---
try:
    from .numista_api import get_coin_details  # /types/{id}
except Exception:
    get_coin_details = None  # fallback if the module/env isn't set

# ================== .env ==================
ROOT = Path(__file__).parent.parent  # project root (one level above /src)
load_dotenv(ROOT / ".env")

# Optional hero banner from ENV (URL) or PATH. If not set, we will embed the local path as base64.
COIN_HERO_URL = (os.getenv("COIN_HERO_URL") or "").strip()
COIN_HERO_PATH = (os.getenv("COIN_HERO_PATH") or r"C:\Users\tehan\Videos\fyp_ml_project_v2\tmp_uploads\coin_bg.jpg").strip()

# ================== eBay helpers ==================
_TOKEN_CACHE = {"access_token": None, "expires_at": 0}
EBAY_ENV = (os.getenv("EBAY_ENV") or "production").lower()
EBAY_OAUTH_URL = (
    "https://api.ebay.com/identity/v1/oauth2/token"
    if EBAY_ENV == "production"
    else "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
)
BROWSE_SEARCH_URL = (
    "https://api.ebay.com/buy/browse/v1/item_summary/search"
    if EBAY_ENV == "production"
    else "https://api.sandbox.ebay.com/buy/browse/v1/item_summary/search"
)
DEFAULT_MP = os.getenv("EBAY_MARKETPLACE") or "EBAY_US"

CURRENCY_SYMBOL = {
    "USD": "$", "EUR": "€", "GBP": "£", "LKR": "Rs ", "AUD": "A$",
    "CAD": "C$", "INR": "₹", "JPY": "¥", "SGD": "S$", "CHF": "CHF ",
}

def _fmt_money(value: float, currency: str | None) -> str:
    sym = CURRENCY_SYMBOL.get(currency or "", "")
    return f"{sym}{value:,.2f} {currency or ''}".strip()

def _basic_auth_header() -> str:
    cid = os.getenv("EBAY_CLIENT_ID")
    csec = os.getenv("EBAY_CLIENT_SECRET")
    if not cid or not csec:
        raise RuntimeError("Missing EBAY_CLIENT_ID / EBAY_CLIENT_SECRET in environment.")
    pair = f"{cid}:{csec}".encode("utf-8")
    return "Basic " + base64.b64encode(pair).decode("ascii")

def _get_access_token(scope: str = "https://api.ebay.com/oauth/api_scope") -> str:
    now = time.time()
    if _TOKEN_CACHE["access_token"] and _TOKEN_CACHE["expires_at"] - now > 120:
        return _TOKEN_CACHE["access_token"]
    headers = {"Authorization": _basic_auth_header(), "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials", "scope": scope}
    resp = requests.post(EBAY_OAUTH_URL, headers=headers, data=data, timeout=20)
    resp.raise_for_status()
    js = resp.json()
    _TOKEN_CACHE["access_token"] = js["access_token"]
    _TOKEN_CACHE["expires_at"] = now + int(js.get("expires_in", 7200))
    return _TOKEN_CACHE["access_token"]

# -------------------- query building --------------------
NOISY = {"replica", "copy", "counterfeit", "fake"}

def _fallback_query(label: str) -> str:
    """Original behavior: clean label and append 'coin'."""
    s = (label or "").replace("_", " ").replace("-", " ")
    words = [w for w in s.split() if w.lower() not in NOISY]
    if "coin" not in [w.lower() for w in words]:
        words.append("coin")
    return " ".join(words)

def _build_queries(label: str) -> list[str]:
    """Return multiple keyword variants for robust matching."""
    if label in LABEL_SYNONYMS and LABEL_SYNONYMS[label]:
        seen = set()
        out = []
        for q in LABEL_SYNONYMS[label]:
            q = (q or "").strip()
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out
    # fallback to a single cleaned query
    return [_fallback_query(label)]

# -------------------- Browse API wrapper --------------------
def get_market_value_from_ebay(pred_label: str, limit: int = 50, marketplace: str = DEFAULT_MP) -> dict:
    """
    Run multiple synonym queries, merge results, dedupe, then compute per-currency stats.
    """
    token = _get_access_token()
    queries = _build_queries(pred_label)

    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": marketplace,
        "Accept": "application/json",
    }

    all_items: List[dict] = []
    seen_ids: set[str] = set()

    for q in queries:
        # 256 = Coins: World (helps reduce irrelevant hits)
        params = {"q": q, "limit": str(limit), "category_ids": "256"}
        try:
            r = requests.get(BROWSE_SEARCH_URL, headers=headers, params=params, timeout=20)
            r.raise_for_status()
            js = r.json()
            items = js.get("itemSummaries", []) or []
        except Exception as e:
            print(f"[eBay] query={q!r} ERROR: {e}")
            items = []

        print(f"[eBay] query={q!r} → {len(items)} items")
        for it in items[:5]:
            print("   •", (it.get("title") or "")[:90], "| price=", it.get("price"))

        for it in items:
            iid = it.get("itemId") or it.get("itemWebUrl") or (it.get("title") or "")
            if iid and iid in seen_ids:
                continue
            if iid:
                seen_ids.add(iid)
            all_items.append(it)

    # ---- aggregate across ALL items ----
    bucket: Dict[str, List[float]] = {}
    samples: List[dict] = []

    for it in all_items:
        p = it.get("price") or {}
        v, cur = p.get("value"), p.get("currency")
        url = it.get("itemWebUrl")
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue

        bucket.setdefault(cur or "", []).append(fv)
        if len(samples) < 5:
            samples.append({
                "title": it.get("title"),
                "price": fv,
                "currency": cur or "",
                "itemWebUrl": url,
            })

    if not any(bucket.values()):
        return {
            "source": "eBay Browse",
            "insufficient": True,
            "n_listings_used": 0,
            "queries_used": queries,
            "query_used": " | ".join(queries),  # backward-compat for UI
            "per_currency": {},
            "sample_listings": [],
        }

    per_currency = {}
    total_used = 0
    for cur, vals in bucket.items():
        vals.sort()
        n = len(vals); total_used += n
        median_val = vals[n//2] if n % 2 else (vals[n//2 - 1] + vals[n//2]) / 2
        per_currency[cur or ""] = {
            "median": round(median_val, 2),
            "low": round(vals[0], 2),
            "high": round(vals[-1], 2),
            "n": n,
        }

    return {
        "source": "eBay Browse",
        "insufficient": False,
        "n_listings_used": total_used,
        "queries_used": queries,
        "query_used": " | ".join(queries),
        "per_currency": per_currency,
        "sample_listings": samples,
    }

# ================== CLASSIFIER (real model) ==================
MODELS_DIR  = ROOT / "models"
LABELS_PATH = ROOT / "class_labels_v3.json"
CANDIDATES  = ["coin_classifier_v3_finetuned.h5"]

_MODEL = None
_CLASS_NAMES: List[str] = []
MODEL_PATH: Optional[str] = None  # chosen model path for both prediction & Grad-CAM

def _select_model_path() -> Optional[str]:
    for n in CANDIDATES:
        p = MODELS_DIR / n
        if p.exists():
            return str(p)
    return None

def _load_class_names() -> List[str]:
    """
    Supports both:
      1) ["label0", "label1", ...]
      2) {"0": "label0", "1": "label1", ...}
    """
    if LABELS_PATH.exists():
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
            if isinstance(data, dict):
                pairs = sorted(((int(k), v) for k, v in data.items()), key=lambda t: t[0])
                return [v for _, v in pairs]
        except Exception as e:
            print("[Labels] Failed to load class_labels.json:", e)
    return []

def _load_tf_model():
    """Load once; also initialize class names (or infer)."""
    global _MODEL, _CLASS_NAMES, MODEL_PATH
    try:
        import tensorflow as tf  # lazy import
        if _MODEL is not None:
            return _MODEL
        MODEL_PATH = _select_model_path()
        if not MODEL_PATH:
            raise FileNotFoundError(
                f"No model file found. Expected one of: {', '.join(CANDIDATES)} in {MODELS_DIR}"
            )
        print(f"[Classifier] Loading model: {MODEL_PATH}")
        _MODEL = tf.keras.models.load_model(MODEL_PATH)

        _CLASS_NAMES = _load_class_names()

        try:
            out_shape = _MODEL.output_shape
            n_out = int(out_shape[-1]) if isinstance(out_shape, tuple) else int(out_shape[0][-1])
        except Exception:
            n_out = len(_CLASS_NAMES) if _CLASS_NAMES else 0

        if n_out <= 1:
            if len(_CLASS_NAMES) < 2:
                _CLASS_NAMES = ["Class_0", "Class_1"]
        else:
            if len(_CLASS_NAMES) != n_out:
                _CLASS_NAMES = [f"class_{i}" for i in range(n_out)]
        print(f"[Classifier] Loaded {len(_CLASS_NAMES)} class names.")
        return _MODEL
    except Exception as e:
        print("[Classifier] model load failed:", e)
        _MODEL = None
        return None

def _prepare_image(img, size=(224, 224)):
    # DO NOT divide by 255 if you use preprocess_input
    arr = np.array(img.convert("RGB").resize(size), dtype=np.float32)
    arr = preprocess_input(arr)      # <- scales to [-1, 1] exactly like training
    return np.expand_dims(arr, axis=0)

def model_predict(image: Image.Image) -> tuple[str, float]:
    mdl = _load_tf_model()
    if mdl is None:
        return "Model_Not_Loaded", 0.0

    x = _prepare_image(image, size=(224, 224))
    preds = mdl.predict(x, verbose=0)
    preds = np.squeeze(preds)

    # Handle binary and multi-class
    if np.ndim(preds) == 0 or np.size(preds) == 1:
        prob1 = float(preds)
        idx = int(prob1 >= 0.5)
        confidence = prob1 if idx == 1 else 1.0 - prob1
    else:
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

    label = _CLASS_NAMES[idx] if 0 <= idx < len(_CLASS_NAMES) else f"class_{idx}"
    return label, confidence

# ================== Grad-CAM helpers (plain-English) ==================
USE_GRADCAM = True
GRADCAM_LAYER_NAME = os.getenv("GRADCAM_LAYER_NAME", "Conv_1")  # adjust to your last conv layer
# Treat very low-confidence predictions as "not a coin" / not a known class
MIN_CONF_FOR_COIN = float(os.getenv("MIN_CONF_FOR_COIN", "0.65"))

# -------------------- Numista-first metadata formatting (NEW) --------------------
def _unwrap(val):
    """Unwrap common Numista shapes: {'text': 'Silver (.917)'} -> 'Silver (.917)'."""
    if isinstance(val, dict):
        if "text" in val and isinstance(val["text"], (str, int, float)):
            return val["text"]
        if "value" in val and isinstance(val["value"], (str, int, float)):
            return val["value"]
    return val

def _deepget(d: dict, *keys, default=None):
    """Return the first present key (case-insensitive) from dict."""
    if not isinstance(d, dict):
        return default
    lower = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        v = lower.get(str(k).lower())
        if v not in (None, "", [], {}):
            return v
    return default

def _coerce_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, dict):
        for k in ("text", "name", "title", "value"):
            if k in x and isinstance(x[k], (str, int, float)):
                return str(x[k])
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, list):
        return ", ".join(_coerce_str(v) for v in x if v)
    return str(x)

def _format_numista_panel(numista_json: dict | None, numista_id: int | str | None, fallback_static: dict | None) -> str:
    """
    Build a clean, Numista-only panel. If Numista data is missing, fallback to a compact static panel.
    """
    if numista_json:
        issuer     = _unwrap(_deepget(numista_json, "issuer", "country", "authority"))
        period     = _unwrap(_deepget(numista_json, "period"))
        ctype      = _unwrap(_deepget(numista_json, "type", "category"))
        year       = _unwrap(_deepget(numista_json, "year", "date", "date_min", "years", "date_range"))
        value      = _unwrap(_deepget(numista_json, "value", "face_value"))
        currency   = _unwrap(_deepget(numista_json, "currency"))
        comp       = _unwrap(_deepget(numista_json, "composition", "metal"))
        weight     = _unwrap(_deepget(numista_json, "weight"))
        diameter   = _unwrap(_deepget(numista_json, "diameter"))
        thickness  = _unwrap(_deepget(numista_json, "thickness"))
        shape      = _unwrap(_deepget(numista_json, "shape"))
        technique  = _unwrap(_deepget(numista_json, "technique", "minting", "method"))
        obv        = _deepget(numista_json, "obverse") or {}
        rev        = _deepget(numista_json, "reverse") or {}
        obv_desc   = _unwrap(_deepget(obv, "description", "desc"))
        obv_let    = _unwrap(_deepget(obv, "legend", "lettering", "text"))
        rev_desc   = _unwrap(_deepget(rev, "description", "desc"))
        rev_let    = _unwrap(_deepget(rev, "legend", "lettering", "text"))
        title      = _unwrap(_deepget(numista_json, "title", "name"))
        # --- FIX: initialize refs before use ---
        refs       = _deepget(numista_json, "references", "refs") or {}

        if isinstance(refs, dict):
            ref_items = []
            for k, v in refs.items():
                if v in (None, "", [], {}):
                    continue
                ref_items.append(f"{k} {v}".strip())
            refs = ref_items
        elif isinstance(refs, list):
            refs = [_coerce_str(x) for x in refs if x]
        else:
            refs = [_coerce_str(refs)] if refs else []

        nnum = f"N#{numista_id}" if numista_id else ""

        lines = []
        lines.append("### Coin details (Numista)")
        if title:     lines.append(f"- **Title:** {_coerce_str(title)}")
        if issuer:    lines.append(f"- **Issuer:** {_coerce_str(issuer)}")
        if period:    lines.append(f"- **Period:** {_coerce_str(period)}")
        if ctype:     lines.append(f"- **Type:** {_coerce_str(ctype)}")
        if year:      lines.append(f"- **Year:** {_coerce_str(year)}")
        if value:     lines.append(f"- **Value:** {_coerce_str(value)}")
        if currency:  lines.append(f"- **Currency:** {_coerce_str(currency)}")
        if comp:      lines.append(f"- **Composition:** {_coerce_str(comp)}")
        if weight:    lines.append(f"- **Weight:** {_coerce_str(weight)}")
        if diameter:  lines.append(f"- **Diameter:** {_coerce_str(diameter)}")
        if thickness: lines.append(f"- **Thickness:** {_coerce_str(thickness)}")
        if shape:     lines.append(f"- **Shape:** {_coerce_str(shape)}")
        if technique: lines.append(f"- **Technique:** {_coerce_str(technique)}")
        if nnum:      lines.append(f"- **Numista #:** {nnum}")
        if refs:
            lines.append(f"- **References:** " + "; ".join(refs))

        if obv_desc or obv_let:
            lines.append("\n**Obverse**")
            if obv_desc: lines.append(_coerce_str(obv_desc))
            if obv_let:  lines.append(f"\n*Lettering:* {_coerce_str(obv_let)}")
        if rev_desc or rev_let:
            lines.append("\n**Reverse**")
            if rev_desc: lines.append(_coerce_str(rev_desc))
            if rev_let:  lines.append(f"\n*Lettering:* {_coerce_str(rev_let)}")

        return "\n".join(lines).strip()

    # ------- Fallback (no Numista available): compact static panel -------
    meta = fallback_static or {}
    if not meta:
        return "_No details found for this coin._"

    basics = [
        ("Denomination", meta.get("Denomination")),
        ("Alloy", meta.get("Alloy") or meta.get("Metal")),
        ("Diameter", meta.get("Diameter")),
        ("Thickness", meta.get("Thickness")),
        ("Weight", meta.get("Weight")),
        ("Shape", meta.get("Shape")),
        ("Edge", meta.get("Edge")),
        ("Mint", meta.get("Mint")),
        ("Die axis", meta.get("DieAxis")),
        ("Condition", meta.get("Condition")),
    ]
    out = ["### Coin details"]
    for k, v in basics:
        if v: out.append(f"- **{k}:** {v}")
    if meta.get("Obverse"): out.append(f"\n**Obverse**\n{meta['Obverse']}")
    if meta.get("Reverse"): out.append(f"\n**Reverse**\n{meta['Reverse']}")
    if meta.get("HistoricalContext"): out.append(f"\n**History (short):** {meta['HistoricalContext']}")
    if meta.get("References"):
        refs = meta["References"]
        out.append(f"\n**References:** " + ("; ".join(refs) if isinstance(refs, list) else str(refs)))
    if meta.get("Note"): out.append(f"\n**Note:** {meta['Note']}")
    return "\n".join(out)

def _compose_overlay_from_cam(base_bgr: np.ndarray, cam: np.ndarray, alpha: float) -> Image.Image:
    """Build a PIL overlay image from base BGR and a 0..1 heatmap with chosen alpha."""
    h, w = base_bgr.shape[:2]
    hm = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_u8 = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_TURBO)
    out = np.clip(hm_color * float(alpha) + base_bgr, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

def _run_gradcam(img_pil: Image.Image, pred_label: str, conf: float, meta: dict,
                 show_annotated: bool, overlay_alpha: float, hot_quantile: float) -> tuple[Image.Image, str]:
    if not USE_GRADCAM:
        return img_pil, "_Grad-CAM disabled._"

    mdl = _load_tf_model()
    if mdl is None:
        return img_pil, "Grad-CAM is unavailable because the model isn't loaded."

    try:
        # Use the new coarse, region-based annotator
        from .gradcam_utils import generate_gradcam, annotate_attention_regions

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img_pil.save(tmp.name)
            img_path = tmp.name

        result = generate_gradcam(
            model=mdl,
            img_path=img_path,
            pred_index=None,
            layer_name=GRADCAM_LAYER_NAME,
            img_size=(224, 224),
            pred_label=pred_label,
            confidence=conf,
            return_array=True,
            explain=True,
            meta=meta
        )

        if not isinstance(result, tuple):
            raise RuntimeError("generate_gradcam returned an unexpected value.")

        if len(result) == 4:
            cam, out_path, expl, base_bgr = result
        elif len(result) == 3:
            cam, out_path, expl = result
            base_bgr = cv2.imread(img_path)
        else:
            raise RuntimeError("generate_gradcam returned an unexpected tuple length.")

        if show_annotated:
            ann_bgr = annotate_attention_regions(
                base_bgr, cam, meta,
                pred_label=pred_label,
                grid=(3, 3),
                topk=3,
                hot_quantile=hot_quantile
            )
            overlay = Image.fromarray(cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB))
        else:
            overlay = _compose_overlay_from_cam(base_bgr, cam, overlay_alpha)

        return overlay, expl if expl is not None else ""
    except Exception as e:
        print("[GradCAM] failed:", e)
        return img_pil, f"Grad-CAM failed: {e}"

# --- quick coin-likeness check used on upload ---
def _looks_like_coin(pil_img: Image.Image) -> bool:
    """
    Heuristic: must have a large circular rim near the center AND a strong edge ring.
    This is stricter than the earlier version so faces/balls/background circles won't pass.
    """
    arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    h, w = gray.shape
    min_r = int(min(h, w) * 0.35)
    max_r = int(min(h, w) * 0.49)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.35,
        minDist=min(h, w) // 2,
        param1=150, param2=40,
        minRadius=min_r, maxRadius=max_r
    )
    if circles is None:
        return False

    x, y, r = circles[0][0].astype(int)
    center_dist = np.hypot(x - w / 2, y - h / 2) / max(r, 1)
    area_ratio = (np.pi * r * r) / float(h * w)
    if not (center_dist < 0.5 and 0.25 < area_ratio < 0.7):
        return False

    edges = cv2.Canny(gray, 80, 160)
    ring_mask = np.zeros_like(edges)
    cv2.circle(ring_mask, (x, y), r, 255, 3)
    ring_hits = int((edges & ring_mask).sum())
    ring_total = int(ring_mask.sum())
    rim_strength = ring_hits / max(ring_total, 1)

    return rim_strength > 0.12

def on_image_upload(image: Image.Image) -> str:
    """
    Status when a file is chosen.
    Conservative gate: show the green success ONLY if the photo looks like a coin
    AND/OR the quick model confidence is above MIN_CONF_FOR_COIN. Otherwise, warn.
    """
    if image is None:
        return ""
    w, h = image.size
    if min(w, h) < 96:
        return "⚠️ Image loaded, but it is very small; results may be unreliable."

    looks_round = _looks_like_coin(image)

    try:
        _, conf = model_predict(image)
    except Exception:
        conf = 0.0

    if conf >= MIN_CONF_FOR_COIN or looks_round:
        return "✅ Image looks like a coin. Click **Analyze**."
    else:
        return "⚠️ This doesn’t look like a coin. Please upload a **coin image** (centered coin side, good lighting)."

def clear_all():
    """Reset input + all outputs + status line."""
    return None, "", "", "", None, "", "⚠️ Upload a clear, centered coin photo to proceed."

# ================== UI (Gradio) ==================
def analyze(image: Image.Image, show_annotated: bool, overlay_alpha: float, hot_q: int):
    # Basic input check
    if image is None:
        status = "❌ Please upload a coin image first."
        return "", "", "", None, "", status

    # 1) prediction
    label, conf = model_predict(image)

    # Confidence gate
    if conf < MIN_CONF_FOR_COIN:
        status = (f"❌ This image does not appear to match any known coin class "
                  f"(confidence {conf*100:.1f}%). Please upload a clearer, centered photo with good lighting.")
        return "", "", "", None, "", status

    # 2) market value
    try:
        mv = get_market_value_from_ebay(label)
    except Exception as e:
        mv = {
            "source": "eBay Browse",
            "insufficient": True,
            "n_listings_used": 0,
            "error": str(e),
            "queries_used": _build_queries(label),
            "query_used": _fallback_query(label),
            "per_currency": {},
            "sample_listings": [],
        }

    # 3) market value markdown
    lines = []
    lines.append("### Market Value")
    q_str = mv.get("query_used") or " / ".join(mv.get("queries_used", []) or [])
    lines.append(f"**Source:** {mv.get('source','eBay Browse')}  •  **Query:** `{q_str}`")
    if mv.get("insufficient"):
        lines.append("> We couldn’t find enough recent listings for this exact query.")
    else:
        per_cur = mv.get("per_currency", {})
        for cur, stats in per_cur.items():
            lines.append(
                f"- **{cur or 'Unknown'}** — "
                f"Median: {_fmt_money(stats['median'], cur)}  •  "
                f"Range: {_fmt_money(stats['low'], cur)} – {_fmt_money(stats['high'], cur)}  •  "
                f"Used listings: {stats['n']}"
            )
        samples = mv.get("sample_listings", [])
        if samples:
            lines.append("\n**Live listings:**")
            for s in samples:
                p = _fmt_money(s['price'], s.get('currency'))
                url = s.get("itemWebUrl") or "#"
                title = (s.get("title") or "").strip()
                lines.append(f"- [{title}]({url}) — {p}")
    mv_md = "\n".join(lines)

    # 4) metadata panel (Numista-first; fallback to static if absent)
    meta = get_metadata(label) or {}
    numista_json = None
    numista_id = None

    for key in ("numista_id", "NumistaId", "numistaTypeId"):
        if key in meta and meta[key]:
            numista_id = meta[key]
            break

    if get_coin_details and numista_id:
        try:
            numista_json = get_coin_details(numista_id)
        except Exception as _e:
            numista_json = None  # fail silent; we'll fallback

    meta_md = _format_numista_panel(numista_json, numista_id, fallback_static=meta)

    # 5) Grad-CAM image + explanation (metadata-aware)
    hot_quantile = max(50, min(99, int(hot_q))) / 100.0
    cam_img, cam_expl = _run_gradcam(image, label, conf, meta, show_annotated, overlay_alpha, hot_quantile)

    # 6) prediction block
    pred_md = f"**Prediction:** {label}  \n**Confidence:** {round(conf*100, 2)}%"

    ready_status = "✅ Ready — you can re-run **Analyze** or upload another image."
    return pred_md, mv_md, meta_md, cam_img, cam_expl, ready_status

# -------------------- theme & layout CSS --------------------
# >>> PASTE BELOW THIS LINE if you only want to update the hero background logic <<<
def _data_url_from_path(path_str: str) -> str:
    """Return CSS url('data:...') for a local image path so browsers can render it."""
    try:
        p = (path_str or "").strip().strip('"')
        if p and os.path.exists(p):
            mime = mimetypes.guess_type(p)[0] or "image/jpeg"
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            return f"url('data:{mime};base64,{b64}')"
    except Exception as e:
        print("[Hero] failed to embed local image:", e)
    return ""

# Build the hero background: prefer COIN_HERO_URL; else embed local COIN_HERO_PATH; else gradient
if COIN_HERO_URL:
    HERO_INLINE = f"url('{COIN_HERO_URL}')"
else:
    HERO_INLINE = _data_url_from_path(COIN_HERO_PATH) or "linear-gradient(120deg, #2e2a20, #1b1f27)"

CUSTOM_CSS = f"""
/* ---- Coin-collecting site aesthetic (dark, elegant, consistent cards) ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Cormorant+Garamond:wght@500;600;700&display=swap');

:root {{
  --bg: #f7f4ee;             /* page parchment */
  --text: #1b1b1b;
  --muted: #6b6b6b;
  --gold: #c7a437;
  --gold-2: #b08d22;
  --link: #0d6efd;
  --radius: 16px;
  --shadow: 0 14px 32px rgba(0,0,0,.18);
  --panel-1: #12161f;        /* card gradient top */
  --panel-2: #0c1119;        /* card gradient bottom */
  --panel-stroke: rgba(212,175,55,.25);
}}

* {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif; }}
h1,h2,h3,.brand-title {{ font-family: 'Cormorant Garamond', Georgia, 'Times New Roman', serif; letter-spacing:.3px; }}

html, body {{ overflow-x: hidden; }}
body, .gradio-container {{ background: var(--bg) !important; color: var(--text); }}

a {{ color: var(--link); }}

/* ====== Navbar ====== */
.header {{
  background:#111;
  border-bottom:2px solid rgba(199,164,55,0.6);
  padding:14px 18px;
}}
.navbar {{ display:flex; align-items:center; justify-content:space-between; max-width:1200px; margin:0 auto; }}
.brand {{ display:flex; align-items:center; gap:12px; }}
.brand-logo {{
  width:40px; height:40px; border-radius:50%;
  background: radial-gradient(circle at 30% 30%, #f2e3b0, #b1862e 60%, #58431a);
  box-shadow: 0 0 0 2px rgba(212,175,55,.25), inset 0 0 20px rgba(0,0,0,.25);
}}
.brand-title {{ font-size:28px; color:#fff; text-transform:uppercase; font-weight:600; }}
.navlinks a {{ margin-left:18px; color:#ddd; text-decoration:none; font-weight:600; font-size:14px; }}
.navlinks a:hover {{ color:#fff; }}

/* ====== Hero image directly under navbar ====== */
.hero-wrap {{ width:100%; }}
.hero {{
  width:100%;
  height:260px;
  background: {HERO_INLINE};
  background-size: cover;
  background-position: center;
  position: relative;
}}
.hero::after {{
  content:"";
  position:absolute; inset:0;
  background: linear-gradient(180deg, rgba(255,255,255,.12), rgba(255,255,255,.08));
}}

/* ====== Thin heading strip under hero (no weird sliders) ====== */
.strip {{
  max-width:1200px; margin: 10px auto 6px; padding:16px;
  background: rgba(255,255,255,.96);
  border:1px solid #eadfbe; border-radius:12px;
  text-align:center;
}}
.strip .title {{ font-weight:700; letter-spacing:.15px; color:#222; }}
.strip .sub   {{ margin-top:6px; color:#555; font-size:14px; }}

/* Remove inner component scrollbars that looked like 'sliders' under the strip */
.gradio-container .gradio-html, .gradio-container .gr-row, .gradio-container .gr-column {{ overflow: visible !important; }}
.gradio-container .gradio-html *::-webkit-scrollbar {{ width:0; height:0; }}
.gradio-container .gradio-html * {{ scrollbar-width: none; }}

/* ====== Global section width ====== */
.section {{ max-width:1200px; margin: 12px auto 8px; }}

/* ====== Consistent dark cards across ALL tabs ====== */
.card {{
  background: linear-gradient(180deg, var(--panel-1), var(--panel-2));
  border:1px solid var(--panel-stroke);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding:18px;
  color:#f2f2f2;
}}
.card h3 {{ color: var(--gold); margin:0 0 10px 0; font-weight:600; }}

/* Make image areas in cards dark & consistent */
.card .image-container,
.card .image-container .image-editor,
.card .image-container .image-preview,
.card .image-container .edit-area {{
  background:#0e141d !important;
  border:1px dashed rgba(255,255,255,.08) !important;
  color:#f2f2f2 !important;
}}
.card .label-wrap label {{ color: var(--gold) !important; font-weight: 600; }}
img.gr-image, .gradio-container .image-container {{ border-radius:14px; overflow:hidden; }}

/* Controls panel inside Identify tab */
.controls {{
  background:#0f131b; border:1px solid rgba(212,175,55,0.22); border-radius:12px; padding:14px;
}}
hr.divider {{
  border:none; height:1px;
  background: linear-gradient(90deg, rgba(212,175,55,0.0), rgba(212,175,55,0.5), rgba(212,175,55,0.0));
  margin:10px 0 14px 0;
}}

/* Tabs – clean underline, no animation */
.tab-nav button, .tab-nav .tabitem {{ color:#555; font-weight:600; }}
.tab-nav .selected, .tab-nav button[aria-selected="true"] {{
  color: #111 !important;
  border-bottom: 2px solid var(--gold) !important;
}}

/* Numeric alignment for prices */
.price, .gr-markdown code {{ font-variant-numeric: tabular-nums; }}
.smallmuted {{ color: #bfc6d2; font-size:13px; }}

/* Full-width footer */
.footer {{
  width:100%;
  background:#111; color:#ddd;
  font-size:12.5px;
  border-top:2px solid rgba(199,164,55,0.6);
  text-align:center;
  margin:18px 0 8px;
  padding:14px 0;
}}

/* prettier error toast */
.gradio-container [role="alert"] {{
  border:1px solid rgba(199,164,55,.65) !important;
  background:#181818 !important;
  color:#f3f3f3 !important;
  box-shadow:0 10px 26px rgba(0,0,0,.35);
  border-left:4px solid #d84c4c !important;
  border-radius:12px !important;
}}
.gradio-container [role="alert"] * {{ color:#f3f3f3 !important; }}

/* visually indicate disabled Analyze button */
#analyze-btn[disabled] {{
  opacity: .55;
  cursor: not-allowed !important;
}}
"""

# -------------------- UI Layout --------------------
with gr.Blocks(title="Coin Collecting • Analyzer", css=CUSTOM_CSS, theme=gr.themes.Default()) as demo:
    # ===== Navbar =====
    gr.HTML(
        """
        <div class="header">
          <div class="navbar">
            <div class="brand">
              <div class="brand-logo"></div>
              <div class="brand-title">Coin Collecting</div>
            
          </div>
        </div>
        """
    )

    # ===== Hero image =====
    gr.HTML('<div class="hero-wrap"><div class="hero"></div></div>')

    # ===== Thin heading strip under hero =====
    gr.HTML(
        """
        <div class="strip">
          <div class="title">Identify Coins • Visual Explainability • Market Insights</div>
          <div class="sub">Upload a coin, see what the model looks at, and browse recent price ranges.</div>
        </div>
        """
    )

    # ===== Main content constrained width =====
    with gr.Column(elem_classes=["section"]):
        with gr.Tabs(elem_classes=["tab-nav"]):
            # --------------- Tab 1: Identify ---------------
            with gr.TabItem("Identify", id=0):
                gr.HTML('<a id="identify"></a>')
                with gr.Row():
                    # Left column: Upload & controls
                    with gr.Column(scale=5):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### Upload a Photo")
                            inp = gr.Image(type="pil", label=None, height=340)
                            status_msg = gr.Markdown("", elem_classes=["smallmuted"], elem_id="status-msg")

                            gr.HTML("""
                                <script>
                                (function () {
                                  const statusEl = document.getElementById('status-msg');
                                  const analyzeBtn = document.getElementById('analyze-btn');

                                  const setAnalyzeEnabled = (enabled) => {
                                    if (!analyzeBtn) return;
                                    if (enabled) {
                                      analyzeBtn.removeAttribute('disabled');
                                    } else {
                                      analyzeBtn.setAttribute('disabled', 'true');
                                    }
                                  };

                                  setAnalyzeEnabled(false);

                                  const evaluateStatus = () => {
                                    if (!statusEl) return;
                                    const txt = (statusEl.innerText || "").trim();
                                    if (txt.startsWith('✅')) {
                                      setAnalyzeEnabled(true);
                                    } else if (txt.startsWith('⚠️') || txt.startsWith('❌')) {
                                      setAnalyzeEnabled(false);
                                    }
                                  };

                                  let clearTimer = null;
                                  const scheduleClearIfSuccess = () => {
                                    const txt = (statusEl.innerText || "").trim();
                                    if (!txt) return;
                                    if (txt.startsWith('✅')) {
                                      if (clearTimer) clearTimeout(clearTimer);
                                      clearTimer = setTimeout(() => {
                                        statusEl.innerText = "";
                                      }, 3000);
                                    }
                                  };

                                  const obs = new MutationObserver(() => {
                                    evaluateStatus();
                                    scheduleClearIfSuccess();
                                  });
                                  if (statusEl) {
                                    obs.observe(statusEl, { childList: true, subtree: true, characterData: true });
                                  }
                                  evaluateStatus();
                                })();
                                </script>
                            """)

                            gr.Markdown("<hr class='divider'/>")
                            with gr.Row(elem_classes=["controls"]):
                                show_ann = gr.Checkbox(value=True, label="Annotate key features on the heatmap")
                                overlay_alpha = gr.Slider(minimum=0.20, maximum=0.85, value=0.40, step=0.05,
                                                          label="Heatmap opacity", info="Lower = subtler overlay; higher = stronger overlay")
                                hot_q = gr.Slider(minimum=80, maximum=98, value=90, step=1,
                                                  label="Hot-spot quantile (%)", info="Top attention band considered 'hot'")
                            analyze_btn = gr.Button("Analyze", variant="primary", elem_id="analyze-btn")
                            clear_btn = gr.Button("Clear")

                    # Right column: Heatmap + explanation
                    with gr.Column(scale=7):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### Heatmap & Explanation")
                            out_cam = gr.Image(label=None, height=360, show_download_button=True)
                            gr.Markdown("<hr class='divider'/>")
                            out_cam_expl = gr.Markdown(label=None)

            # --------------- Tab 2: Market Value ---------------
            with gr.TabItem("Market Value", id=1):
                gr.HTML('<a id="market"></a>')
                with gr.Row():
                    with gr.Column(scale=12):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### Recent Listings & Price Ranges")
                            out_mv = gr.Markdown()
                            gr.Markdown("<div class='smallmuted'>Estimates reflect recent listings and can vary by grade, rarity, and provenance.</div>")

            # --------------- Tab 3: Encyclopedia ---------------
            with gr.TabItem("Coin Encyclopedia", id=2):
                gr.HTML('<a id="encyclopedia"></a>')
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### Identification")
                            out_pred = gr.Markdown()
                    with gr.Column(scale=8):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### Please find the relevant coin details below")
                            out_meta = gr.Markdown()

    # ===== Single full-width footer =====
    gr.HTML("<div class='footer'>© Coin Collecting — tools for numismatists and enthusiasts</div>")

    # ===== Wire actions =====
    analyze_btn.click(
        fn=analyze,
        inputs=[inp, show_ann, overlay_alpha, hot_q],
        outputs=[out_pred, out_mv, out_meta, out_cam, out_cam_expl, status_msg]
    )
    inp.change(fn=on_image_upload, inputs=inp, outputs=status_msg)
    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[inp, out_pred, out_mv, out_meta, out_cam, out_cam_expl, status_msg]
    )

if __name__ == "__main__":
    demo.launch()
