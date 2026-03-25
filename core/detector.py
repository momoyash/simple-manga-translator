"""
core/detector.py  --  simple-manga-translator

Finds speech bubbles in manga pages using two methods:
  1. Deep learning model (primary)
  2. Contour analysis / CV (fallback + supplement)

Debug usage:
  python core/detector.py -i page.png -o debug.png
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = Path(__file__).parent.parent / "models" / "detector.onnx"
INPUT_SIZE  = 640
MIN_SCORE   = 0.30

# Detection classes
BUBBLE      = 0
TEXT_BUBBLE = 1
TEXT_FREE   = 2


# ── Data ───────────────────────────────────────────────────────────────────────

@dataclass
class Region:
    """A detected region on a manga page."""
    pts:        np.ndarray        # (4,2) corner points
    contour:    Optional[np.ndarray]
    score:      float
    area:       int
    method:     str = "cv"        # "model" or "cv"
    kind:       int = BUBBLE


# ── Model loading ──────────────────────────────────────────────────────────────

_session = None


def _load():
    global _session
    if _session is not None:
        return True
    if not MODEL_PATH.exists():
        print(f"[detector] Model not found at {MODEL_PATH} — falling back to CV")
        return False
    try:
        import onnxruntime as ort
        _session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        size_mb = MODEL_PATH.stat().st_size // 1_000_000
        print(f"[detector] Model loaded ({size_mb} MB)")
        return True
    except Exception as e:
        print(f"[detector] Could not load model: {e} — falling back to CV")
        return False


# ── Deep model detection ───────────────────────────────────────────────────────

def _prep_image(img: np.ndarray) -> Tuple[np.ndarray, float]:
    h, w    = img.shape[:2]
    scale   = INPUT_SIZE / max(h, w)
    nw, nh  = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas  = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    rgb     = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[np.newaxis], scale


def _to_regions(labels, boxes, scores, scale, h, w) -> List[Region]:
    out = []
    for label, box, score in zip(labels, boxes, scores):
        if score < MIN_SCORE:
            continue
        label = int(label)
        x1, y1, x2, y2 = (v / scale for v in box.astype(float))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        bw, bh  = x2 - x1, y2 - y1
        if bw < 5 or bh < 5:
            continue
        pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.int32)
        out.append(Region(pts=pts, contour=None, score=float(score),
                          area=int(bw * bh), method="model", kind=label))
    return out


def scan_deep(img: np.ndarray, min_score: float = MIN_SCORE) -> List[Region]:
    """Run the deep learning model on a manga page."""
    if not _load():
        return []
    h, w    = img.shape[:2]
    tensor, scale = _prep_image(img)
    labels, boxes, scores = _session.run(None, {
        "images":            tensor,
        "orig_target_sizes": np.array([[h, w]], dtype=np.int64),
    })
    return _to_regions(labels[0], boxes[0], scores[0], scale, h, w)


# ── CV contour detection ───────────────────────────────────────────────────────

def scan_cv(img: np.ndarray, debug: bool = False) -> List[Region]:
    """Find bubble-shaped regions using contour analysis."""
    h, w       = img.shape[:2]
    page_area  = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )
    closed = cv2.morphologyEx(
        cv2.bitwise_not(binary),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=2
    )

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    found = []
    for i, c in enumerate(contours):
        if hierarchy[0][i][3] != -1:
            continue
        area = cv2.contourArea(c)
        if not (page_area * 0.001 < area < page_area * 0.25):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bh == 0 or not (0.2 < bw / bh < 5.0):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area == 0 or area / hull_area < 0.5:
            continue
        if bw * bh == 0 or area / (bw * bh) < 0.3:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        brightness = cv2.mean(gray, mask=mask)[0]
        if brightness < 180:
            continue
        corners = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.int32)
        score = (area / hull_area) * 0.4 + (area / (bw * bh)) * 0.3 + (brightness / 255) * 0.3
        found.append(Region(pts=corners, contour=c, score=float(score),
                            area=int(area), method="cv"))

    if debug:
        print(f"[detector/cv] {len(found)} regions found")

    found.sort(key=lambda r: r.area, reverse=True)
    return found


# ── Merge helpers ──────────────────────────────────────────────────────────────

def _box(pts): return pts[:,0].min(), pts[:,1].min(), pts[:,0].max(), pts[:,1].max()

def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    inter = (ix2-ix1) * (iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0

def subtract(new: List[Region], existing: List[Region], thresh=0.3) -> List[Region]:
    """Return regions in `new` that don't overlap with any in `existing`."""
    if not new or not existing:
        return new
    boxes = [_box(r.pts) for r in existing]
    return [r for r in new if not any(_iou(_box(r.pts), b) > thresh for b in boxes)]


# ── Main entry point ───────────────────────────────────────────────────────────

def find(img: np.ndarray, min_score: float = MIN_SCORE,
         debug: bool = False) -> List[Region]:
    """
    Find all speech bubbles in a manga page.
    Combines deep model + CV for maximum coverage.
    """
    deep = scan_deep(img, min_score=min_score)
    cv   = scan_cv(img, debug=debug)

    if not deep:
        if debug: print("[detector] No model results — using CV only")
        return cv

    extra = subtract(cv, deep)
    if debug:
        print(f"[detector] deep={len(deep)}  cv_extra={len(extra)}")

    return deep + extra


# ── Debug visualizer ───────────────────────────────────────────────────────────

_PALETTE = {
    "model/bubble":      (0,   200,   0),
    "model/text_bubble": (0,   160, 255),
    "model/text_free":   (255, 100,   0),
    "cv":                (255, 200,   0),
}
_LABELS = ["bubble", "text_bubble", "text_free"]


def draw(img: np.ndarray, regions: List[Region],
         save_to: Optional[str] = None) -> np.ndarray:
    """Draw detected regions on the image. Saves if save_to is given."""
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    for r in regions:
        key   = f"{r.method}/{_LABELS[r.kind]}" if r.method == "model" else "cv"
        color = _PALETTE.get(key, (200, 200, 200))
        cv2.polylines(out, [r.pts.reshape(-1,1,2).astype(np.int32)], True, color, 2)
        cx, cy = int(r.pts[:,0].mean()), int(r.pts[:,1].mean())
        cv2.putText(out, f"{_LABELS[r.kind]} {r.score:.2f}", (cx-30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    for i, line in enumerate([
        "GREEN=bubble  ORANGE=text_bubble  BLUE=text_free",
        "YELLOW=cv_supplement"
    ]):
        cv2.putText(out, line, (10, 18+i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)

    if save_to:
        cv2.imwrite(save_to, out)
        print(f"[detector] Saved: {save_to}")
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Debug bubble detection on a manga page")
    p.add_argument("-i", "--input",  required=True)
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--score", type=float, default=MIN_SCORE)
    args = p.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print(f"[error] Cannot load: {args.input}"); exit(1)

    regions = find(img, min_score=args.score, debug=True)
    deep = sum(1 for r in regions if r.method == "model")
    cv   = sum(1 for r in regions if r.method == "cv")
    print(f"\nTotal: {len(regions)}  (model={deep}, cv={cv})")
    for i, r in enumerate(regions):
        print(f"  [{i+1}] {r.method}/{_LABELS[r.kind]}  area={r.area}  score={r.score:.3f}")

    out = args.output or str(Path(args.input).stem) + "_debug.png"
    draw(img, regions, save_to=out)
