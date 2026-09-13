"""
Face detection + embedding.

Two interchangeable backends behind one interface:

  1. InsightFace (buffalo_l)  -> RetinaFace detector + ArcFace 512-d embeddings.
     Stronger accuracy. Runs on ONNX Runtime.
  2. OpenCV (YuNet + SFace)   -> 128-d embeddings, ships inside opencv-python.
     Slightly weaker, but installs with zero pain.

The abstraction exists because install reliability matters more than the last
2% of accuracy when you are on a deadline. Both backends return L2-normalised
embeddings, so cosine similarity is a plain dot product and the rest of the
system does not care which one is active.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(os.environ.get("FRS_MODEL_DIR", Path.home() / ".frs_models"))


@dataclass
class Face:
    """One detected face."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    embedding: np.ndarray  # L2-normalised, float32
    det_score: float
    # Raw detector row, needed by the OpenCV backend for aligned cropping.
    _raw: np.ndarray | None = None

    @property
    def area(self) -> int:
        return int(self.bbox[2] * self.bbox[3])


def _l2_normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def _download(url: str, dest: Path, min_bytes: int = 500_000) -> None:
    """Download a model file, refusing git-lfs pointer stubs."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[models] downloading {dest.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())

    size = tmp.stat().st_size
    if size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{dest.name} downloaded as only {size} bytes - this is a git-lfs "
            f"pointer, not the real model. Download it manually from:\n  {url}"
        )
    tmp.rename(dest)
    print(f"[models] saved {dest} ({size / 1e6:.1f} MB)")


class FaceEngine:
    """Interface both backends implement."""

    name: str = "base"
    dim: int = 0
    #: Sensible starting threshold. The real value comes from evaluate.py.
    default_threshold: float = 0.5

    def detect(self, image_bgr: np.ndarray) -> list[Face]:
        raise NotImplementedError

    def embed_largest(self, image_bgr: np.ndarray) -> Face | None:
        """Return only the biggest face, which is almost always the subject."""
        faces = self.detect(image_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: f.area)


class InsightFaceEngine(FaceEngine):
    name = "insightface:buffalo_l (RetinaFace + ArcFace)"
    dim = 512
    # ArcFace cosine similarity for the same identity typically lands well
    # above 0.4; different identities cluster near 0.0-0.2.
    default_threshold = 0.40

    def __init__(self, det_size: int = 640) -> None:
        from insightface.app import FaceAnalysis  # imported lazily

        self.app = FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"]
        )
        # ctx_id=-1 forces CPU.
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def detect(self, image_bgr: np.ndarray) -> list[Face]:
        out = []
        for f in self.app.get(image_bgr):
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            out.append(
                Face(
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    embedding=_l2_normalise(f.embedding),
                    det_score=float(f.det_score),
                )
            )
        return out


class OpenCVEngine(FaceEngine):
    name = "opencv:YuNet + SFace"
    dim = 128
    # OpenCV's own documented cosine threshold for SFace.
    default_threshold = 0.363

    YUNET_URL = (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )
    SFACE_URL = (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    )

    def __init__(self, score_threshold: float = 0.7) -> None:
        det_path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
        rec_path = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
        _download(self.YUNET_URL, det_path, min_bytes=100_000)
        _download(self.SFACE_URL, rec_path, min_bytes=1_000_000)

        self.detector = cv2.FaceDetectorYN.create(
            str(det_path), "", (320, 320), score_threshold, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")

    def detect(self, image_bgr: np.ndarray) -> list[Face]:
        h, w = image_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, raw = self.detector.detect(image_bgr)
        if raw is None:
            return []

        out = []
        for row in raw:
            # alignCrop uses the 5 landmarks to warp the face to a canonical
            # pose before embedding. Skipping alignment costs real accuracy.
            aligned = self.recognizer.alignCrop(image_bgr, row)
            feat = self.recognizer.feature(aligned)
            x, y, bw, bh = [int(v) for v in row[:4]]
            out.append(
                Face(
                    bbox=(x, y, bw, bh),
                    embedding=_l2_normalise(feat),
                    det_score=float(row[-1]),
                    _raw=row,
                )
            )
        return out


_ENGINE: FaceEngine | None = None


def get_engine(prefer: str | None = None) -> FaceEngine:
    """
    Return a cached engine. Tries InsightFace, falls back to OpenCV.

    Set FRS_BACKEND=opencv (or pass prefer='opencv') to skip InsightFace
    entirely - useful when the install is fighting you.
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    prefer = prefer or os.environ.get("FRS_BACKEND", "auto")

    if prefer in ("auto", "insightface"):
        try:
            _ENGINE = InsightFaceEngine()
            print(f"[engine] using {_ENGINE.name}")
            return _ENGINE
        except Exception as e:
            if prefer == "insightface":
                raise
            print(f"[engine] InsightFace unavailable ({e}); falling back to OpenCV")

    _ENGINE = OpenCVEngine()
    print(f"[engine] using {_ENGINE.name}")
    return _ENGINE


def read_image(data: bytes) -> np.ndarray | None:
    """Decode raw bytes (an uploaded file) into a BGR image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
