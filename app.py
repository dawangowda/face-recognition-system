"""
FastAPI service for face enrollment and identification.

    uvicorn app:app --reload

Interactive docs at http://127.0.0.1:8000/docs - useful for the demo, since
you can upload images straight from the browser with no client to write.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response

from db import FaceDB, load_threshold
from face_engine import get_engine, read_image

GALLERY_PATH = Path("gallery.npz")
engine = None
db: FaceDB | None = None
THRESHOLD = 0.4


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load models and gallery once at boot, not per request."""
    global engine, db, THRESHOLD
    engine = get_engine()
    db = FaceDB(GALLERY_PATH)
    THRESHOLD = load_threshold(fallback=engine.default_threshold)
    print(f"[app] threshold={THRESHOLD} | {len(db.names)} identities enrolled")
    yield


app = FastAPI(
    title="Face Recognition Identification System",
    description="Enroll individuals, identify probe faces, reject unknowns.",
    version="1.0.0",
    lifespan=lifespan,
)

# Optional Prometheus instrumentation. The service runs fine without it.
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )

    IDENTIFY_TOTAL = Counter(
        "frs_identify_total", "Identification attempts", ["outcome"]
    )
    ENROLL_TOTAL = Counter("frs_enroll_total", "Enrollment requests")
    LATENCY = Histogram("frs_request_seconds", "Request latency", ["endpoint"])
    METRICS = True
except ImportError:  # pragma: no cover
    METRICS = False


@app.get("/")
def root() -> dict:
    """Landing route - points a first-time visitor at the interactive docs."""
    return {
        "service": "Face Recognition Identification System",
        "docs": "/docs",
        "endpoints": [
            "GET  /health",
            "POST /enroll",
            "POST /identify",
            "GET  /people",
            "DELETE /people/{name}",
            "GET  /metrics",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "backend": engine.name if engine else None,
        "embedding_dim": engine.dim if engine else None,
        "threshold": THRESHOLD,
        "gallery": db.stats() if db else None,
    }


@app.post("/enroll")
async def enroll(
    name: str = Form(..., description="Identity label"),
    files: list[UploadFile] = File(..., description="One or more images"),
) -> dict:
    """
    Register a person from one or more images.

    Multiple images are strongly preferred: they are averaged into a centroid,
    which suppresses per-image lighting and pose noise. Images where no face is
    detected are reported back rather than silently dropped, so the caller knows
    the enrollment is thinner than they intended.
    """
    t0 = time.perf_counter()
    name = name.strip()
    if not name:
        raise HTTPException(400, "name must not be empty")

    embeddings, accepted, rejected = [], [], []
    for f in files:
        img = read_image(await f.read())
        if img is None:
            rejected.append({"file": f.filename, "reason": "unreadable image"})
            continue
        faces = engine.detect(img)
        if not faces:
            rejected.append({"file": f.filename, "reason": "no face detected"})
            continue
        if len(faces) > 1:
            # Enrolling from a group photo silently binds the wrong person's
            # face to this label, so refuse rather than guess.
            rejected.append(
                {"file": f.filename, "reason": f"{len(faces)} faces; need exactly 1"}
            )
            continue
        embeddings.append(faces[0].embedding)
        accepted.append(f.filename)

    if not embeddings:
        raise HTTPException(
            422, {"error": "no usable faces found", "rejected": rejected}
        )

    total = db.enroll(name, embeddings)
    db.save()
    if METRICS:
        ENROLL_TOTAL.inc()
        LATENCY.labels("enroll").observe(time.perf_counter() - t0)

    return {
        "name": name,
        "accepted": accepted,
        "rejected": rejected,
        "samples_for_identity": total,
        "gallery_size": len(db.names),
    }


@app.post("/identify")
async def identify(
    file: UploadFile = File(...),
    threshold: float | None = Query(
        None, description="Override the calibrated threshold"
    ),
    all_faces: bool = Query(False, description="Identify every face, not just largest"),
) -> dict:
    """
    Identify faces in an image.

    Any face whose best cosine similarity falls below the threshold is returned
    as `unknown` - the system declines to guess rather than forcing a match to
    its nearest neighbour.
    """
    t0 = time.perf_counter()
    t = threshold if threshold is not None else THRESHOLD

    img = read_image(await file.read())
    if img is None:
        raise HTTPException(400, "could not decode image")

    faces = engine.detect(img)
    if not faces:
        if METRICS:
            IDENTIFY_TOTAL.labels("no_face").inc()
        return {"threshold": t, "faces_detected": 0, "results": []}

    if not all_faces:
        faces = [max(faces, key=lambda f: f.area)]

    results = []
    for face in faces:
        m = db.identify(face.embedding, threshold=t)
        if METRICS:
            IDENTIFY_TOTAL.labels("known" if m.is_known else "unknown").inc()
        results.append(
            {
                "identity": m.name or "unknown",
                "is_known": m.is_known,
                "similarity": round(m.score, 4),
                "margin_over_runner_up": round(m.margin, 4),
                "runner_up": m.runner_up,
                "bbox": {
                    "x": face.bbox[0],
                    "y": face.bbox[1],
                    "w": face.bbox[2],
                    "h": face.bbox[3],
                },
                "detection_score": round(face.det_score, 4),
            }
        )

    if METRICS:
        LATENCY.labels("identify").observe(time.perf_counter() - t0)

    return {
        "threshold": t,
        "faces_detected": len(faces),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "results": results,
    }


@app.get("/people")
def people() -> dict:
    return db.stats()


@app.delete("/people/{name}")
def delete_person(name: str) -> dict:
    if not db.remove(name):
        raise HTTPException(404, f"'{name}' is not enrolled")
    db.save()
    return {"deleted": name, "gallery_size": len(db.names)}


@app.get("/metrics")
def metrics() -> Response:
    if not METRICS:
        return JSONResponse(
            {"error": "prometheus_client not installed"}, status_code=501
        )
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)