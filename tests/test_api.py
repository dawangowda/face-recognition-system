"""Smoke-test the FastAPI endpoints with a stub engine (no model weights needed)."""
import io, sys, numpy as np, cv2
import face_engine
from face_engine import Face

# Deterministic stub: the image's mean colour decides the identity cluster.
class StubEngine(face_engine.FaceEngine):
    name = "stub"; dim = 64; default_threshold = 0.5
    def detect(self, img):
        if img is None or img.mean() < 5:   # "black" image = no face
            return []
        seed = int(img[:, :, 0].mean())
        rng = np.random.default_rng(seed)
        v = rng.normal(size=64); v = v / np.linalg.norm(v)
        n = 2 if img.shape[1] > 300 else 1   # wide image = two faces
        return [Face(bbox=(10*i, 10, 50, 50), embedding=v if i == 0 else -v,
                     det_score=0.99) for i in range(n)]

face_engine._ENGINE = StubEngine()
face_engine.get_engine = lambda prefer=None: face_engine._ENGINE

from pathlib import Path
Path("gallery.npz").unlink(missing_ok=True)
Path("threshold.json").unlink(missing_ok=True)

from fastapi.testclient import TestClient
import app as appmod
client = TestClient(appmod.app)

def png(val, w=100):
    img = np.full((100, w, 3), val, np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()

with client:
    r = client.get("/health"); assert r.status_code == 200, r.text
    print("GET /health ->", r.json()["status"], "| threshold", r.json()["threshold"])

    # identify on empty gallery must not 500
    r = client.post("/identify", files={"file": ("a.png", png(50), "image/png")})
    assert r.status_code == 200, r.text
    print("identify on empty gallery ->", r.json()["results"][0]["identity"])

    # enroll Alice from 2 images of the same cluster
    r = client.post("/enroll", data={"name": "Alice"},
        files=[("files", ("a1.png", png(50), "image/png")),
               ("files", ("a2.png", png(50), "image/png"))])
    assert r.status_code == 200, r.text
    print("POST /enroll ->", r.json()["samples_for_identity"], "samples")

    client.post("/enroll", data={"name": "Bob"},
        files=[("files", ("b1.png", png(150), "image/png"))])

    # known face
    r = client.post("/identify", files={"file": ("q.png", png(50), "image/png")}).json()
    assert r["results"][0]["identity"] == "Alice", r
    print("identify Alice ->", r["results"][0]["identity"],
          f"sim={r['results'][0]['similarity']}")

    # unknown face must be rejected
    r = client.post("/identify", files={"file": ("u.png", png(200), "image/png")}).json()
    assert r["results"][0]["identity"] == "unknown", r
    print("identify stranger ->", r["results"][0]["identity"],
          f"sim={r['results'][0]['similarity']}")

    # no face detected
    r = client.post("/identify", files={"file": ("k.png", png(0), "image/png")}).json()
    assert r["faces_detected"] == 0
    print("no-face image ->", r["faces_detected"], "faces")

    # multi-face enrollment must be refused
    r = client.post("/enroll", data={"name": "Carol"},
        files=[("files", ("group.png", png(50, w=400), "image/png"))])
    assert r.status_code == 422, r.text
    print("group-photo enroll -> correctly refused (422)")

    # all_faces on a group photo
    r = client.post("/identify?all_faces=true",
        files={"file": ("g.png", png(50, w=400), "image/png")}).json()
    assert r["faces_detected"] == 2
    print("group identify (all_faces) ->", [x["identity"] for x in r["results"]])

    # corrupt upload
    r = client.post("/identify", files={"file": ("x.png", b"notanimage", "image/png")})
    assert r.status_code == 400
    print("corrupt upload -> 400 as expected")

    r = client.get("/people").json(); print("GET /people ->", r["per_identity"])
    r = client.delete("/people/Bob"); assert r.status_code == 200
    assert client.delete("/people/Nobody").status_code == 404
    print("DELETE /people ->", r.json())

    m = client.get("/metrics")
    print("GET /metrics ->", m.status_code,
          "frs_identify_total present:", "frs_identify_total" in m.text)

Path("gallery.npz").unlink(missing_ok=True)
print("\nALL API TESTS PASSED")
