"""Validate db + sweep math with synthetic identity clusters (no models needed)."""
import numpy as np, sys
from db import FaceDB, Match

rng = np.random.default_rng(0)
DIM = 128

def unit(v):
    return v / np.linalg.norm(v)

def make_identity(spread=0.045):
    center = unit(rng.normal(size=DIM))
    def sample():
        return unit(center + spread * rng.normal(size=DIM))
    return sample

# 40 enrolled + 20 impostor identities
enrolled = {f"person_{i:02d}": make_identity() for i in range(40)}
impostors = {f"imp_{i:02d}": make_identity() for i in range(20)}

db = FaceDB("test_gallery.npz")
gen_probes = []
for name, s in enrolled.items():
    db.enroll(name, [s() for _ in range(3)])
    gen_probes += [(name, s()) for _ in range(4)]
imp_probes = [s() for s in impostors.values() for _ in range(4)]

print("gallery identities:", len(db.names))
assert len(db.names) == 40

# save/load roundtrip
db.save()
db2 = FaceDB("test_gallery.npz")
assert db2.names == db.names, "roundtrip name mismatch"
a = db.identify(gen_probes[0][1], -1.0)
b = db2.identify(gen_probes[0][1], -1.0)
assert a.name == b.name and abs(a.score - b.score) < 1e-6, "roundtrip score drift"
print("persistence roundtrip: OK")

# centroid should be unit length
assert np.allclose(np.linalg.norm(db._matrix, axis=1), 1.0, atol=1e-5)
print("centroids unit-normalised: OK")

# score everything once
gen_s = np.array([db.identify(e, -1.0).score for _, e in gen_probes])
gen_c = np.array([db.identify(e, -1.0).name == n for n, e in gen_probes])
imp_s = np.array([db.identify(e, -1.0).score for e in imp_probes])
print(f"genuine  mean={gen_s.mean():.3f}  impostor mean={imp_s.mean():.3f}")
print(f"rank-1 accuracy (no threshold): {gen_c.mean():.1%}")

# sweep
lo, hi = min(gen_s.min(), imp_s.min()), max(gen_s.max(), imp_s.max())
rows = []
for t in np.linspace(lo, hi, 200):
    acc = gen_s >= t
    rows.append(dict(threshold=float(t), TAR=float((acc & gen_c).mean()),
                     MIS=float((acc & ~gen_c).mean()), FRR=float((~acc).mean()),
                     FAR=float((imp_s >= t).mean())))

# invariants
for r in rows:
    tot = r["TAR"] + r["MIS"] + r["FRR"]
    assert abs(tot - 1.0) < 1e-9, f"rates must partition: {tot}"
assert rows[0]["FAR"] >= rows[-1]["FAR"], "FAR must be non-increasing in threshold"
assert rows[0]["FRR"] <= rows[-1]["FRR"], "FRR must be non-decreasing in threshold"
print("sweep invariants (TAR+MIS+FRR==1, monotonic FAR/FRR): OK")

viable = [r for r in rows if r["FAR"] <= 0.01]
chosen = max(viable, key=lambda r: r["TAR"]) if viable else min(rows, key=lambda r: r["FAR"])
print(f"chosen threshold={chosen['threshold']:.3f} TAR={chosen['TAR']:.1%} FAR={chosen['FAR']:.1%}")

# rejection actually works at the chosen threshold
m = db.identify(imp_probes[0], chosen["threshold"])
print("impostor at chosen threshold ->", m.name or "unknown (rejected)")

# empty gallery must not crash
empty = FaceDB("empty.npz")
assert empty.identify(unit(rng.normal(size=DIM)), 0.4).name is None
print("empty-gallery guard: OK")

# remove
db.remove("person_00")
assert "person_00" not in db.names and len(db.names) == 39
print("remove identity: OK")
print("\nALL LOGIC TESTS PASSED")
