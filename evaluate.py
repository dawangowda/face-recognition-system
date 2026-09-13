"""
Threshold calibration and open-set evaluation.

This is the core of the assignment. Detection and embedding are library calls;
choosing the matching threshold is the part that requires judgement, so it is
measured rather than guessed.

Protocol
--------
Identities with enough images are split into two disjoint groups:

  * ENROLLED  - a few images become the gallery centroid, the rest are probes.
                Correct behaviour: return the right name.
  * IMPOSTORS - never enrolled at all. Every image is a probe.
                Correct behaviour: return "unknown".

Holding impostor identities out entirely is what makes this an *open-set*
evaluation. A closed-set test (every probe belongs to someone enrolled) cannot
measure the rejection mechanism, and would make the system look far better than
it is in deployment, where most faces seen are strangers.

Metrics swept across candidate thresholds
-----------------------------------------
  TAR  true accept rate      enrolled probe -> correct name, above threshold
  FAR  false accept rate     impostor probe -> any name, above threshold
  FRR  false reject rate     enrolled probe -> wrongly rejected as unknown
  MIS  misidentification     enrolled probe -> *wrong* name, above threshold

Operating point
---------------
Access control is asymmetric: admitting a stranger is a security failure, while
rejecting a known person is an inconvenience they can retry. The threshold is
therefore chosen as the lowest value whose FAR stays under a target budget
(default 1%), maximising TAR subject to that constraint - not the value that
maximises raw accuracy.

Usage
-----
    python evaluate.py                      # auto-downloads LFW via scikit-learn
    python evaluate.py --data-dir ./faces   # your own  person_name/*.jpg  folders
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from face_engine import get_engine

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def load_lfw_paths() -> dict[str, list[Path]]:
    """
    Trigger scikit-learn's LFW download, then read the full-resolution files
    off disk.

    sklearn's in-memory arrays are downscaled and pre-cropped, which starves the
    detector. The funneled JPEGs it unpacks are 250x250 and run through the real
    detect -> align -> embed path, so the evaluation exercises the same code as
    production.
    """
    from sklearn.datasets import get_data_home, fetch_lfw_people

    print("[data] fetching LFW (~200 MB on first run, then cached)...")
    fetch_lfw_people(min_faces_per_person=3, download_if_missing=True)

    root = Path(get_data_home()) / "lfw_home"
    for candidate in ("lfw_funneled", "lfw"):
        d = root / candidate
        if d.is_dir():
            return _scan_dir(d)
    raise RuntimeError(f"could not locate unpacked LFW images under {root}")


def _scan_dir(root: Path) -> dict[str, list[Path]]:
    people: dict[str, list[Path]] = defaultdict(list)
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = sorted(
            f for f in person_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS
        )
        if imgs:
            people[person_dir.name] = imgs
    if not people:
        raise RuntimeError(f"no person_name/*.jpg folders found under {root}")
    return dict(people)


# --------------------------------------------------------------------------
# embedding extraction
# --------------------------------------------------------------------------

def embed_paths(engine, paths: list[Path]) -> list[np.ndarray]:
    """Embed the largest detected face in each image. Undetected images drop out."""
    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        face = engine.embed_largest(img)
        if face is not None:
            out.append(face.embedding)
    return out


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def build_split(
    people: dict[str, list[Path]],
    n_identities: int,
    impostor_frac: float,
    min_images: int,
    enroll_per_identity: int,
    seed: int,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    rng = np.random.default_rng(seed)
    eligible = sorted(k for k, v in people.items() if len(v) >= min_images)
    if len(eligible) < 4:
        raise RuntimeError(
            f"only {len(eligible)} identities have >= {min_images} images; "
            "lower --min-images or supply more data"
        )

    rng.shuffle(eligible)
    eligible = eligible[:n_identities]
    n_imp = max(1, int(len(eligible) * impostor_frac))

    impostor_names = eligible[:n_imp]
    enrolled_names = eligible[n_imp:]
    if not enrolled_names:
        raise RuntimeError("no identities left to enroll; lower --impostor-frac")

    enrolled = {n: people[n] for n in enrolled_names}
    impostors = {n: people[n] for n in impostor_names}
    return enrolled, impostors


def evaluate(args) -> dict:
    engine = get_engine()
    print(f"[eval] backend: {engine.name} ({engine.dim}-d)")

    people = (
        _scan_dir(Path(args.data_dir)) if args.data_dir else load_lfw_paths()
    )
    print(f"[data] {len(people)} identities available")

    enrolled_src, impostor_src = build_split(
        people,
        args.identities,
        args.impostor_frac,
        args.min_images,
        args.enroll_per_identity,
        args.seed,
    )
    print(
        f"[eval] {len(enrolled_src)} enrolled identities, "
        f"{len(impostor_src)} impostor identities held out"
    )

    # ---- build the gallery ----
    from db import FaceDB

    gallery_path = Path("eval_gallery.npz")
    gallery_path.unlink(missing_ok=True)
    db = FaceDB(gallery_path)

    genuine_probes: list[tuple[str, np.ndarray]] = []
    undetected = 0

    for i, (name, paths) in enumerate(sorted(enrolled_src.items()), 1):
        k = args.enroll_per_identity
        enroll_embs = embed_paths(engine, paths[:k])
        probe_embs = embed_paths(engine, paths[k : k + args.max_probes])
        undetected += (k - len(enroll_embs)) + (
            len(paths[k : k + args.max_probes]) - len(probe_embs)
        )
        if not enroll_embs:
            continue
        db.enroll(name, enroll_embs)
        genuine_probes.extend((name, e) for e in probe_embs)
        if i % 25 == 0:
            print(f"[eval]   enrolled {i}/{len(enrolled_src)}")

    impostor_probes: list[np.ndarray] = []
    for name, paths in sorted(impostor_src.items()):
        impostor_probes.extend(embed_paths(engine, paths[: args.max_probes]))

    print(
        f"[eval] gallery={len(db.names)} identities | "
        f"genuine probes={len(genuine_probes)} | "
        f"impostor probes={len(impostor_probes)} | "
        f"images with no face detected={undetected}"
    )
    if not genuine_probes or not impostor_probes:
        raise RuntimeError("not enough probes; increase --identities or --max-probes")

    # ---- score every probe once, threshold afterwards ----
    # Matching with threshold=-1 accepts everything, so the raw best-match
    # name and score can be reused across the whole sweep. Re-running the
    # matcher per threshold would be wasted work.
    gen_scores, gen_correct = [], []
    for true_name, emb in genuine_probes:
        m = db.identify(emb, threshold=-1.0)
        gen_scores.append(m.score)
        gen_correct.append(m.name == true_name)

    imp_scores = [db.identify(e, threshold=-1.0).score for e in impostor_probes]

    gen_scores = np.array(gen_scores)
    gen_correct = np.array(gen_correct, dtype=bool)
    imp_scores = np.array(imp_scores)

    # ---- sweep ----
    lo = float(min(gen_scores.min(), imp_scores.min()))
    hi = float(max(gen_scores.max(), imp_scores.max()))
    grid = np.linspace(lo, hi, args.steps)

    rows = []
    for t in grid:
        accepted = gen_scores >= t
        tar = float(np.mean(accepted & gen_correct))
        mis = float(np.mean(accepted & ~gen_correct))
        frr = float(np.mean(~accepted))
        far = float(np.mean(imp_scores >= t))
        rows.append(
            {"threshold": float(t), "TAR": tar, "FAR": far, "FRR": frr, "MIS": mis}
        )

    # ---- pick the operating point ----
    viable = [r for r in rows if r["FAR"] <= args.target_far]
    if viable:
        chosen = max(viable, key=lambda r: r["TAR"])
        rationale = (
            f"lowest threshold holding FAR at or below {args.target_far:.1%}, "
            f"then maximising TAR"
        )
    else:
        chosen = min(rows, key=lambda r: r["FAR"])
        rationale = (
            f"target FAR of {args.target_far:.1%} unreachable on this data; "
            "fell back to the minimum-FAR point"
        )

    # Equal error rate is a threshold-independent quality number for the
    # embedding model itself - useful for comparing backends.
    eer_row = min(rows, key=lambda r: abs(r["FAR"] - r["FRR"]))

    result = {
        "backend": engine.name,
        "embedding_dim": engine.dim,
        "gallery_identities": len(db.names),
        "genuine_probes": int(len(gen_scores)),
        "impostor_probes": int(len(imp_scores)),
        "enroll_images_per_identity": args.enroll_per_identity,
        "undetected_images": undetected,
        "threshold": round(chosen["threshold"], 4),
        "rationale": rationale,
        "at_threshold": {k: round(v, 4) for k, v in chosen.items()},
        "equal_error_rate": round((eer_row["FAR"] + eer_row["FRR"]) / 2, 4),
        "eer_threshold": round(eer_row["threshold"], 4),
        "score_distributions": {
            "genuine_mean": round(float(gen_scores.mean()), 4),
            "genuine_std": round(float(gen_scores.std()), 4),
            "impostor_mean": round(float(imp_scores.mean()), 4),
            "impostor_std": round(float(imp_scores.std()), 4),
        },
        "sweep": rows,
    }

    Path("evaluation_results.json").write_text(json.dumps(result, indent=2))
    Path("threshold.json").write_text(
        json.dumps(
            {"threshold": result["threshold"], "backend": engine.name}, indent=2
        )
    )
    _plot(rows, chosen["threshold"], gen_scores, imp_scores)
    gallery_path.unlink(missing_ok=True)
    return result


def _plot(rows, chosen_t, gen_scores, imp_scores) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping figures")
        return

    t = [r["threshold"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for key, label in [
        ("TAR", "True accept (correct ID)"),
        ("FAR", "False accept (impostor named)"),
        ("FRR", "False reject (known -> unknown)"),
        ("MIS", "Misidentification (wrong name)"),
    ]:
        ax1.plot(t, [r[key] for r in rows], label=label)
    ax1.axvline(chosen_t, ls="--", c="k", label=f"chosen = {chosen_t:.3f}")
    ax1.set_xlabel("cosine similarity threshold")
    ax1.set_ylabel("rate")
    ax1.set_title("Threshold sweep")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.hist(imp_scores, bins=40, alpha=0.6, label="impostor pairs", density=True)
    ax2.hist(gen_scores, bins=40, alpha=0.6, label="genuine pairs", density=True)
    ax2.axvline(chosen_t, ls="--", c="k")
    ax2.set_xlabel("best-match cosine similarity")
    ax2.set_ylabel("density")
    ax2.set_title("Score separation")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("evaluation.png", dpi=130)
    print("[plot] wrote evaluation.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None, help="folder of person_name/*.jpg")
    ap.add_argument("--identities", type=int, default=120)
    ap.add_argument("--impostor-frac", type=float, default=0.4)
    ap.add_argument("--min-images", type=int, default=4)
    ap.add_argument("--enroll-per-identity", type=int, default=2)
    ap.add_argument("--max-probes", type=int, default=5)
    ap.add_argument("--target-far", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    r = evaluate(args)

    print("\n" + "=" * 62)
    print(f"  backend              {r['backend']}")
    print(f"  gallery identities   {r['gallery_identities']}")
    print(f"  genuine probes       {r['genuine_probes']}")
    print(f"  impostor probes      {r['impostor_probes']}")
    print("-" * 62)
    print(f"  CHOSEN THRESHOLD     {r['threshold']}")
    print(f"  rationale            {r['rationale']}")
    a = r["at_threshold"]
    print(f"  true accept  (TAR)   {a['TAR']:.1%}")
    print(f"  false accept (FAR)   {a['FAR']:.1%}")
    print(f"  false reject (FRR)   {a['FRR']:.1%}")
    print(f"  misidentified (MIS)  {a['MIS']:.1%}")
    print(f"  equal error rate     {r['equal_error_rate']:.1%}")
    d = r["score_distributions"]
    print(
        f"  genuine  {d['genuine_mean']:.3f} +/- {d['genuine_std']:.3f}   "
        f"impostor {d['impostor_mean']:.3f} +/- {d['impostor_std']:.3f}"
    )
    print("=" * 62)
    print("wrote evaluation_results.json, threshold.json, evaluation.png")


if __name__ == "__main__":
    main()
