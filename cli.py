"""
Command-line interface - the same engine and database as the API, without
needing the server running. Handy for generating README screenshots fast.

    python cli.py enroll  --name "Alice" --images photos/alice/*.jpg
    python cli.py identify --image test.jpg
    python cli.py identify --image group.jpg --all-faces --annotate out.jpg
    python cli.py list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from db import FaceDB, load_threshold
from face_engine import get_engine

GALLERY = "gallery.npz"


def cmd_enroll(args) -> int:
    engine = get_engine()
    db = FaceDB(GALLERY)

    embeddings = []
    for pattern in args.images:
        matches = sorted(Path().glob(pattern))
        if not matches:
            # A pattern that matches nothing is almost always an empty folder
            # or a typo - saying so beats reporting "unreadable file: *.jpg".
            if any(c in pattern for c in "*?["):
                print(f"  no files matched: {pattern}")
                continue
            matches = [Path(pattern)]
        for path in matches:
            img = cv2.imread(str(path))
            if img is None:
                print(f"  skip {path}: unreadable")
                continue
            faces = engine.detect(img)
            if len(faces) != 1:
                print(f"  skip {path}: found {len(faces)} faces, need exactly 1")
                continue
            embeddings.append(faces[0].embedding)
            print(f"  ok   {path}")

    if not embeddings:
        print("no usable faces; nothing enrolled")
        return 1

    total = db.enroll(args.name, embeddings)
    db.save()
    print(f"\nenrolled '{args.name}' with {len(embeddings)} new image(s), "
          f"{total} total. Gallery holds {len(db.names)} identities.")
    return 0


def cmd_identify(args) -> int:
    engine = get_engine()
    db = FaceDB(GALLERY)
    if not db.names:
        print("gallery is empty - enroll someone first")
        return 1

    t = args.threshold if args.threshold is not None else load_threshold(
        fallback=engine.default_threshold
    )

    img = cv2.imread(args.image)
    if img is None:
        print(f"could not read {args.image}")
        return 1

    faces = engine.detect(img)
    if not faces:
        print("no face detected")
        return 0
    if not args.all_faces:
        faces = [max(faces, key=lambda f: f.area)]

    print(f"threshold = {t:.4f}\n")
    for i, face in enumerate(faces, 1):
        m = db.identify(face.embedding, threshold=t)
        label = m.name or "UNKNOWN"
        print(f"face {i}: {label:<24} similarity={m.score:.4f}  "
              f"margin={m.margin:+.4f}  (closest: {m.runner_up or '-'})")

        if args.annotate:
            x, y, w, h = face.bbox
            colour = (0, 170, 0) if m.is_known else (0, 0, 220)
            cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(
                img, f"{label} {m.score:.2f}", (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
            )

    if args.annotate:
        cv2.imwrite(args.annotate, img)
        print(f"\nannotated image written to {args.annotate}")
    return 0


def cmd_list(args) -> int:
    db = FaceDB(GALLERY)
    stats = db.stats()
    if not stats["identities"]:
        print("gallery is empty")
        return 0
    print(f"{stats['identities']} identities, {stats['total_samples']} samples\n")
    for name, n in stats["per_identity"].items():
        print(f"  {name:<30} {n} image(s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Face recognition CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll")
    e.add_argument("--name", required=True)
    e.add_argument("--images", nargs="+", required=True)
    e.set_defaults(func=cmd_enroll)

    i = sub.add_parser("identify")
    i.add_argument("--image", required=True)
    i.add_argument("--threshold", type=float, default=None)
    i.add_argument("--all-faces", action="store_true")
    i.add_argument("--annotate", default=None, help="write a boxed output image")
    i.set_defaults(func=cmd_identify)

    l = sub.add_parser("list")
    l.set_defaults(func=cmd_list)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())