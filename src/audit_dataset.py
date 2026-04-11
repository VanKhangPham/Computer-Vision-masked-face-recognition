from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import CFG, DATA_RAW, LOGS_DIR
from src.vision_utils import configure_output_encoding, detect_faces, load_face_detector


configure_output_encoding()


@dataclass
class AuditRow:
    class_name: str
    path: Path
    face_count: int
    largest_face_ratio: float
    status: str


def audit_image(path: Path, class_name: str, face_net, min_face_ratio: float) -> AuditRow:
    image = cv2.imread(str(path))
    if image is None:
        return AuditRow(class_name, path, 0, 0.0, "invalid")

    height, width = image.shape[:2]
    image_area = max(1, height * width)
    faces = detect_faces(image, face_net, confidence_threshold=CFG.DATA_FACE_CONFIDENCE)

    largest_face_ratio = 0.0
    for x1, y1, x2, y2, _ in faces:
        face_area = max(1, (x2 - x1) * (y2 - y1))
        largest_face_ratio = max(largest_face_ratio, face_area / image_area)

    if not faces:
        status = "no_face"
    elif largest_face_ratio < min_face_ratio:
        status = "small_face"
    elif len(faces) > 1:
        status = "multi_face"
    else:
        status = "ok"

    return AuditRow(class_name, path, len(faces), largest_face_ratio, status)


def write_report(rows: list[AuditRow], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class_name", "path", "face_count", "largest_face_ratio", "status"])
        for row in rows:
            writer.writerow(
                [
                    row.class_name,
                    row.path.as_posix(),
                    row.face_count,
                    f"{row.largest_face_ratio:.6f}",
                    row.status,
                ]
            )


def move_rows(rows: list[AuditRow], destination_root: Path, statuses: set[str]) -> int:
    moved = 0
    for row in rows:
        if row.status not in statuses:
            continue
        destination = destination_root / row.class_name / row.path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            stem = destination.stem
            suffix = destination.suffix
            counter = 1
            while destination.exists():
                destination = destination_root / row.class_name / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.move(str(row.path), str(destination))
        moved += 1
    return moved


def summarize(rows: list[AuditRow]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        summary[row.status] = summary.get(row.status, 0) + 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw mask dataset quality.")
    parser.add_argument(
        "--min-face-ratio",
        type=float,
        default=0.04,
        help="Minimum face area / image area ratio before flagging as small_face.",
    )
    parser.add_argument(
        "--move-status",
        nargs="*",
        default=[],
        choices=["invalid", "no_face", "small_face", "multi_face"],
        help="Statuses to move out of data/raw into data/review.",
    )
    parser.add_argument(
        "--report",
        default=str(LOGS_DIR / "dataset_audit.csv"),
        help="CSV report output path.",
    )
    parser.add_argument(
        "--review-dir",
        default=str(ROOT_DIR / "data" / "review"),
        help="Directory used when moving noisy files.",
    )
    args = parser.parse_args()

    print("=" * 50)
    print(" DATASET AUDIT")
    print("=" * 50)
    print(f"Raw dir        : {DATA_RAW}")
    print(f"Min face ratio : {args.min_face_ratio:.3f}")

    face_net = load_face_detector()
    rows: list[AuditRow] = []

    for class_name in CFG.CLASSES:
        class_dir = DATA_RAW / class_name
        if not class_dir.exists():
            print(f"Missing class dir: {class_dir}")
            continue

        files = sorted([p for p in class_dir.iterdir() if p.is_file()])
        print(f"Scanning {class_name}: {len(files)} files")
        for path in files:
            rows.append(audit_image(path, class_name, face_net, args.min_face_ratio))

    report_path = Path(args.report)
    write_report(rows, report_path)
    print(f"\nSaved report: {report_path}")

    summary = summarize(rows)
    for status in ["ok", "multi_face", "small_face", "no_face", "invalid"]:
        print(f"{status:10s}: {summary.get(status, 0)}")

    if args.move_status:
        moved = move_rows(rows, Path(args.review_dir), set(args.move_status))
        print(f"\nMoved {moved} files to: {args.review_dir}")


if __name__ == "__main__":
    main()
