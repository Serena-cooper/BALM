# compare_results_similarity.py
# -*- coding: utf-8 -*-

import argparse
import csv
import io
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two results.tar.gz files by exact pixel equality. "
            "For each matched anomaly image, similarity = same pixels / total pixels. "
            "Then average over all matched images."
        )
    )

    parser.add_argument(
        "--tar-a",
        type=str,
        required=True,
        help="Path to the first results.tar.gz file.",
    )

    parser.add_argument(
        "--tar-b",
        type=str,
        required=True,
        help="Path to the second results.tar.gz file.",
    )

    parser.add_argument(
        "--path-keyword",
        type=str,
        default="anomaly_images_thresholded",
        help=(
            "Only compare image files whose path contains this keyword. "
            "Default: anomaly"
        ),
    )

    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Optional CSV path to save per-image similarity results.",
    )

    return parser.parse_args()


def normalize_member_name(name: str) -> str:
    name = name.replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    name = name.lstrip("/")
    return name


def is_target_image(name: str, path_keyword: str) -> bool:
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_EXTS:
        return False

    if path_keyword:
        return path_keyword.lower() in name.lower()

    return True


def collect_image_members(tar_path: Path, path_keyword: str) -> Dict[str, str]:
    members = {}

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue

            norm_name = normalize_member_name(member.name)

            if is_target_image(norm_name, path_keyword):
                if norm_name in members:
                    raise RuntimeError(f"Duplicate normalized path in {tar_path}: {norm_name}")
                members[norm_name] = member.name

    return members


def read_image_from_tar(tar: tarfile.TarFile, member_name: str) -> np.ndarray:
    extracted = tar.extractfile(member_name)
    if extracted is None:
        raise RuntimeError(f"Failed to extract member: {member_name}")

    data = extracted.read()

    with Image.open(io.BytesIO(data)) as img:
        arr = np.array(img)

    return arr


def pixel_similarity(arr_a: np.ndarray, arr_b: np.ndarray) -> float:
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}")

    if arr_a.ndim == 2:
        same = arr_a == arr_b
        return float(np.sum(same) / same.size)

    if arr_a.ndim == 3:
        same_pixel = np.all(arr_a == arr_b, axis=-1)
        return float(np.sum(same_pixel) / same_pixel.size)

    same = arr_a == arr_b
    return float(np.sum(same) / same.size)


def compare_tar_files(
    tar_a_path: Path,
    tar_b_path: Path,
    path_keyword: str,
) -> Tuple[float, List[dict]]:
    members_a = collect_image_members(tar_a_path, path_keyword)
    members_b = collect_image_members(tar_b_path, path_keyword)

    names_a = set(members_a.keys())
    names_b = set(members_b.keys())

    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)

    if only_a or only_b:
        message = []

        if only_a:
            message.append("Files only in tar-a:")
            message.extend(f"  {name}" for name in only_a[:50])
            if len(only_a) > 50:
                message.append(f"  ... and {len(only_a) - 50} more")

        if only_b:
            message.append("Files only in tar-b:")
            message.extend(f"  {name}" for name in only_b[:50])
            if len(only_b) > 50:
                message.append(f"  ... and {len(only_b) - 50} more")

        raise RuntimeError("\n".join(message))

    common_names = sorted(names_a)

    if len(common_names) == 0:
        raise RuntimeError(
            f"No matched image files found with path keyword: {path_keyword!r}"
        )

    rows = []

    with tarfile.open(tar_a_path, "r:gz") as tar_a, tarfile.open(tar_b_path, "r:gz") as tar_b:
        for name in common_names:
            arr_a = read_image_from_tar(tar_a, members_a[name])
            arr_b = read_image_from_tar(tar_b, members_b[name])

            sim = pixel_similarity(arr_a, arr_b)

            rows.append(
                {
                    "path": name,
                    "similarity": sim,
                    "shape": str(tuple(arr_a.shape)),
                    "dtype_a": str(arr_a.dtype),
                    "dtype_b": str(arr_b.dtype),
                }
            )

    mean_similarity = float(np.mean([row["similarity"] for row in rows]))

    return mean_similarity, rows


def save_csv(rows: List[dict], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "similarity", "shape", "dtype_a", "dtype_b"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    tar_a_path = Path(args.tar_a).expanduser().resolve()
    tar_b_path = Path(args.tar_b).expanduser().resolve()

    if not tar_a_path.exists():
        raise FileNotFoundError(f"tar-a not found: {tar_a_path}")

    if not tar_b_path.exists():
        raise FileNotFoundError(f"tar-b not found: {tar_b_path}")

    mean_similarity, rows = compare_tar_files(
        tar_a_path=tar_a_path,
        tar_b_path=tar_b_path,
        path_keyword=args.path_keyword,
    )

    print(f"Compared images: {len(rows)}")
    print(f"Average pixel similarity: {mean_similarity:.8f}")

    if args.csv_out is not None:
        save_csv(rows, Path(args.csv_out).expanduser().resolve())
        print(f"Per-image results saved to: {args.csv_out}")


if __name__ == "__main__":
    main()