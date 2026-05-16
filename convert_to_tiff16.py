from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTS


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)

        if img.mode == "P":
            img = img.convert("RGBA")
        elif img.mode == "1":
            img = img.convert("L")

        array = np.array(img)

    return array


def to_float16(array: np.ndarray, normalize: bool) -> np.ndarray:
    src_dtype = array.dtype

    if np.issubdtype(src_dtype, np.floating):
        out = array.astype(np.float32, copy=False)
    else:
        out = array.astype(np.float32)

    if normalize and np.issubdtype(src_dtype, np.integer):
        info = np.iinfo(src_dtype)
        if info.max > 0:
            out = out / float(info.max)

    return out.astype(np.float16)


def convert_one(
    src_path: Path,
    dst_path: Path,
    normalize: bool = False,
    overwrite: bool = False,
) -> None:
    if dst_path.exists() and not overwrite:
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    image = load_image(src_path)
    image_f16 = to_float16(image, normalize=normalize)

    tifffile.imwrite(dst_path, image_f16)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "递归遍历输入目录下的图片，转换为 TIFF float16，"
            "并在输出目录中保持原有相对路径和文件名前缀不变。"
        )
    )
    parser.add_argument("--input_dir", type=str, default="./results/anomaly_images_thresholded", help="输入根目录")
    parser.add_argument("--output_dir", type=str, default="./results/anomaly_images", help="输出根目录")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出文件已存在，则覆盖",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"输入目录不存在或不是文件夹: {input_dir}")

    all_files = [p for p in input_dir.rglob("*") if is_image_file(p)]
    converted = 0
    failed = 0

    for src_path in all_files:
        rel_path = src_path.relative_to(input_dir)
        dst_path = (output_dir / rel_path).with_suffix(".tiff")

        try:
            convert_one(
                src_path=src_path,
                dst_path=dst_path,
                overwrite=args.overwrite,
            )
            converted += 1
            # print(f"[OK] {src_path} -> {dst_path}")
        except (UnidentifiedImageError, OSError, ValueError) as e:
            failed += 1
            print(f"[FAILED] {src_path}: {e}")

    print(f"\n完成: 成功 {converted} 个, 失败 {failed} 个")


if __name__ == "__main__":
    main()