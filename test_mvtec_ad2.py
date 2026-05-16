import argparse
import os
import warnings
from contextlib import nullcontext
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.mvtec_ad2 import MVTecAD2TestDataset
from decoder import SimpleDecoder, SimplePredictor
from models.model import BACKBONES, FeatureExtractor
from peft_local.peft_func import PeftType

warnings.filterwarnings("ignore", message="invalid value encountered in divide")


class RETURN_VALUES(Enum):
    PIX_F1_RAW = "F1-PIXEL-RAW"
    PIX_THRESHOLD_RAW = "THRESHOLD-PIXEL-RAW"

    def __str__(self):
        return self.value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="./mvtec_ad_2/",
        help="MVTec AD 2 root directory. It should contain the 8 category folders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test_private", "test_private_mixed"],
        choices=["test_private", "test_private_mixed"],
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional subset of categories. By default all folders under --data-path are used.",
    )

    parser.add_argument(
        "-m",
        "--model",
        default=str(BACKBONES.TIPSV2),
        type=BACKBONES,
        choices=list(BACKBONES),
    )
    parser.add_argument(
        "-p",
        "--model-path",
        default="./experiments/tipsv2_2500/model.pkl",
    )
    parser.add_argument(
        "--peft-type",
        default=PeftType.DORA,
        type=PeftType,
        choices=list(PeftType),
    )
    parser.add_argument("--peft-rank", default=64, type=int)
    parser.add_argument("--image-size", default=672, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--mean-kernel-size", default=5, type=int)
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--tta-rotations",
        nargs="+",
        default=[0, 90],
        type=int,
        choices=[0, 90, 180, 270],
        help="Rotation TTA degrees to average. Examples: --tta-rotations 0, --tta-rotations 0 90 180 270.",
    )

    parser.add_argument("--save-images", action="store_true", default=False)
    parser.add_argument(
        "--save-threshold",
        default=0.0345,
        type=float,
        help="Optional threshold applied to anomaly score maps before saving images. If set, saved maps are binary 0/255.",
    )
    parser.add_argument("--out-path", default="./results/anomaly_images_thresholded")

    return parser.parse_args()


def discover_categories(root_path, requested_categories=None):
    if requested_categories is not None and len(requested_categories) > 0:
        return requested_categories

    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    categories = sorted(
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if len(categories) == 0:
        raise RuntimeError(f"No category folders found under {root}")
    return categories


def save_predictions_with_paths(tensors, relative_paths, out_base_path, suffix=""):
    base_out = Path(out_base_path)

    for tensor, rel_path in zip(tensors, relative_paths):
        out_file = base_out / rel_path
        if suffix:
            out_file = out_file.with_stem(f"{out_file.stem}")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        img_np = tensor.detach().cpu().squeeze().numpy()
        img_np = np.clip(img_np, 0.0, 1.0)
        img_np = (img_np * 255.0).astype(np.uint8)

        if img_np.ndim == 3 and img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(out_file), img_np)


def save_predictions_at_original_size(
    tensors,
    relative_paths,
    orig_sizes,
    out_base_path,
    suffix="",
    threshold=None,
):
    resized = []
    for tensor, orig_size in zip(tensors, orig_sizes):
        height = int(orig_size[0].item())
        width = int(orig_size[1].item())
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if threshold is not None:
            tensor = (tensor >= threshold).float()

        resized.append(tensor)
    save_predictions_with_paths(resized, relative_paths, out_base_path, suffix=suffix)


def remap_and_load_model_state(model, model_state):
    current_model_keys = set(model.state_dict().keys())
    remapped_model_state = {}

    for k, v in model_state.items():
        new_k = k

        if k.endswith(".attn.proj.weight"):
            candidate = k.replace(".attn.proj.weight", ".attn.proj.layer.weight")
            if candidate in current_model_keys:
                new_k = candidate
        elif k.endswith(".attn.proj.bias"):
            candidate = k.replace(".attn.proj.bias", ".attn.proj.layer.bias")
            if candidate in current_model_keys:
                new_k = candidate

        if new_k in current_model_keys:
            remapped_model_state[new_k] = v

    model.load_state_dict(remapped_model_state, strict=False)


def evaluate_category(
    model,
    decoder,
    predictor,
    dataset,
    batch_size,
    num_workers,
    feat_size,
    image_size,
    kernel=None,
    fast=True,
    save_images=False,
    save_threshold=None,
    out_path="./visual_results/",
    tta_rotations=None,
):
    if tta_rotations is None:
        tta_rotations = [0]

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": False,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(**loader_kwargs)

    num_samples = len(dataset)

    pool = None
    if kernel is not None and kernel > 1:
        pool = nn.AvgPool2d((kernel, kernel), 1, kernel // 2).cuda()

    ptr = 0
    amp_context = (
        torch.amp.autocast("cuda", dtype=torch.float16) if fast else nullcontext()
    )

    with tqdm(
            total=num_samples,
            desc="Evaluating",
            unit="img",
            dynamic_ncols=True,
            leave=True,
        ) as pbar:
        for sample in loader:
            image = sample["image"].cuda(non_blocking=True)
            paths = sample["path"]
            orig_sizes = sample["orig_size"]

            batch_sz = image.size(0)
            idx = slice(ptr, ptr + batch_sz)

            with torch.inference_mode():
                with amp_context:
                    mask_pred_sum = None

                    for rotation in tta_rotations:
                        rot_k = int(rotation) // 90

                        if rot_k == 0:
                            image_rot = image
                        else:
                            image_rot = torch.rot90(image, k=rot_k, dims=(-2, -1))

                        summary, ftrs = model(image_rot)

                        batch_size_actual, num_tokens, channels = ftrs.shape
                        actual_feat_size = int(num_tokens ** 0.5)
                        if actual_feat_size * actual_feat_size != num_tokens:
                            raise RuntimeError(
                                f"Invalid feature token shape: ftrs.shape={ftrs.shape}. "
                                f"num_tokens={num_tokens} is not a perfect square."
                            )

                        ftrs = (
                            ftrs.permute(0, 2, 1)
                            .contiguous()
                            .reshape(
                                batch_size_actual,
                                channels,
                                actual_feat_size,
                                actual_feat_size,
                            )
                        )

                        mask_pred_rot, _ = decoder(ftrs)
                        _ = predictor(summary)

                        mask_pred_rot = torch.nn.functional.interpolate(
                            mask_pred_rot.sigmoid(),
                            (image_size, image_size),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(1)

                        if rot_k != 0:
                            mask_pred_rot = torch.rot90(
                                mask_pred_rot,
                                k=-rot_k,
                                dims=(-2, -1),
                            )

                        if mask_pred_sum is None:
                            mask_pred_sum = mask_pred_rot
                        else:
                            mask_pred_sum = mask_pred_sum + mask_pred_rot

                    mask_pred = mask_pred_sum / float(len(tta_rotations))

                    if pool is not None:
                        mask_pred = pool(mask_pred)

            if save_images:
                save_predictions_at_original_size(
                    mask_pred,
                    paths,
                    orig_sizes,
                    out_path,
                    suffix="pred_raw",
                    threshold=save_threshold,
                )
            
            ptr += batch_sz
            pbar.update(batch_sz)



def average_results(results):
    keys = list(results[0].keys())
    avg = {}
    for key in keys:
        avg[key] = float(np.mean([r[key] for r in results]))
    return avg


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu")


def load_model(args):
    model = FeatureExtractor(args.model, args.image_size).model
    feat_dim = model.feature_dim
    feat_size = args.image_size // model.patch_size

    model.add_peft(args.peft_rank, peft_type=args.peft_type)

    num_up_layers = 1
    decoder = SimpleDecoder(feat_dim, num_up_layers, 1)

    if args.model == BACKBONES.RADIO:
        predictor = SimplePredictor(3 * feat_dim)
    elif hasattr(BACKBONES, "TIPSV2") and args.model == BACKBONES.TIPSV2:
        predictor = SimplePredictor(2 * feat_dim)
    else:
        predictor = SimplePredictor(feat_dim)

    state_dicts = load_checkpoint(args.model_path)
    remap_and_load_model_state(model, state_dicts["model_state_dict"])
    decoder.load_state_dict(state_dicts["decoder_state_dict"], strict=False)
    predictor.load_state_dict(state_dicts["predictor_state_dict"], strict=False)

    model.cuda().eval()
    decoder.cuda().eval()
    predictor.cuda().eval()

    img_transform = model.get_img_transform()

    return model, decoder, predictor, img_transform, feat_size


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation script requires CUDA.")

    if args.save_threshold is not None and not 0.0 <= args.save_threshold <= 1.0:
        raise ValueError(f"--save-threshold must be in [0, 1], got {args.save_threshold}")

    categories = discover_categories(args.data_path, args.categories)
    model, decoder, predictor, img_transform, feat_size = load_model(args)

    os.makedirs(args.out_path, exist_ok=True)

    for category in categories:
        print(f"\n========== Category: {category} ==========")

        for split in args.splits:
            print(f"\nSplit: {split}")
            dataset = MVTecAD2TestDataset(
                path=args.data_path,
                category=category,
                transform=img_transform,
                split=split,
            )

            evaluate_category(
                model=model,
                decoder=decoder,
                predictor=predictor,
                dataset=dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                feat_size=feat_size,
                image_size = args.image_size,
                kernel=args.mean_kernel_size,
                fast=args.fast,
                save_images=args.save_images,
                save_threshold=args.save_threshold,
                out_path=args.out_path,
                tta_rotations=args.tta_rotations,
            )


if __name__ == "__main__":
    main()