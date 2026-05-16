import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from aux_dataset import AuxilaryDataset
from decoder import SimpleDecoder, SimplePredictor
from logger import log_results
from models.model import BACKBONES, FeatureExtractor
from peft_local.peft_func import PeftType
from myutils import (
    OPTIMIZERS,
    SCHEDULERS,
    get_optimizer,
    get_scheduler,
    torch_seed,
)

warnings.filterwarnings("ignore", message="invalid value encountered in divide")


def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "-m",
        "--model",
        default=str(BACKBONES.TIPSV2),
        type=BACKBONES,
        choices=list(BACKBONES),
    )
    parser.add_argument("-as", "--accumulation-steps", default=4, type=int)
    parser.add_argument("-bs", "--batch-size", default=16, type=int)
    parser.add_argument("-wd", "--weight-decay", default=1e-2, type=float)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument(
        "--optimizer",
        default=str(OPTIMIZERS.ADAMW),
        type=OPTIMIZERS,
        choices=list(OPTIMIZERS),
    )
    parser.add_argument(
        "--scheduler",
        default=str(SCHEDULERS.NONE),
        type=SCHEDULERS,
        choices=list(SCHEDULERS),
    )
    parser.add_argument(
        "--peft-type", default=PeftType.DORA, type=PeftType, choices=list(PeftType)
    )
    parser.add_argument("--peft-rank", default=64, type=int)

    parser.add_argument("--seed", default=12, type=int)

    parser.add_argument(
        "-d",
        "--data-path",
        default="./synthetic_dataset_flux_filter_dinov3/",
    )
    parser.add_argument("--image-size", default=672, type=int)
    parser.add_argument(
        "-o", "--out-path", default="./experiments/tipsv2_2500/", type=str
    )

    parser.add_argument("-t", "--train-steps", default=2500, type=int)

    parser.add_argument("--mean-kernel-size", default=5, type=int)

    parser.add_argument("--threshold-aware-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixed-pixel-threshold", default=0.05, type=float)

    parser.add_argument("--mask-margin-weight", default=0.05, type=float)
    parser.add_argument("--mask-normal-weight", default=0.0, type=float)
    parser.add_argument("--pos-logit-margin", default=1, type=float)
    parser.add_argument("--neg-logit-margin", default=1, type=float)
    parser.add_argument("--normal-logit-margin", default=-2.9444, type=float)

    args = parser.parse_args()
    return args


def turn_to_exp(conf_map):
    return 1 + conf_map.exp()


def probability_to_logit(probability, eps=1e-6):
    probability = float(np.clip(probability, eps, 1.0 - eps))
    return float(np.log(probability / (1.0 - probability)))


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


def logit_margin_loss(
    logits,
    masks,
    pos_margin=1.0,
    neg_margin=1.0,
    boundary_logit=0.0,
):
    masks = masks.float()
    normal_masks = 1.0 - masks

    shifted_logits = logits - boundary_logit

    pos_loss = F.softplus(pos_margin - shifted_logits) * masks
    neg_loss = F.softplus(shifted_logits + neg_margin) * normal_masks

    pos_denom = masks.sum().clamp_min(1.0)
    neg_denom = normal_masks.sum().clamp_min(1.0)

    return pos_loss.sum() / pos_denom + neg_loss.sum() / neg_denom


def normal_suppression_loss(logits, masks, margin=-2.9444):
    normal_masks = 1.0 - masks.float()
    loss = F.relu(logits - margin) * normal_masks
    denom = normal_masks.sum().clamp_min(1.0)
    return loss.sum() / denom


def main(args):
    torch_seed(args.seed)
    model = FeatureExtractor(args.model, args.image_size).model

    feat_dim = model.feature_dim
    feat_size = args.image_size // model.patch_size

    freeze_parameters(model)

    peft_rank = args.peft_rank
    model.add_peft(peft_rank, peft_type=args.peft_type)

    num_up_layers = 1

    decoder = SimpleDecoder(feat_dim, num_up_layers, 1)

    if args.model in [BACKBONES.RADIO]:
        predictor = SimplePredictor(feat_dim * 3)
    elif args.model in [BACKBONES.TIPSV2]:
        predictor = SimplePredictor(feat_dim * 2)
    else:
        predictor = SimplePredictor(feat_dim)

    model.cuda()
    decoder.cuda()
    predictor.cuda()

    model.train()
    decoder.train()
    predictor.train()

    opt = get_optimizer(args.optimizer)
    optimizer = opt(
        [
            {"params": model.parameters()},
            {"params": decoder.parameters()},
            {"params": predictor.parameters()},
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer.zero_grad()

    scheduler = get_scheduler(
        args.scheduler, optimizer, num_iterations=args.train_steps
    )

    img_transform = model.get_img_transform()

    mask_transform = T.Compose(
        [
            T.Resize((feat_size * (2**num_up_layers), feat_size * (2**num_up_layers))),
        ]
    )

    dataset = AuxilaryDataset(args.data_path, img_transform, mask_transform)

    batch_size = args.batch_size // args.accumulation_steps

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    infinite_loader = InfiniteDataloader(loader)

    alpha = 0.1
    accumulation_steps = args.accumulation_steps
    boundary_logit = probability_to_logit(args.fixed_pixel_threshold)

    img_loss = torchvision.ops.sigmoid_focal_loss
    mask_foc_loss = torchvision.ops.sigmoid_focal_loss
    mask_l1_loss = torch.nn.L1Loss(reduction="none")

    tqdm_obj = tqdm(range(args.train_steps))

    for it in tqdm_obj:
        inner_loop = range(accumulation_steps)
        loss = 0

        for _, sample in zip(inner_loop, infinite_loader):
            image = sample["image"].cuda()
            mask_gt = sample["mask"].cuda()
            score_gt = sample["is_anom"].cuda()

            summary, ftrs = model(image)

            ftrs = ftrs.permute(0, 2, 1)
            ftrs = ftrs.reshape(-1, feat_dim, feat_size, feat_size)

            mask, c = decoder(ftrs)

            score = predictor(summary).squeeze(1)
            l_img = img_loss(score, score_gt, reduction="mean")

            if args.threshold_aware_loss:
                mask_for_loss = mask - boundary_logit
                margin_boundary_logit = boundary_logit
            else:
                mask_for_loss = mask
                margin_boundary_logit = 0.0

            inner_mask_loss = 5 * mask_foc_loss(
                mask_for_loss, mask_gt, reduction="none"
            ) + mask_l1_loss(mask_for_loss.sigmoid().clip(0.1, 0.9), mask_gt)

            c = turn_to_exp(c)
            l_mask_1 = c * inner_mask_loss
            l_mask_2 = alpha * c.log()
            l_mask = l_mask_1 - l_mask_2
            l_mask = l_mask.mean()

            if args.mask_margin_weight > 0:
                l_margin = logit_margin_loss(
                    mask,
                    mask_gt,
                    pos_margin=args.pos_logit_margin,
                    neg_margin=args.neg_logit_margin,
                    boundary_logit=margin_boundary_logit,
                )
            else:
                l_margin = mask.new_tensor(0.0)

            if args.mask_normal_weight > 0:
                l_normal = normal_suppression_loss(
                    mask,
                    mask_gt,
                    margin=args.normal_logit_margin,
                )
            else:
                l_normal = mask.new_tensor(0.0)

            loss = (
                l_img
                + l_mask
                + args.mask_margin_weight * l_margin
                + args.mask_normal_weight * l_normal
            )
            loss /= accumulation_steps
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if it % 1 == 0:
            weighted_margin = args.mask_margin_weight * l_margin

            tqdm_obj.set_description(
                (
                    "Current loss: {:.4f}, img loss {:.4f}, mask loss {:.4f}, "
                ).format(
                    loss.item(),
                    l_img.item(),
                    l_mask.item(),
                    l_margin.item(),
                    weighted_margin.item(),
                )
            )

    os.makedirs(args.out_path, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
        },
        f"{args.out_path}/model.pkl",
    )


if __name__ == "__main__":
    args = get_args()
    main(args)