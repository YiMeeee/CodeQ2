from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import MeniscusMultiModalDataset, build_clinical_stats, build_fold_split
from src.models import BioMeniscusNetPP, MultiTaskMeniscusLoss
from src.utils import AverageMeter, compute_metrics, ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MultiTaskMeniscusLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    is_train: bool,
    grad_accum_steps: int,
    amp: bool,
    clip_grad_norm: float,
) -> Dict[str, float]:
    model.train(is_train)

    loss_meter = AverageMeter("loss")
    all_grade_true: List[np.ndarray] = []
    all_grade_logits: List[np.ndarray] = []
    all_surgery_true: List[np.ndarray] = []
    all_surgery_prob: List[np.ndarray] = []

    pbar = tqdm(loader, desc="train" if is_train else "val", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        batch = move_to_device(batch, device)

        with torch.cuda.amp.autocast(enabled=amp):
            grade_logits, ordinal_logits, surgery_logit = model(
                batch["coronal"], batch["sagittal"], batch["clinical"]
            )
            loss = criterion(
                grade_logits=grade_logits,
                ordinal_logits=ordinal_logits,
                surgery_logit=surgery_logit,
                grade_target=batch["grade"],
                surgery_target=batch["surgery"],
            )
            if is_train and grad_accum_steps > 1:
                loss = loss / grad_accum_steps

        if is_train:
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        loss_meter.update(loss.item() * (grad_accum_steps if is_train else 1.0), n=batch["grade"].size(0))

        all_grade_true.append(batch["grade"].detach().cpu().numpy())
        all_grade_logits.append(grade_logits.detach().cpu().numpy())
        all_surgery_true.append(batch["surgery"].detach().cpu().numpy())
        all_surgery_prob.append(torch.sigmoid(surgery_logit).detach().cpu().numpy())

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    grade_true = np.concatenate(all_grade_true)
    grade_logits_np = np.concatenate(all_grade_logits)
    surgery_true = np.concatenate(all_surgery_true)
    surgery_prob = np.concatenate(all_surgery_prob)

    out = compute_metrics(
        grade_true=grade_true,
        grade_logits=grade_logits_np,
        surgery_true=surgery_true,
        surgery_prob=surgery_prob,
    )
    out["loss"] = float(loss_meter.avg)
    return out


def make_loaders(cfg: Dict):
    df = pd.read_csv(cfg["data"]["csv_path"])
    if "fold" not in df.columns:
        raise ValueError("metadata.csv 缺少 fold 列。请先运行 create_folds.py")

    train_df, val_df = build_fold_split(
        df=df,
        train_folds=cfg["data"]["train_folds"],
        val_fold=cfg["data"]["val_fold"],
    )

    clinical_stats = build_clinical_stats(
        train_df,
        continuous_cols=cfg["data"]["clinical"]["continuous_cols"],
        categorical_cols=cfg["data"]["clinical"]["categorical_cols"],
    )

    train_ds = MeniscusMultiModalDataset(
        df=train_df,
        image_size=cfg["data"]["image_size"],
        view_cols=cfg["data"]["views"],
        continuous_cols=cfg["data"]["clinical"]["continuous_cols"],
        categorical_cols=cfg["data"]["clinical"]["categorical_cols"],
        label_col=cfg["data"]["label_col"],
        surgery_col=cfg["data"]["surgery_col"],
        clinical_stats=clinical_stats,
        is_train=True,
    )
    val_ds = MeniscusMultiModalDataset(
        df=val_df,
        image_size=cfg["data"]["image_size"],
        view_cols=cfg["data"]["views"],
        continuous_cols=cfg["data"]["clinical"]["continuous_cols"],
        categorical_cols=cfg["data"]["clinical"]["categorical_cols"],
        label_col=cfg["data"]["label_col"],
        surgery_col=cfg["data"]["surgery_col"],
        clinical_stats=clinical_stats,
        is_train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        drop_last=False,
    )
    return train_loader, val_loader, train_ds.clinical_dim


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["seed"])
    ensure_dir(cfg["output_dir"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, clinical_in_dim = make_loaders(cfg)

    model = BioMeniscusNetPP(
        backbone_name=cfg["model"]["backbone_name"],
        pretrained=cfg["model"]["pretrained"],
        clinical_in_dim=clinical_in_dim,
        feature_dim=cfg["model"]["feature_dim"],
        clinical_hidden=cfg["model"]["clinical_hidden"],
        num_classes=cfg["model"]["num_classes"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    class_weights = torch.tensor(cfg["loss"]["class_weights"], dtype=torch.float32, device=device)
    criterion = MultiTaskMeniscusLoss(
        class_weights=class_weights,
        lambda_ord=cfg["loss"]["lambda_ord"],
        lambda_surgery=cfg["loss"]["lambda_surgery"],
        lambda_consistency=cfg["loss"]["lambda_consistency"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["scheduler"]["t0"],
        T_mult=cfg["scheduler"]["t_mult"],
        eta_min=cfg["scheduler"]["eta_min"],
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["amp"] and device.type == "cuda")

    best_score = -1.0
    wait = 0
    history = []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            is_train=True,
            grad_accum_steps=cfg["training"]["grad_accum_steps"],
            amp=cfg["training"]["amp"] and device.type == "cuda",
            clip_grad_norm=cfg["training"]["clip_grad_norm"],
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                is_train=False,
                grad_accum_steps=1,
                amp=cfg["training"]["amp"] and device.type == "cuda",
                clip_grad_norm=0.0,
            )

        scheduler.step(epoch - 1)

        score = 0.65 * val_metrics["grade_qwk"] + 0.35 * val_metrics["surgery_auc"]
        row = {"epoch": epoch, "score": score, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_qwk={val_metrics['grade_qwk']:.4f} | "
            f"val_auc={val_metrics['surgery_auc']:.4f} | "
            f"score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            wait = 0
            ckpt_path = os.path.join(cfg["output_dir"], "best_model.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "clinical_in_dim": clinical_in_dim,
                    "best_score": best_score,
                },
                ckpt_path,
            )
            print(f"[Saved] {ckpt_path}")
        else:
            wait += 1

        if wait >= cfg["training"]["early_stopping_patience"]:
            print("Early stopping triggered.")
            break

    hist_path = os.path.join(cfg["output_dir"], "history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"Training done. Best score={best_score:.4f}")
    print(f"History saved to: {hist_path}")


if __name__ == "__main__":
    main()
