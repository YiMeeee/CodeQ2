from __future__ import annotations

import argparse

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets import MeniscusMultiModalDataset, build_clinical_stats
from src.models import BioMeniscusNetPP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_csv", type=str, default="data/test_metadata.csv")
    parser.add_argument("--output_csv", type=str, default="outputs/predictions.csv")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    infer_df = pd.read_csv(args.input_csv)
    # 推理时统计量建议使用训练集统计，这里为可直接运行先使用输入集估计
    clinical_stats = build_clinical_stats(
        infer_df,
        continuous_cols=cfg["data"]["clinical"]["continuous_cols"],
        categorical_cols=cfg["data"]["clinical"]["categorical_cols"],
    )

    ds = MeniscusMultiModalDataset(
        df=infer_df,
        image_size=cfg["data"]["image_size"],
        view_cols=cfg["data"]["views"],
        continuous_cols=cfg["data"]["clinical"]["continuous_cols"],
        categorical_cols=cfg["data"]["clinical"]["categorical_cols"],
        label_col=cfg["data"]["label_col"],
        surgery_col=cfg["data"]["surgery_col"],
        clinical_stats=clinical_stats,
        is_train=False,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

    model = BioMeniscusNetPP(
        backbone_name=cfg["model"]["backbone_name"],
        pretrained=False,
        clinical_in_dim=ckpt["clinical_in_dim"],
        feature_dim=cfg["model"]["feature_dim"],
        clinical_hidden=cfg["model"]["clinical_hidden"],
        num_classes=cfg["model"]["num_classes"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    pred_grade = []
    pred_surgery = []

    with torch.no_grad():
        for batch in loader:
            coronal = batch["coronal"].to(device)
            sagittal = batch["sagittal"].to(device)
            clinical = batch["clinical"].to(device)

            grade_logits, _, surgery_logit = model(coronal, sagittal, clinical)
            grade = grade_logits.argmax(dim=1).cpu().numpy()
            surgery = torch.sigmoid(surgery_logit).cpu().numpy()

            pred_grade.extend(grade.tolist())
            pred_surgery.extend(surgery.tolist())

    out = infer_df.copy()
    out["pred_grade"] = pred_grade
    out["pred_surgery_prob"] = pred_surgery
    out.to_csv(args.output_csv, index=False)
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
