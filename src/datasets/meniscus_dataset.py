from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class ClinicalStats:
    cont_mean: np.ndarray
    cont_std: np.ndarray
    cat_maps: Dict[str, Dict[str, int]]


def build_transforms(image_size: int, is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _read_image(path: str) -> Image.Image:
    if path.lower().endswith(".npy"):
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    return Image.open(path).convert("RGB")


def build_clinical_stats(
    df: pd.DataFrame,
    continuous_cols: List[str],
    categorical_cols: List[str],
) -> ClinicalStats:
    cont_arr = df[continuous_cols].astype(float).values
    cont_mean = cont_arr.mean(axis=0)
    cont_std = cont_arr.std(axis=0)
    cont_std = np.where(cont_std < 1e-6, 1.0, cont_std)

    cat_maps: Dict[str, Dict[str, int]] = {}
    for c in categorical_cols:
        values = sorted(df[c].astype(str).fillna("unknown").unique().tolist())
        cat_maps[c] = {v: i for i, v in enumerate(values)}

    return ClinicalStats(cont_mean=cont_mean, cont_std=cont_std, cat_maps=cat_maps)


class MeniscusMultiModalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int,
        view_cols: Dict[str, str],
        continuous_cols: List[str],
        categorical_cols: List[str],
        label_col: str,
        surgery_col: str,
        clinical_stats: ClinicalStats,
        is_train: bool,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.transform = build_transforms(image_size=image_size, is_train=is_train)
        self.view_cols = view_cols
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.label_col = label_col
        self.surgery_col = surgery_col
        self.stats = clinical_stats

        self.clinical_dim = len(continuous_cols) + sum(
            len(clinical_stats.cat_maps[c]) for c in categorical_cols
        ) + 2

        self.age_idx = self.continuous_cols.index("age") if "age" in self.continuous_cols else 0
        self.bmi_idx = self.continuous_cols.index("bmi") if "bmi" in self.continuous_cols else 0
        self.energy_idx = (
            self.continuous_cols.index("energy_score")
            if "energy_score" in self.continuous_cols
            else 0
        )

    def __len__(self) -> int:
        return len(self.df)

    def _encode_clinical(self, row: pd.Series) -> np.ndarray:
        cont = row[self.continuous_cols].astype(float).values
        cont = (cont - self.stats.cont_mean) / self.stats.cont_std

        cat_vecs = []
        for c in self.categorical_cols:
            mapping = self.stats.cat_maps[c]
            v = str(row[c]) if pd.notna(row[c]) else "unknown"
            idx = mapping.get(v, 0)
            one_hot = np.zeros(len(mapping), dtype=np.float32)
            one_hot[idx] = 1.0
            cat_vecs.append(one_hot)

        age = cont[self.age_idx]
        bmi = cont[self.bmi_idx]
        energy = cont[self.energy_idx]
        interactions = np.array([bmi * age, bmi * energy], dtype=np.float32)

        full = np.concatenate([cont.astype(np.float32)] + cat_vecs + [interactions], axis=0)
        return full

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        coronal = _read_image(str(row[self.view_cols["coronal_col"]]))
        sagittal = _read_image(str(row[self.view_cols["sagittal_col"]]))

        coronal = self.transform(coronal)
        sagittal = self.transform(sagittal)

        clinical = self._encode_clinical(row)

        grade = int(row[self.label_col])
        if self.surgery_col in row and pd.notna(row[self.surgery_col]):
            surgery = int(row[self.surgery_col])
        else:
            surgery = int(grade >= 2)

        return {
            "coronal": coronal,
            "sagittal": sagittal,
            "clinical": torch.tensor(clinical, dtype=torch.float32),
            "grade": torch.tensor(grade, dtype=torch.long),
            "surgery": torch.tensor(surgery, dtype=torch.float32),
        }


def build_fold_split(
    df: pd.DataFrame,
    train_folds: List[int],
    val_fold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["fold"].isin(train_folds)].copy()
    val_df = df[df["fold"] == val_fold].copy()
    return train_df, val_df
