from __future__ import annotations

from typing import Tuple

import timm
import torch
import torch.nn as nn


class ClinicalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossViewClinicalFusion(nn.Module):
    """
    高级融合模块：
    1) Coronal/Sagittal token + Clinical token 做跨模态自注意力
    2) 用临床先验生成门控，对视觉特征进行条件调制
    """

    def __init__(
        self,
        vis_dim: int,
        clin_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj_clin = nn.Linear(clin_dim, vis_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=vis_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(vis_dim)
        self.ffn = nn.Sequential(
            nn.Linear(vis_dim, vis_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vis_dim * 2, vis_dim),
        )
        self.norm2 = nn.LayerNorm(vis_dim)

        self.gate = nn.Sequential(
            nn.Linear(vis_dim + clin_dim, vis_dim),
            nn.ReLU(inplace=True),
            nn.Linear(vis_dim, vis_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        v_cor: torch.Tensor,
        v_sag: torch.Tensor,
        f_clin: torch.Tensor,
    ) -> torch.Tensor:
        clin_token = self.proj_clin(f_clin)
        tokens = torch.stack([v_cor, v_sag, clin_token], dim=1)  # [B, 3, D]

        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))

        vis_fused = 0.5 * (tokens[:, 0] + tokens[:, 1])
        cond_gate = self.gate(torch.cat([vis_fused, f_clin], dim=1))
        vis_cond = vis_fused * cond_gate

        return vis_cond


class BioMeniscusNetPP(nn.Module):
    """
    Bio-MeniscusNet++
    - 双视图主干
    - 临床先验编码
    - 跨模态门控注意力融合
    - 多任务输出：4级分级 + 手术指征
    - 序关系辅助头（ordinal logits）
    """

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        clinical_in_dim: int,
        feature_dim: int = 512,
        clinical_hidden: int = 128,
        num_classes: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=3,
        )
        backbone_dim = self.backbone.num_features
        self.vis_proj = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.clin_encoder = ClinicalEncoder(clinical_in_dim, clinical_hidden, dropout=dropout)
        self.fusion = CrossViewClinicalFusion(
            vis_dim=feature_dim,
            clin_dim=clinical_hidden,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + clinical_hidden, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

        self.ordinal_head = nn.Sequential(
            nn.Linear(feature_dim + clinical_hidden, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes - 1),
        )

        self.surgery_head = nn.Sequential(
            nn.Linear(feature_dim + clinical_hidden, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
        )

    def extract_visual(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.vis_proj(feat)

    def forward(
        self,
        coronal: torch.Tensor,
        sagittal: torch.Tensor,
        clinical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_cor = self.extract_visual(coronal)
        v_sag = self.extract_visual(sagittal)
        f_clin = self.clin_encoder(clinical)

        vis_cond = self.fusion(v_cor=v_cor, v_sag=v_sag, f_clin=f_clin)
        fused = torch.cat([vis_cond, f_clin], dim=1)

        grade_logits = self.classifier(fused)
        ordinal_logits = self.ordinal_head(fused)
        surgery_logit = self.surgery_head(fused).squeeze(1)
        return grade_logits, ordinal_logits, surgery_logit
