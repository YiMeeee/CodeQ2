# Bio-MeniscusNet++

这是一个可直接运行的完整多模态训练框架，基于论文思路实现，并进一步升级为 **Bio-MeniscusNet++**：

- 双视图 MRI 主干（Coronal + Sagittal）
- 临床先验编码（年龄/BMI/能量评分/性别/盘状半月板）
- 跨模态门控注意力融合（Cross-View Clinical Gated Attention）
- 多任务学习（分级 + 手术指征）
- 序关系约束（ordinal 辅助头）

## 1. 目录结构

- [config.yaml](config.yaml)
- [train.py](train.py)
- [predict.py](predict.py)
- [create_folds.py](create_folds.py)
- [requirements.txt](requirements.txt)
- [src/datasets/meniscus_dataset.py](src/datasets/meniscus_dataset.py)
- [src/models/biomeniscusnet_pp.py](src/models/biomeniscusnet_pp.py)
- [src/models/losses.py](src/models/losses.py)
- [src/utils/metrics.py](src/utils/metrics.py)

## 2. 数据格式

请准备 `data/metadata.csv`，至少包含以下列：

- `patient_id`
- `coronal_path`（冠状位图像路径，支持 png/jpg/npy）
- `sagittal_path`（矢状位图像路径，支持 png/jpg/npy）
- `age`
- `bmi`
- `energy_score`（0~3）
- `sex`（如 M/F）
- `discoid`（如 0/1）
- `grade`（0~3）
- `surgery`（0/1，可选；若缺失会自动用 `grade>=2` 生成）
- `fold`（0~4，若没有可先生成）

## 3. 安装

1. 安装依赖：

`pip install -r requirements.txt`

## 4. 生成 fold（如果 metadata 没有 fold 列）

`python create_folds.py --input_csv data/metadata.csv --output_csv data/metadata_with_folds.csv`

然后把 [config.yaml](config.yaml) 的 `data.csv_path` 改成 `data/metadata_with_folds.csv`。

## 5. 训练

`python train.py --config config.yaml`

输出：

- 最优模型：`outputs/bio_meniscusnet_pp/best_model.pt`
- 训练日志：`outputs/bio_meniscusnet_pp/history.json`

## 6. 推理

`python predict.py --config config.yaml --checkpoint outputs/bio_meniscusnet_pp/best_model.pt --input_csv data/test_metadata.csv --output_csv outputs/predictions.csv`

## 7. 论文实验建议

基于当前代码可以做：

- 模态消融：只用图像 / 只用临床 / 全融合
- 视图消融：Coronal only / Sagittal only / 双视图
- 生物力学特征消融：去掉 `energy_score`
- 关键指标：QWK、Macro-F1、AUC、Grade3 Recall
