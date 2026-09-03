<div align="center">

# FlowRT-DETR

### 面向交通监控检测闪烁的光流引导 RT-DETR

[![License](https://img.shields.io/badge/license-Apache--2.0-2f80ed.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.8-3776ab.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-ee4c2c.svg)

[English](README.md) · [数据准备](docs/DATA.md) · [模型说明](docs/MODEL_CARD.md) · [复现说明](docs/REPRODUCIBILITY.md) · [BBFR](docs/BBFR.md)

<img src="docs/assets/flowrtdetr_overview.png" width="82%" alt="FlowRT-DETR 总览">

</div>

## 项目简介

逐帧交通目标检测器即使具有较高 mAP，也可能在运动模糊、遮挡和小目标
场景中发生短时漏检与检测框闪烁。FlowRT-DETR 使用冻结的 SEA-RAFT 提取
跨帧运动线索，通过轻量的尺度感知门控（ASG）残差模块融合高分辨率 S3
特征，并提供专门衡量检测闪烁的 Bounding-Box Flicker Rate（BBFR）。

仓库同时包含双帧推理、光流与 ASG 可视化、P/R 评测、延迟测试、退化案例
筛选及排队稳定性分析工具。

> **发布状态：** 当前源码包不包含检测器和 SEA-RAFT 权重。下表来自论文
> 报告结果，尚未在整理后的仓库上重新训练复核。正式对外发布前请阅读
> [复现说明](docs/REPRODUCIBILITY.md)。

## 论文报告结果

论文采用 UA-DETRAC 官方视频划分。BBFR 越低越好，其余精度指标越高越好。

| 方法 | mAP50 | mAP50:95 | mAP75 | P | R | F1 | BBFR-D ↓ | BBFR-T ↓ | 参数量 | FPS |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RT-DETR-R18 | 71.1 | 55.4 | 65.3 | 60.4 | 82.5 | 69.7 | 30.15 | 10.72 | 20.1 M | 92.8 |
| **FlowRT-DETR-R18** | **76.6** | **59.6** | **70.9** | **69.8** | **86.2** | **77.1** | **25.51** | **9.57** | 20.2 M | 28.2 |

论文中的速度在单张 NVIDIA A800-SXM4-80GB 上测得，实际结果会受硬件、
软件环境、输入尺寸和计时范围影响。

## 安装

研究环境为 Python 3.8、PyTorch 2.0.1、torchvision 0.15.2 和 CUDA 11.7。
代码使用了 torchvision beta datapoints API，不建议直接换用新版 torchvision。

```bash
conda create -n flowrtdetr python=3.8 -y
conda activate flowrtdetr
pip install torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

也可使用 `conda env create -f environment.yml` 创建环境。

## 数据与光流权重

将 UA-DETRAC 转为 COCO 检测格式：

```text
data/ua_detrac/
├── annotations/{train.json,val.json}
└── images/{train,val}/MVI_xxxxx/imgxxxxx.jpg
```

标注应使用 `video_id` 或 `file_name` 的父目录保存视频归属，建议同时提供
`frame_id`。详细约定见 [DATA.md](docs/DATA.md)。

从官方 [SEA-RAFT model zoo](https://github.com/princeton-vl/SEA-RAFT#model-zoo)
获取与 `optical_flow/config/kitti-S.json` 兼容的权重，并保存为：

```text
optical_flow/weights/kitti-S.pth
```

## 训练、验证与推理

```bash
# 训练
python tools/train.py \
  -c configs/flowrtdetr/flowrtdetr_r18_uadetrac.yml --amp

# 验证
python tools/train.py \
  -c configs/flowrtdetr/flowrtdetr_r18_uadetrac.yml \
  -r output/flowrtdetr_r18_uadetrac/best.pth --test-only

# 双帧推理
python tools/infer_pair.py \
  -c configs/flowrtdetr/flowrtdetr_r18_uadetrac.yml \
  -r path/to/best.pth \
  --prev-im-file path/to/earlier.jpg \
  --im-file path/to/current.jpg \
  --flow-ckpt optical_flow/weights/kitti-S.pth \
  -o output/inference.jpg
```

发布配置默认 `frame_offset: 2`，它对应现存实验代码快照；如需严格相邻帧，
请设为 `1`。这是会改变实验定义的重要参数。

## BBFR 评测

```bash
python tools/bbfr/eval_bbfr.py \
  -c configs/flowrtdetr/flowrtdetr_r18_uadetrac.yml \
  -r path/to/best.pth \
  --score-thresh 0.3 --iou-thresh 0.5 \
  --max-lost 5 --min-track-len 3 \
  --out-json output/bbfr.json
```

<div align="center">
<img src="docs/assets/bbfr_metric.png" width="66%" alt="BBFR 指标示意图">
</div>

更多命令见 [tools/README.md](tools/README.md)。

## 引用与许可

论文 DOI 或正式出版信息确定后，请更新 [CITATION.cff](CITATION.cff) 和英文
README 中的 BibTeX。项目代码采用 [Apache-2.0](LICENSE)；仓库内 SEA-RAFT
派生代码保留 [BSD 3-Clause](third_party/SEA-RAFT-LICENSE) 许可。项目基于
[RT-DETR](https://github.com/lyuwenyu/RT-DETR) 与
[SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)。
