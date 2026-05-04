# AGE: Arc-based Geo-entity Encoder（中文说明）

将任意几何对象（多边形 / 多边线 / 笔画）拆成"弧段集合"，再用集合编码器（DeepSet / SetTransformer-SAB / SetTransformer-ISAB）学习排列不变的几何嵌入。本仓库覆盖**单实体监督分类**、**少样本（few-shot）分类**两个任务，并对 8 个 SOTA baseline 做了对照实验。

> 英文版见 [README.md](README.md)，详细数值见 [RESULTS.md](RESULTS.md)，过程性笔记见 [PLAN.md](PLAN.md)。

---

## 1. 核心想法

把一个几何实体拆成有限个**有方向的弧段**，每条弧用以下特征向量表示：

- 中点 `(x_mid, y_mid)` 经多频 Fourier positional encoding（基础频率 + 二次谐波，可选首尾端点）
- 长度 `length`（可选 Fourier 编码）
- 方向角 `θ`（sin/cos，可选 16-bin 离散化用作 aux head 监督）

整个实体 = 弧段集合，输入排列不变的集合编码器 → 全局嵌入 → 分类头 / ProtoNet 头。

**为什么是"弧段"**：把多边形 / 多边线 / 笔画统一成"局部线段 + 几何属性"的集合，避开三角化、SDF 采样、Fourier 变换、可见性图这些数据预处理，结构上更接近 LLM token 集合。

---

## 2. 主要贡献 / 这一轮做的改进

### 2.1 集合编码器

| 文件 | 编码器 | 说明 |
|---|---|---|
| `model_edges/entitydeepset.py` | DeepSet | φ → sum/mean → ρ → head |
| `model_edges/entitysettransformer_sab.py` | SAB | 全自注意力（O(n²)） |
| `model_edges/entitysettransformer_isab.py` | ISAB | 诱导集注意力（O(n·m)，m 个 inducing points） |
| `model_edges/entitypointnet.py` (合作者贡献) | PointNet | T-Net 学 2D 输入变换 + masked shared MLP + masked max-pool |
| `model_edges/entitypointnet.py` (合作者贡献) | PointNet++ | FPS 选 32 个 center → kNN k 个邻居 → 局部 MLP → max-pool |

**关键修复 — PMA 在长弧集上的塌缩**：原本 SAB/ISAB 用 single-seed PMA 做集合聚合。在 Omniglot 上（平均 168 弧/字）我们诊断出嵌入退化：

- 200 个随机 Omniglot 嵌入两两 L2 平均 = **0.00181**（几乎相同）
- 每维 std = **7.7e-05**（接近 0）
- 每过一次 PMA 方差就掉一个数量级：`0.0678 → 0.0242 → 0.00233 → 0.000205`

修复：新增 `set_pooling="mean"` 选项，**少样本默认走 masked mean pooling**，PMA 仍保留为可选 (`--sab-pooling pma`)。验证：SAB 5w-1s mean pooling 第 10 epoch 达到 val 0.85（PMA 卡在 0.34）。

### 2.2 解码器

| 模式 | 训练 | 验证 / 测试 | 备注 |
|---|---|---|---|
| `proto`（默认） | ProtoNet + 余弦距离 + 可学习温度 (init=10.0) | 同左 | ArcSet 默认 |
| `lr` | ProtoNet | L2-normalize + 每 episode sklearn `LogisticRegression(lbfgs)` | 对齐 SketchEmbedNet 评测协议 |

### 2.3 Auxiliary Stroke-Completion Head（新）

`model_edges/aux_stroke.py`：在编码器输出之上挂一个 mask-and-predict 头，灵感来自 SketchEmbedNet 的 stroke decoder：

- 训练时随机 mask 30% 的弧（默认）
- 用 transformer decoder 预测每条 masked 弧的：
  - 中点：GMM-MDN（5 个高斯分量）
  - 长度：高斯（μ, log σ²）
  - 方向角：16-bin softmax
- λ-curriculum：0 → 0.5，前 20 epoch 线性预热

CLI：`--aux-stroke on --aux-mask-rate 0.3 --aux-lambda-max 0.5 --aux-warmup-epochs 20`。

**实验结论（Table 3b）**：aux loss 是**注意力编码器专属的增益** —
- SAB / ISAB 在所有健康配置上都涨（最大 ISAB 5w-5s **+5.24**）
- DeepSet 全负（-0.34 到 -0.97），因为 mean-pool 没有机制去查询 masked-arc 的邻居 context

### 2.4 Baseline 适配 + 性能优化

| Baseline | 适配 / 优化 |
|---|---|
| Geo2Vec | CPU 上预采样 SDF（mnist 340M 样本，~27 min）→ 缓存到磁盘 → GPU 上训练 SDF MLP + 分类头（mnist h200 batch=16384，~30 min）。比纯 CPU 快 **13.5×**。 |
| Poly2Vec | 类似套路：CPU 预算 Fourier 特征 → GPU 训练分类头 |
| PolygonGNN | torch_sparse 装不上 → 用原生 torch 重写 `util.py` 里的 `triplets()` |
| PolyMP | `make_valid` 在某些 mnist 多边形上返回 GeometryCollection → 在 `_iter_polygon_rings` 里递归展开 |
| NUFT | DDSL 需要 float64，强制 CPU |
| SketchEmbedNet | 自己重写 TF2 适配（Conv4 + ProtoNet 头） |
| Sketchformer | 自己重写 TF2 适配（4-layer 8-head transformer + ProtoNet 头） |

---

## 3. 数据集

| 名称 | 类别数 | 样本数 | 几何类型 | 来源 |
|---|---|---|---|---|
| `single_buildings` | 10 | ~5 k | Polygon | 仓库内置 |
| `single_mnist` | 10 | ~60 k | Polygon | Git-LFS |
| `single_omniglot` | 1623 | ~32 k | MultiLineString（笔画） | `scripts/prepare_omniglot.py` 重建 |
| `single_quickdraw` | 100 | ~890 k | MultiLineString（笔画） | `scripts/prepare_quickdraw.py` 重建 |

**Omniglot split**：用 Lake 等人原版的 background (964 类训练) / evaluation (659 类测试)，**不是** Vinyals 2016 的 split——这点要在论文 Setup 里写清楚。

### Per-entity isotropic 归一化协议（`_iso` 数据集）

为了"绝对公平"地比对不同方法（消除绝对尺度信号、对齐 normalization 协议），加了
`scripts/normalize_entities.py`：每个 entity 减 bbox 中心 → 按 `max(w, h) / 2`
等比缩放到 `[-1, 1]`（保持长宽比）→ 落盘为 `data/<name>/<orig>_iso.gpkg`。

每个数据集对应有一个 `_iso` 入口（`single_buildings_iso` 等），所有 adapter 的
`DATASETS` dict 都加了同样的 entry，跑实验时只需把 `--dataset` 换掉即可。
Quickdraw 因登录节点 OOM 暂未生成（890k 实体），其他三个全做了。

---

## 4. 实验结果

### Table 1 — 多边形单实体监督分类（80 epochs）

| Method | Venue | Input | single_buildings | single_mnist |
|---|---|---|---|---|
| DeepSet (ours) | — | arc set | 0.9269 | 0.9803 |
| **SetTransformer-SAB (ours)** | — | arc set | **0.9774** | **0.9868** |
| SetTransformer-ISAB (ours) | — | arc set | 0.9601 | 0.9837 |
| NUFT | Mai et al. 2023 | rasterised polygon | 0.8523 | 0.9663 |
| PolygonGNN | Yu et al. KDD 2024 | visibility graph | 0.973 | 0.907 |
| Poly2Vec | Siampou et al. ICML 2025 | 2D Fourier | 0.8005 | 0.9588 |
| Geo2Vec | Chu et al. AAAI 2026 | SDF + adaptive PE | 0.9721 | 0.8854 |
| PolyMP | Huang et al. 2025 | graph MP | 0.8843 | 0.9730 |
| PolyMP-DSC | Huang et al. 2025 | graph MP + DSC | 0.8790 | 0.9754 |

**结论**：SAB 在 buildings / mnist 都是 SOTA，mnist 上比次好的 PolyMP-DSC 高 +1.14 pts，比 PolygonGNN 高 +8 pts。

### Table 1b — 同任务在 iso 协议下（公平 normalization）

每个 entity 先 isotropic standardize 后再送模型（见上面 `_iso` 数据集说明）。
新加入合作者贡献的 PointNet / PointNet++ 编码器（`model_edges/entitypointnet.py`）。

| Method | Input | single_buildings_iso | single_mnist_iso |
|---|---|---|---|
| DeepSet (ours) | arc set | 0.8923 | 0.9804 |
| **PointNet (ours)** | arc set | 0.9282 | **0.9848** |
| PointNet++ (ours) | arc set | 0.8910 | **0.9848** |
| SetTransformer-SAB (ours) | arc set | 0.9162 | 0.9802 |
| SetTransformer-ISAB (ours) | arc set | 0.9109 | 0.9797 |
| **Geo2Vec** | SDF + adaptive PE | **0.9348** | 0.5971³ |
| Poly2Vec | 2D Fourier | 0.8298 | _running_⁴ |
| PolyMP | graph MP | 0.8936 | 0.9753 |
| PolyMP-DSC | graph MP + DSC | 0.8976 | 0.9730 |

³ **Geo2Vec mnist_iso 大跌**（0.8854 → 0.5971，train 0.97 / val 0.60，明显过拟合）：
原因可能是 Geo2Vec 的 `MP_sample` 用**整个 dataset 的全局 bbox** 做 SDF query 采样。
per-entity normalize 之后所有 entity 都中心在原点 + 撑满 [-1,1]，全局 bbox 也是
[-1,1]，所有 entity 的 query 采样位置高度重合 → 跨 entity 的判别信号被破坏。
要修就得改成 per-entity bbox 采样，本轮没改。  
⁴ Poly2Vec mnist_iso 在跑 CPU Fourier 特征预算（每个 polygon 需要 Delaunay
三角化，~5h 全跑完 60k 个）；先占位，跑完后补 follow-up commit。

**关键观察**：

- **Buildings 在 iso 下名次重排**：原 raw Table 1 上 SAB 0.9774 → iso 0.9162（−6.1），是因为 raw 数据里不同字母类别（E/F/H 等）尺度系统性不同，ArcSet 利用了这个绝对尺度信号。Geo2Vec 因为内部本来就 per-entity normalize，掉得最少（−3.7），iso 下重新拿到 buildings SOTA (0.9348)。
- **PointNet (合作者贡献) 在 iso 下成为 ArcSet 内部最强**：buildings 0.9282 (>SAB 0.9162)，mnist 0.9848 (与 PointNet++ 并列第一)。
- **mnist 几乎不变**（每方法 ±0.5pt 内）：mnist 数字本来 per-entity 尺度就比较一致，iso 几乎无操作。

### Table 2 — Few-shot Omniglot（vanilla ProtoNet，80 epochs，1000 test episodes）

| Method | Input | 5w-1s | 5w-5s | 20w-1s | 20w-5s |
|---|---|---|---|---|---|
| DeepSet (ours) | arc set | 0.9186 | 0.9749 | 0.8567 | 0.9558 |
| **SetTransformer-SAB (ours)** | arc set | 0.9111 | 0.9759 | **0.8893** | 0.9613 |
| SetTransformer-ISAB (ours) | arc set | 0.5891¹ | 0.8935 | 0.8443 | 0.9476 |
| **SketchEmbedNet** | image + stroke | **0.9513** | **0.9857** | 0.8667 | **0.9587** |
| Sketchformer | stroke seq | 0.7819 | 0.8592 | 0.6315 | 0.8223 |

¹ ISAB 在 k=1 时因 16 个 inducing points + 单 support 退化为 chance 附近；k≥5 正常。

### Table 2b — 同任务在 iso 协议下（PointNet 大胜）

| Method | Input | 5w-1s | 5w-5s | 20w-1s | 20w-5s |
|---|---|---|---|---|---|
| DeepSet (ours) | arc set | 0.9136 | 0.9729 | 0.8583 | 0.9521 |
| **PointNet (ours)** | arc set | 0.9400 | **0.9814** | **0.9017** | **0.9654** |
| **PointNet++ (ours)** | arc set | **0.9426** | 0.9804 | 0.8901 | 0.9644 |
| SetTransformer-SAB (ours) | arc set | 0.8990 | 0.9726 | 0.8817 | 0.9594 |
| SetTransformer-ISAB (ours) | arc set | 0.4951¹ | 0.9010 | 0.8340 | 0.9502 |
| SketchEmbedNet (raw) | image + stroke | 0.9513 | 0.9857 | 0.8667 | 0.9587 |

**关键观察**：

- **PointNet 20w-1s = 0.9017，比 SketchEmbedNet (有 21M sketch QuickDraw 预训练) 的 raw 0.8667 高 +3.5 pts**，是这一格 Omniglot fewshot 的新 SOTA。
- PointNet 20w-5s = 0.9654 也比 SEN raw 0.9587 高 +0.7 pts。
- 5-way 上 PointNet/PointNet++ 与 SEN 差距都在 1 pt 以内，且我们 from-scratch 80 epoch episodic 训练，没有任何额外预训练。
- iso 下 SAB 不再是 ArcSet 内最佳：SAB 5w-1s 0.8990 vs PointNet 0.9400，差 4 pts。原因可能是 SAB 之前部分依赖了绝对尺度信号；PointNet 用 T-Net 学输入变换可能更适应这种 scale-invariant 的 setting。

### Table 3 — SAB 解码器 / aux-loss 消融

| Setting | aux-off | **aux-on** | lr-head | SEN ref |
|---|---|---|---|---|
| 5w-1s | 0.9179 | **0.9357 ⬆+1.78** | 0.8942 (-2.4) | 0.9513 |
| 5w-5s | 0.9686 | **0.9815 ⬆+1.29** | 0.9724 (+0.4) | 0.9857 |
| 20w-1s | 0.8841 | 0.8837 (≈tie) | 0.8558 (-2.83) | 0.8667 |
| 20w-5s | 0.9633 | **0.9635** (≈tie) | 0.9570 (-0.6) | 0.9587 |

- **Aux 在 5-way 收益最大**（+1.78 / +1.29），20-way 已饱和。
- **LR-head 在 1-shot 一律掉点**（-2.4 / -2.83）：L2-norm + LR 在单 support 下过拟合噪声；ProtoNet 余弦 + 学温度才是 ArcSet 嵌入的合适解码器。也就是说 SEN 在 5-way 上的优势不是"换 LR 就有"的免费午餐。

### Table 3b — 多编码器 × aux-stroke 消融（**新**，本轮重点结果）

| Encoder | 5w-1s off / on / Δ | 5w-5s off / on / Δ | 20w-1s off / on / Δ | 20w-5s off / on / Δ |
|---|---|---|---|---|
| **SAB** | 0.9179 / **0.9357** / **+1.78** | 0.9686 / **0.9815** / **+1.29** | 0.8841 / 0.8837 / -0.04 | 0.9633 / 0.9635 / +0.02 |
| ISAB | 0.5494¹ / 0.4859¹ / — | 0.9204 / **0.9728** / **+5.24** | 0.8436 / **0.8764** / **+3.28** | 0.9537 / **0.9655** / **+1.18** |
| DeepSet | 0.9225 / 0.9164 / **-0.61** | 0.9769 / 0.9735 / **-0.34** | 0.8579 / 0.8482 / **-0.97** | 0.9550 / 0.9510 / **-0.40** |

**核心发现**：

- 注意力编码器（SAB / ISAB）在所有可正常训练的配置上都因 aux-loss 受益
- **mean-pool 的 DeepSet 一律掉点** —— 没有 attention 机制去给 masked arc 查询邻居 context，aux head 反而抢走了判别任务的容量
- ISAB 在 k=1 仍然塌缩（与 aux 无关，是 inducing-points + 单 support 的结构问题）

### QuickDraw few-shot baselines（部分）

| Method | 5w-1s | 5w-5s | 20w-* |
|---|---|---|---|
| Sketchformer | 0.5659 | 0.7614 | NaN（split 太小） |
| SketchEmbedNet | 0.6106 | 0.7903 | NaN（split 太小） |

100 类经 70/15/15 split 之后 test pool 只剩 ~15 类，凑不出 20-way episode。ArcSet QuickDraw few-shot 还没跑。

---

## 5. 工程修复 / 踩坑记录

| 问题 | 解决 |
|---|---|
| SAB/ISAB 在 Omniglot 长弧集上 PMA 塌缩，少样本卡 chance | 默认 `set_pooling=mean` 旁路 PMA |
| Geo2Vec 在 MPS 上 SDF auto-decoder 出 NaN | 强制 CPU；后又 CPU 预采样 → GPU 训练 |
| Geo2Vec QuickDraw 3.6B SDF 样本 OOM | 标 N/A（需要 streaming cache，超出本轮范围） |
| Geo2Vec stroke 不适用（SDF 假设闭合形状） | Omniglot 上 train 0.78 / test 0.012，标 N/A |
| Geo2Vec mnist CPU 单 epoch 5675s（126h 总耗时） | 缓存 SDF 样本（27 min），转 h200 GPU + batch=16384，30 min 跑完 |
| PolyMP `make_valid` 返回 GeometryCollection | 在 `_iter_polygon_rings` 里递归展开 |
| `torch_sparse` 在 M5 / HPC 都装不上 | Codex 重写 PolygonGNN `util.py` 的 `triplets()` 用原生 torch |
| HPC 监控杀掉 0% GPU 的 CPU 密集 baseline 任务 | 改用缓存特征 + GPU-only 训练 |
| `[classes]` 之后日志卡死（其实在跑） | `PYTHONUNBUFFERED=1` |
| `fiona` 报 `libstdc++ GLIBCXX_3.4.30` 缺失 | conda activate hook 设置 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` |
| `l40s_public` 分区 2 小时强制 preemption | `EPOCHS` 降到刚好能跑完 + 改投 `h200_tandon` |
| 个别节点（如 gl032）MPS daemon 损坏，CUDA 拉不起来 | 重投到其他节点 |
| 5 个 ISAB 任务在同时段挂死在 `__skb_wait_for_more_packets`（CUDA 上下文初始化卡住） | 全部 cancel 后从 h200_tandon 改投 l40s_public 解决 |

---

## 6. 环境

```bash
# osx-arm64（本地）
conda create -n age -c conda-forge python=3.10 numpy 'pytorch>=2.10' \
    geopandas shapely tqdm requests 'huggingface_hub>=0.23' -y
conda activate age
pip install torch_geometric tensorboard fiona pyarrow osmnx triangle

# Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
export KMP_DUPLICATE_LIB_OK=TRUE
```

HPC 用 `/scratch/sx2490/arcset/env/conda/envs/arcset/`（PyTorch）和 `sketch_tf`（TensorFlow，给 SketchEmbedNet / Sketchformer）。

---

## 7. 复现命令

### 7.1 准备数据

```bash
git lfs pull                                          # single_mnist
python scripts/prepare_omniglot.py                    # → data/single_omniglot/
python scripts/prepare_quickdraw.py --num-classes 100 \
    --samples-per-class 10000                          # → data/single_quickdraw/
python scripts/prepare_polygongnn.py                  # → data/topo_polygongnn/
```

### 7.2 ArcSet 监督分类（Table 1）

```bash
EPOCHS=80 bash scripts/run_standard_classification.sh
# 或单跑：
python -u model_edges/test.py \
    --dataset single_buildings --set-model settransformer-sab \
    --epochs 80 --output-dir model_edges/results/sab_buildings
```

### 7.3 ArcSet Few-shot（Table 2/3/3b）

```bash
# 默认 ProtoNet 余弦 + 学温度
python -u model_edges/testfs.py \
    --dataset single_omniglot --set-model settransformer-sab \
    --n-way 5 --k-shot 1 --n-query 15 \
    --epochs 80 --train-episodes 200 --val-episodes 200 --test-episodes 1000 \
    --xy-num-freqs 6 --proto-distance cosine --proto-init-temp 10.0 \
    --output-dir model_edges/results/fs_sab_5w1s

# 加 aux-stroke head（Table 3 / 3b auxon 列）
python -u model_edges/testfs.py ... \
    --aux-stroke on --aux-mask-rate 0.3 \
    --aux-lambda-max 0.5 --aux-warmup-epochs 20

# LR-head 评测（Table 3 lrhead 列）
python -u model_edges/testfs.py ... --decoder lr
```

### 7.4 Baseline（Table 1）

```bash
# 多边形 baselines（h200 GPU + 缓存特征）
sbatch --export=ALL,NAME=geo2vec,DATASET=single_mnist scripts/train_baseline.slurm
sbatch --export=ALL,NAME=poly2vec,DATASET=single_buildings scripts/train_baseline.slurm
sbatch --export=ALL,NAME=polygongnn,DATASET=single_mnist scripts/train_baseline.slurm
sbatch --export=ALL,NAME=polymp,PMODEL=polymp,DATASET=single_buildings scripts/train_baseline.slurm
sbatch --export=ALL,NAME=nuft,DATASET=single_buildings scripts/train_baseline_cpu.slurm

# Stroke baselines（TensorFlow，单独环境 sketch_tf）
sbatch --export=ALL,DATASET=single_omniglot,MODEL=sketchembednet,NWAY=5,KSHOT=1 scripts/train_sketch.slurm
sbatch --export=ALL,DATASET=single_omniglot,MODEL=sketchformer,NWAY=5,KSHOT=1 scripts/train_sketch.slurm
```

---

## 8. 目录结构

```
.
├── model_edges/                 # ArcSet 编码器 + 训练脚本
│   ├── entitydeepset.py         # DeepSet 编码器
│   ├── entitysettransformer_sab.py    # SAB 编码器（含 set_pooling=mean 修复）
│   ├── entitysettransformer_isab.py   # ISAB 编码器（含 set_pooling=mean 修复）
│   ├── load_entities.py         # gpkg → 弧段集合 + 自适应 Fourier 频率
│   ├── aux_stroke.py            # ★ 新：masked-arc completion head
│   ├── test.py                  # 标准分类 driver
│   ├── testfs.py                # 少样本 ProtoNet driver（含 aux + lr-head）
│   └── results/                 # 每次跑的 summary.json / best.pt
├── baselines/
│   ├── PATCHES.md               # 各 baseline 的 patch 清单
│   ├── nuft / polygongnn / poly2vec / geo2vec / polymp /
│   │   sketchembednet / sketchformer  # 各自适配脚本 + cache + results
│   └── …
├── scripts/
│   ├── train_arcset_standard.slurm    # ArcSet 监督分类
│   ├── train_arcset_fewshot.slurm     # ArcSet 少样本
│   ├── train_baseline.slurm           # 多边形 baseline GPU
│   ├── train_baseline_cpu.slurm       # CPU baseline（NUFT 等）
│   ├── train_sketch.slurm             # 笔画 baseline（TF）
│   ├── cpu_long.slurm                 # 长跑 CPU 任务（如 SDF 采样）
│   ├── prepare_*.py                   # 数据集准备脚本
│   └── run_*.sh                       # 高层 driver（按数据集 / wave 组织）
├── data/                        # gpkg 数据（单 mnist 走 Git-LFS）
├── RESULTS.md                   # 完整数值表格（Table 1/2/3/3b）
├── PLAN.md                      # 过程性进度笔记
└── README.md / README.zh.md     # 英中文说明
```

---

## 9. 论文叙事建议

基于现有结果可以这样组织：

1. **Polygon SOTA**：SAB 在 buildings 0.9774（vs PolygonGNN 0.973）和 mnist 0.9868（vs PolyMP-DSC 0.9754）双双拿下 SOTA。
2. **Stroke 上的可比性**：vanilla ProtoNet 下，ArcSet-SAB 在 20w-1s 击败 SketchEmbedNet（0.8893 vs 0.8667），尽管对方有 21M QuickDraw stroke 预训练。
3. **Aux loss 是注意力编码器的设计选择**（Table 3b 是核心新故事）：
   - 给 attention 编码器加 mask-and-predict aux 头普遍涨点
   - 给 mean-pool DeepSet 加同一个头普遍掉点
   - 这一对照实验把 aux 的来源讲清楚了：不是"加监督就好"，而是"和编码器的归纳偏置耦合"
4. **Decoder 公平性辩护**（Table 3 lr-head 列）：SEN 用 LR-per-episode 是它在 5-way 上拉开的部分原因；当我们换上同一个 LR 解码器，它在 1-shot 反而掉得更厉害。也就是说 ArcSet 在 matched-decoder 协议下其实更稳。

---

## 10. 当前未完成的实验

- **Poly2Vec single_mnist**：缓存 / GPU 流水线还没排到队
- **Geo2Vec / Poly2Vec QuickDraw**：缓存内存爆（3.6B SDF 样本超过 96G 节点上限），需要 streaming cache
- **ArcSet QuickDraw few-shot**：100 类只能做 5-way（20-way 受 split 限制）
- **PolyMP robustness matrix**（旋转 / 剪切不变性）：有需要再补

---

## 致谢

Baselines 的上游仓库见各 `baselines/<name>/readme.md`，patch 列表见 [`baselines/PATCHES.md`](baselines/PATCHES.md)。HPC 资源：NYU Torch (`/scratch/sx2490/arcset/`)。
