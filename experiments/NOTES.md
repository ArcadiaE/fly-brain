# 本地环境与测试记录

仓库：https://github.com/eonsystemspbc/fly-brain （浅克隆，2026-09-17）
本地路径：`F:\Claude项目\fly-brain`

## 环境

- 机器：ASUS GU603ZM，RTX 3060 Laptop 6 GB，Windows 11
- Python 3.10.11 venv：`.venv`（`py -3.10 -m venv .venv`）
- torch 2.14.0+cu126（`--index-url https://download.pytorch.org/whl/cu126`）
- numpy 1.26.4, pandas, pyarrow, scipy, matplotlib, tqdm, joblib
- brian2 2.8.0.4（`--only-binary=:all:`，源码构建会失败）
- 未安装：Brian2CUDA / GeNN / NEST GPU（需要 Linux/WSL + CUDA toolkit）

激活：
```
.venv\Scripts\activate
python experiments/smoke_test.py --exp sugar --t 1
```

## 模型

- 数据：FlyWire v783，138,639 个神经元，15,091,983 个非零突触权重
- 模型：Shiu et al. 2024 (Nature) 的 LIF + alpha 突触，dt = 0.1 ms
- PyTorch 后端 `code/run_pytorch.py` 自带 ATan 代理梯度，可微
- 首次运行会在 `data/` 下生成 `weight_coo.pkl` / `weight_csr.pkl` 缓存（各约 290 MB，已 gitignore）

## 冒烟测试结果（2026-09-17）

| 实验 | batch | 模拟时长 | 墙钟时间 | 每步耗时 | 峰值显存 | 活跃神经元 |
|---|---|---|---|---|---|---|
| sugar GRN 200 Hz | 1 | 1.0 s | 9.4 s | 0.94 ms | 0.20 GB | 383–394 |
| P9 前进 100 Hz | 1 | 1.0 s | 9.4 s | 0.94 ms | 0.20 GB | 61 |
| sugar GRN 200 Hz | 8 | 0.5 s | 85.6 s | 17.1 ms | 0.39 GB | 366 |

- batch=1 时约 0.1 倍实时。要做实时交互需要提速约 10 倍。
- batch=8 每步反而慢 18 倍，见下方稀疏矩阵乘法基准。

## 稀疏矩阵乘法基准（batch 1 vs 8，每步一次 W^T·spikes）

| 布局 | batch 1 | batch 8 |
|---|---|---|
| 当前 `matmul(s, csr.T)` | 1.20 ms | 17.9 ms |
| `csr @ s.T` | 0.83 ms | 15.8 ms |
| 预转置 W^T csr | 0.85 ms | 15.5 ms |
| coo @ s.T | 2.42 ms | 12.8 ms |

结论：batch>1 的慢是 cuSPARSE 批量 SpMM 本身，不是布局问题。
batch=1 时 SpMM 就是每步的主要开销。

提速方向：每步实际只有约 2 个脉冲（16k 脉冲 / 10k 步），
用事件驱动方式只取放电神经元对应的 W^T 行（index_select）再累加，
理论上能省掉 99% 以上的乘法。这是做实时交互的关键优化。

## 可视化

`python experiments/visualize.py --exp sugar --t 1` → `experiments/out/sugar_t1.0s.png`
栅格图按首次放电延迟排序，可看到从味觉神经元向下游逐层传播的级联。

## 三维可视化（2026-09-17 晚）

数据：`data/annotations/Supplemental_file1_neuron_annotations.tsv`
（flyconnectome/flywire_annotations，Schlegel 2024 + Berg 2025，CC BY 4.0）。
138,625 / 138,639 个模型神经元有坐标；附带 super_class、cell_type、递质预测、左右侧。

验证：糖味觉预设刺激的 21 个神经元全是左侧 LB3 味觉神经元；
下游放电最强的包括 CB0700、MNx01 等脑运动神经元（吻管伸出），生物学上合理。

### 离线回放  `experiments/export_3d.py`
跑两遍仿真（第一遍找最活跃下游，第二遍同种子录膜电位），把坐标、脉冲、
12 条膜电位波形打包进 `brain3d_template.html`，输出单文件
`experiments/out/brain3d_<exp>.html`（约 3 MB，双击打开，需联网加载 three.js）。
`run_replay.bat` 一键生成并打开。

### 实时控制台  `experiments/live_server.py` + `brain_live.html`
aiohttp 服务：仿真在后台线程跑 GPU，每 20 步（2 ms 脑时间）通过 WebSocket
推一帧二进制（放电神经元索引 + 探针膜电位）。页面功能：
- 预设实验一键叠加；按细胞类型搜索后点击即刺激（可选左右侧、频率）
- 当前刺激列表可逐组移除；暂停/继续/重置/清空
- 三维脑：放电闪黄、刺激红、探针蓝；悬停看类型；点击加入示波器
- 示波器：最近 200 ms 膜电位；"示波器→最活跃"自动选最活跃的 8 个下游
- 日志面板
`run_live.bat` 一键启动。也可 `preview_start fly-brain-live`（F:\.claude\launch.json）。

性能：每步取一次脉冲会导致每步 GPU 同步，2.5 ms/步；改为每块取一次后回到约 1 ms/步。

## 提速到实时（2026-09-17 深夜）`experiments/fast_brain.py`

诊断：GPU 空闲时稠密 SpMV 816 µs/步；但一个 138k 元素的加法也要 16 µs，
一步约 75 个内核 → Windows WDDM 驱动的内核启动开销才是主瓶颈，不只是算法。
**注意**：基准测试前必须停掉 live_server，否则数字翻倍。

方案（FastBrain）：
1. 事件驱动：W^T 按 256 宽切成定宽"突触块"（151k 块，零填充，坏地址指向垃圾区），
   每步用 cumsum 压缩把放电神经元的块 id 塞进 1024 个固定槽位，gather + index_add。
2. 全部张量定形 → 整块 S 步录成一张 CUDA Graph，一次启动。
3. 延迟环长度 D=延迟步数+1，S 取 D 的倍数，环槽位变成编译期常量。
4. LIF 用 lerp/where/addcmul 合并，每步约 33 个内核。
5. 脉冲导出直接复用压缩后的活动块列表（主机侧映射回神经元）。

| 版本 | dt | µs/步 | 实时比 |
|---|---|---|---|
| 原 run_pytorch（稠密 SpMV） | 0.1 | 940 | 0.10× |
| FastBrain 非 Graph | 0.1 | 2209 | 0.05× |
| FastBrain + Graph v1 | 0.1 | 372 | 0.27× |
| FastBrain + Graph v2（合并内核） | 0.1 | 224 | 0.45× |
| FastBrain + Graph v2 | 0.25 | 175–252 | **≈1.0×** |

精度验证 `experiments/validate_fast.py`（糖味觉 1 s，逐神经元放电率）：
| 对比 | Jaccard(活跃集) | 放电率相关 |
|---|---|---|
| 参考 vs 参考（不同种子） | 0.963 | 0.994 |
| 参考 vs Fast dt 0.1 | 0.905 | 0.993 |
| 参考 vs Fast dt 0.25 | 0.934 | 0.993 |
| 参考 vs Fast dt 0.5 | 0.904 | 0.994 |
→ 新内核与参考的差异 ≈ 随机涨落，dt 0.25 作为实时模式。

显存：约 0.45 GB（突触块表 int64+float32 约 470 MB）。溢出（>1024 活动块/步）会计数并在页面日志警告。

再提速的路：Triton（Windows 需 triton-windows 社区包）或自写 CUDA 内核把 33 个内核融合成 3 个，可到 3–5× 实时。

## 实时控制台更新
- 默认用 FastBrain dt 0.25，`--dt 0.1` 参考步长，`--dense` 回退旧模型，`--speed 0` 不限速
- 页面：速度上限下拉；探针列表可逐个移除/清空，也可点示波器左侧名字移除；帧格式改为每块一帧（含 S 步探针电压）

## 验证矩阵全量结果（2026-10-12）`experiments/validate_matrix.py --full`

sugar + p9 × 0.5/1.0 s × 5 种子 × dt {0.1, 0.25, 0.5} + sugar 10 s 漂移。
结果：`out/validation/validate_matrix_20261012_113944.csv`（134 行）。

| 条件 | rate_corr | jaccard | count_ratio | 脉冲精度(±2ms) |
|---|---|---|---|---|
| 参考换种子地板（sugar） | 0.991 | 0.928 | 0.997 | 0.342 |
| FastBrain（sugar，全部 dt） | 0.990 | 0.92 | ~1.0 | 0.34 |
| 参考换种子地板（p9，低活动） | 0.976 | 0.543 | 0.961 | 0.141 |
| FastBrain（p9，全部 dt） | ~0.97 | ~0.6 | ~1.06 | ~0.12 |

- 所有指标在所有 dt 下与参考模型自身种子噪声不可区分；dt 0.5 无退化。
- p9 的 Jaccard 两边都低是因为活跃神经元仅 ~60 个，采样涨落大——地板本身也只有 0.54。
- 10 s 漂移：逐秒 rate_corr 0.989–0.996，count_ratio 0.95–1.03，无发散。
- 结论：FastBrain 引擎通过系统验证，可作为 Phase 1 研究仪器。
