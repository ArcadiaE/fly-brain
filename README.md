<div align="center">

# FastBrain · 果蝇全脑实时仿真

**Real-time whole-brain *Drosophila* simulation on consumer GPUs**
**在消费级 GPU 上实时运行果蝇全脑模型**

[中文](#zh) · [English](#en) · [项目展示页 Project site](https://arcadiae.github.io/fly-brain/)

<img src="docs/assets/hero_brain.png" alt="Whole fly brain with firing neurons highlighted" width="100%">

</div>

---

<a id="zh"></a>
# 中文版

**目录**：[项目简介](#zh-intro) · [本分支新增](#zh-new) · [核心结果](#zh-results) · [实时控制台](#zh-console) · [快速开始](#zh-quick) · [项目结构](#zh-structure) · [数据](#zh-data) · [研究路线](#zh-roadmap) · [许可证与致谢](#zh-license)

<a id="zh-intro"></a>
## 项目简介

这个仓库在一张 6 GB 的游戏显卡上实时跑完果蝇全脑模型的仿真：138,639 个神经元、约 1500 万个突触，0.5 秒的脑活动 0.98 秒算完，是上游 PyTorch 后端速度的 27 倍。

仓库 fork 自 [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain)（Eon Systems PBC）。上游实现了 Shiu et al. 2024 发表在 [Nature](https://www.nature.com/articles/s41586-024-07763-9) 的果蝇全脑渗漏积分发放模型，连接组数据来自 [FlyWire](https://flywire.ai/) v783 公开版。模型支持对任意神经元集合做激活或沉默，看脉冲怎么在全脑传播，上游还提供了 Brian2、Brian2CUDA、PyTorch、NEST GPU、GeNN、Brian2GeNN 六个后端的统一基准。本分支在其上加了第七个后端 FastBrain，以及一套能在浏览器里边跑边看的实时控制台。

<div align="center">
<img src="docs/assets/ai_fly_macro.png" alt="A fruit fly on a dewy grass blade" width="70%">
</div>

[返回目录](#zh)

<a id="zh-new"></a>
## 本分支新增

| 组件 | 说明 |
|---|---|
| FastBrain 引擎 `experiments/fast_brain.py` | 事件驱动突触块压缩加 CUDA Graph 整段回放，每步 CUDA 内核数从约 75 降到约 33，显存占用约 0.45 GB |
| 第七个基准后端 `code/run_fastbrain.py` | 接入上游基准体系（`main.py --fastbrain`），与其他后端同口径计时、同格式导出脉冲 |
| 系统验证 `experiments/validate_matrix.py` | 刺激 × 时长 × 种子 × dt 全矩阵验证，含种子噪声地板和 10 秒漂移检查 |
| 实时控制台 `experiments/live_server.py` | 浏览器里实时看全脑放电，可施加刺激、加探针、暂停重置，用法见[控制台教程](experiments/LIVE_CONSOLE.md) |
| 3D 回放 `experiments/export_3d.py` | 把一次仿真导出成单文件 HTML，含神经元坐标、脉冲动画和膜电位波形（`run_replay.bat`） |
| 项目展示页 [`docs/index.html`](docs/index.html) | 公开的项目网站，含交互 demo，[在线打开](https://arcadiae.github.io/fly-brain/) |

[返回目录](#zh)

<a id="zh-results"></a>
## 核心结果

糖味觉 GRN 200 Hz 刺激，仿真时长 0.5 s，硬件为 RTX 3060 Laptop（6 GB 显存）：

| 后端 | 运行次数 | 仿真耗时 | 实时比 |
|---|---|---|---|
| PyTorch (CUDA) | 1 | 26.30 s | 0.019× |
| FastBrain dt 0.25 | 1 | 0.98 s | 0.51× |
| FastBrain dt 0.1 | 1 | 2.43 s | 0.21× |

各后端的耗时对比如下图。PyTorch 后端跑 0.5 秒的脑活动要 26 秒，FastBrain 在 dt 0.25 下不到 1 秒，基本实现实时。

<img src="docs/assets/chart_speed.png" alt="Benchmark chart" width="100%">

速度不能以牺牲精度为代价。验证矩阵覆盖 2 种刺激 × 2 种时长 × 5 个种子 × 3 种 dt，FastBrain 与参考实现的差异在所有指标上都落在参考模型自身的种子噪声之内：放电率相关系数 0.990，参考模型自身换种子的噪声地板是 0.991。10 秒长程仿真无漂移。下图是糖刺激实验的全脑脉冲栅格，完整验证数据见[展示页](https://arcadiae.github.io/fly-brain/)。

<img src="docs/assets/raster_sugar.png" alt="Spike raster of the sugar experiment" width="100%">

[返回目录](#zh)

<a id="zh-console"></a>
## 实时控制台

`run_live.bat` 会启动一个本地服务并在浏览器打开控制台页面。页面主体是全脑三维动画，每个亮点是一次神经元放电；侧栏可以按细胞类型选择神经元施加刺激或沉默，可以给任意神经元挂膜电位探针（最多 16 个），可以随时暂停、继续和重置。仿真在 GPU 上边算边推流，所有操作下一帧生效。

完整的操作说明和复现步骤单独写在 [experiments/LIVE_CONSOLE.md](experiments/LIVE_CONSOLE.md)。

[返回目录](#zh)

<a id="zh-quick"></a>
## 快速开始

```bash
conda env create -f environment.yml && conda activate brain-fly   # 完整后端（Linux/WSL2）
# 本机最小环境（Windows）：Python 3.10 venv + torch(cu126) + brian2，见 experiments/NOTES.md

python main.py --fastbrain --t_run 0.5 --n_run 1          # FastBrain 基准
python experiments/validate_matrix.py --quick             # 引擎验证（冒烟）
python experiments/validate_matrix.py --full              # 引擎验证（完整矩阵）
run_live.bat                                              # 实时交互控制台
run_replay.bat                                            # 生成并打开 3D 回放
```

[返回目录](#zh)

<a id="zh-structure"></a>
## 项目结构

```
fly-brain/
├── main.py                     # 基准入口（CLI）
├── code/
│   ├── benchmark.py            # 调度器：配置、日志、分发
│   ├── run_fastbrain.py        # FastBrain 基准后端（本分支新增）
│   ├── run_pytorch.py          # PyTorch 参考后端（含代理梯度）
│   └── ...                     # Brian2 / Brian2CUDA / NEST GPU / GeNN / Brian2GeNN
├── experiments/
│   ├── fast_brain.py           # FastBrain 引擎
│   ├── validate_matrix.py      # 系统验证矩阵
│   ├── live_server.py          # 实时控制台服务
│   ├── LIVE_CONSOLE.md         # 控制台使用教程
│   ├── NOTES.md                # 实验日志（全部性能/精度数字）
│   └── RESEARCH.md             # 研究情报与路线图
├── docs/
│   ├── index.html              # 项目展示页
│   └── assets/                 # 数据图表、截图与插画
└── data/                       # FlyWire v783 数据、基准结果、spike parquet
```

[返回目录](#zh)

<a id="zh-data"></a>
## 数据

模型使用 FlyWire 连接组 v783 公开版。基准输出在 `data/results/local_fastbrain/`，格式为一脉冲一行的 parquet，列包括 `time_ms`、`trial`、`neuron_index`、`flywire_id`、`exp_name`。论文配套数据（`nature_2026_07`，600 个 parquet）存在 Google Drive，校验方式见[上游 README](README_UPSTREAM.md)。

[返回目录](#zh)

<a id="zh-roadmap"></a>
## 研究路线

- Phase 0（已完成）：引擎系统验证，结果见上方核心结果。
- Phase 1（进行中）：把脉冲全脑模型和公开钙成像数据做动力学级基准（DANDI [001869](https://dandiarchive.org/dandiset/001869)、Aimon 2019 CRCNS），并对已发表结论做敏感性和 null 审计。
- Phase 2（待定）：贝叶斯优化刺激逆向设计，或把蘑菇体学习嵌入全脑。

每条结论都要带 degree-matched 重连 null 对照，这是这个仓库的硬性规矩。详细情报见 [experiments/RESEARCH.md](experiments/RESEARCH.md)。

<div align="center">
<img src="docs/assets/ai_robot_embodied.png" alt="Concept: fly-brain-driven robot" width="70%">
</div>

[返回目录](#zh)

<a id="zh-license"></a>
## 许可证与致谢

GPL-2.0-or-later（见 [LICENSE](LICENSE)）。上游 © Eon Systems PBC；Shiu et al. 原始代码（`code/paper-phil-drosophila/`）为 MIT。详细安装说明（NEST GPU 源码编译、Brian2GeNN 独立环境等）见 [README_UPSTREAM.md](README_UPSTREAM.md)。

[返回目录](#zh)

<div align="center">
<img src="docs/assets/ai_viola_fly.png" alt="Viola smiling at a fruit fly on her fingertip" width="70%">
</div>

---

<div align="center">
<img src="docs/assets/ai_fly_cyber.png" alt="Cyber-style Drosophila illustration" width="60%">
</div>

---

<a id="en"></a>
# English Version

**Contents**: [Introduction](#en-intro) · [What's new in this fork](#en-new) · [Key results](#en-results) · [Live console](#en-console) · [Quickstart](#en-quick) · [Project structure](#en-structure) · [Data](#en-data) · [Research roadmap](#en-roadmap) · [License & credits](#en-license)

<a id="en-intro"></a>
## Introduction

This repository simulates the whole fruit-fly brain in real time on a 6 GB gaming GPU: 138,639 neurons and about 15 million synapses, with 0.5 s of brain activity computed in 0.98 s — 27 times faster than the upstream PyTorch backend.

It is a fork of [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain) (Eon Systems PBC). The upstream implements the whole-brain leaky integrate-and-fire model from Shiu et al. 2024 ([Nature](https://www.nature.com/articles/s41586-024-07763-9)), built on the public [FlyWire](https://flywire.ai/) v783 connectome. The model can activate or silence arbitrary neuron sets and shows how spikes propagate across the brain, and upstream ships a unified benchmark across six backends: Brian2, Brian2CUDA, PyTorch, NEST GPU, GeNN and Brian2GeNN. This fork adds a seventh backend, FastBrain, plus a live console that lets you watch and perturb the running simulation in a browser.

<div align="center">
<img src="docs/assets/ai_fly_macro.png" alt="A fruit fly on a dewy grass blade" width="70%">
</div>

[Back to contents](#en)

<a id="en-new"></a>
## What's new in this fork

| Component | Description |
|---|---|
| FastBrain engine `experiments/fast_brain.py` | Event-driven synapse-block compression with whole-graph CUDA Graph replay; about 33 CUDA kernels per step instead of about 75; roughly 0.45 GB of VRAM |
| 7th benchmark backend `code/run_fastbrain.py` | Integrated into the upstream benchmark (`main.py --fastbrain`), with the same timing conventions and spike-export schema as the other backends |
| Systematic validation `experiments/validate_matrix.py` | Full stimulus × duration × seed × dt matrix, with a seed-noise floor and a 10 s drift check |
| Live console `experiments/live_server.py` | Watch whole-brain firing in a browser while you stimulate, silence, probe, pause and reset; see the [console guide](experiments/LIVE_CONSOLE.md) |
| 3D replay `experiments/export_3d.py` | Exports a run as a single-file HTML with neuron coordinates, spike animation and voltage traces (`run_replay.bat`) |
| Project site [`docs/index.html`](docs/index.html) | Public showcase page with interactive demos, [open online](https://arcadiae.github.io/fly-brain/) |

[Back to contents](#en)

<a id="en-results"></a>
## Key results

Sugar GRN 200 Hz stimulation, 0.5 s of simulated time, on an RTX 3060 Laptop (6 GB VRAM):

| Backend | Runs | Sim time | Real-time ratio |
|---|---|---|---|
| PyTorch (CUDA) | 1 | 26.30 s | 0.019× |
| FastBrain dt 0.25 | 1 | 0.98 s | 0.51× |
| FastBrain dt 0.1 | 1 | 2.43 s | 0.21× |

The chart below compares all backends. PyTorch needs 26 seconds for half a second of brain activity; FastBrain at dt 0.25 finishes in under a second, which is effectively real time.

<img src="docs/assets/chart_speed.png" alt="Benchmark chart" width="100%">

Speed is useless if accuracy slips. The validation matrix covers 2 stimuli × 2 durations × 5 seeds × 3 dt values, and on every metric the difference between FastBrain and the reference stays inside the reference model's own seed noise: firing-rate correlation 0.990, against a seed-noise floor of 0.991 for the reference itself. No drift over a 10 s run. Below is the whole-brain spike raster of the sugar experiment; full validation data is on the [project site](https://arcadiae.github.io/fly-brain/).

<img src="docs/assets/raster_sugar.png" alt="Spike raster of the sugar experiment" width="100%">

[Back to contents](#en)

<a id="en-console"></a>
## Live console

`run_live.bat` starts a local server and opens the console in your browser. The main view is a 3D animation of the whole brain where each flash is one neuron firing. In the side panel you can stimulate or silence neurons by cell type, attach membrane-voltage probes to any neurons (up to 16), and pause, resume or reset at any time. The simulation runs on the GPU and streams frames live, so every action takes effect on the next frame.

Full usage and reproduction steps are in [experiments/LIVE_CONSOLE.md](experiments/LIVE_CONSOLE.md).

[Back to contents](#en)

<a id="en-quick"></a>
## Quickstart

```bash
conda env create -f environment.yml && conda activate brain-fly   # full backends (Linux/WSL2)
# minimal Windows env: Python 3.10 venv + torch(cu126) + brian2 — see experiments/NOTES.md

python main.py --fastbrain --t_run 0.5 --n_run 1          # FastBrain benchmark
python experiments/validate_matrix.py --quick             # validation (smoke)
python experiments/validate_matrix.py --full              # validation (full matrix)
run_live.bat                                              # interactive live console
run_replay.bat                                            # generate & open 3D replay
```

[Back to contents](#en)

<a id="en-structure"></a>
## Project structure

```
fly-brain/
├── main.py                     # Benchmark entrypoint (CLI)
├── code/
│   ├── benchmark.py            # Orchestrator: config, logging, dispatch
│   ├── run_fastbrain.py        # FastBrain backend (new in this fork)
│   ├── run_pytorch.py          # PyTorch reference backend (surrogate gradients)
│   └── ...                     # Brian2 / Brian2CUDA / NEST GPU / GeNN / Brian2GeNN
├── experiments/
│   ├── fast_brain.py           # FastBrain engine
│   ├── validate_matrix.py      # Validation matrix
│   ├── live_server.py          # Live console server
│   ├── LIVE_CONSOLE.md         # Console usage guide
│   ├── NOTES.md                # Lab notebook (all performance/accuracy numbers)
│   └── RESEARCH.md             # Research landscape recon & roadmap
├── docs/
│   ├── index.html              # Project showcase site
│   └── assets/                 # Data figures, screenshots & illustrations
└── data/                       # FlyWire v783 data, benchmark results, spike parquets
```

[Back to contents](#en)

<a id="en-data"></a>
## Data

The model uses the public FlyWire connectome v783. Benchmark outputs live in `data/results/local_fastbrain/` as one-row-per-spike parquets with columns `time_ms`, `trial`, `neuron_index`, `flywire_id` and `exp_name`. The paper-suite bundle (`nature_2026_07`, 600 parquets) is on Google Drive; verification instructions are in the [upstream README](README_UPSTREAM.md).

[Back to contents](#en)

<a id="en-roadmap"></a>
## Research roadmap

- Phase 0 (done): systematic engine validation, see Key results above.
- Phase 1 (in progress): a dynamics-level benchmark of the spiking whole-brain model against public calcium-imaging datasets (DANDI [001869](https://dandiarchive.org/dandiset/001869), Aimon 2019 CRCNS), plus a sensitivity and null audit of published conclusions.
- Phase 2 (TBD): Bayesian-optimization inverse stimulus design, or mushroom-body learning embedded in the whole brain.

Every claim in this repo ships with degree-matched rewired null controls; that is a hard rule here. Full recon is in [experiments/RESEARCH.md](experiments/RESEARCH.md).

<div align="center">
<img src="docs/assets/ai_robot_embodied.png" alt="Concept: fly-brain-driven robot" width="70%">
</div>

[Back to contents](#en)

<a id="en-license"></a>
## License & credits

GPL-2.0-or-later (see [LICENSE](LICENSE)). Upstream © Eon Systems PBC; original Shiu et al. code (`code/paper-phil-drosophila/`) is MIT. Full installation details (NEST GPU source build, separate Brian2GeNN env, etc.) are in [README_UPSTREAM.md](README_UPSTREAM.md).

[Back to contents](#en)

<div align="center">
<img src="docs/assets/ai_viola_fly.png" alt="Viola smiling at a fruit fly on her fingertip" width="70%">
</div>
