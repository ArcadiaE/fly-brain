# 研究方向情报汇总（2026-10-11 五路文献侦查）

用途：选定主攻方向前的占地判定。所有结论以"是否存在更近的相反证据"为准，引用见各节。

## 占领度地图

| 方向 | 判定 | 关键事实 |
|---|---|---|
| 实时交互全脑仿真 | 仅剩窄缝 | Sandia Loihi 2（arXiv 2508.16792）已占"实时"（356× Brian2）；FastFly（github.com/eonfathom/FastFly）架构与 FastBrain 雷同但未公布验证/速度；fruit-fly-lab 有 Brian2 逐脉冲验证但 CPU 慢 10×；浏览器游戏生态 2026 年爆发（awesome-fly: github.com/cobanov/awesome-fly）。可宣称上限：首个**验证过的**消费级 GPU 实时+交互全脑仿真器 → 工具论文（Frontiers Neuroinform / JOSS 级） |
| 冻结连接组+薄学习接口做具身控制 | ~70% 被围 | FlyDoom（github.com/eganeganegan/flydoom）几乎同方案（业余、无结果、子图、rate-based）；FlyGM（arXiv 2602.17997，清华）占"连接组=归纳偏置"头条（端到端训练版）；arXiv 2606.17745 预示粗指标上与重连 null 打平。剩余空隙：冻结+脉冲+全脑+学习编码/读出+控制+严格 null 层级+样本效率曲线，无人发表。窗口以月计 |
| BO/梯度 刺激逆向设计 | 组合空着 | 组件有主：OMiSO（arXiv 2507.07858）、Triplett NeurIPS 2023（PMC10246014）、DBS-BO 系列、Bashivan Science 2019。无人把优化器接到全脑连接组模型上。差异点：split-GAL4 驱动线使"刺激方案"可直接交付湿实验。风险：Pospisil 2024 effectome 邻近、BO 维度可信度、模型保真度上限 |
| 蘑菇体学习嵌入全脑 | 真空 | 连接组约束 MB 学习全是孤立的（Eschbach 2020 Nat Neurosci 幼虫；Bennett 2021 Nat Commun 复现 blocking；Nature 2024 模块模型复现消退）；全脑 v783 仿真全部无可塑性。最接近：20fly8（真实 v783 MB 布线+多巴胺规则玩 2048，孤立、未发表）。数据基础已验证：5177 KC / 96 MBON / 331 DAN（见 _audit_mb.py） |
| 现有模型缺陷修复 | 富矿 | 见下节 |

## Shiu 2024 模型的公认缺陷（修复/对比类工作的攻击点）

1. **动力学级验证缺失（领域最大空洞）**：91% 准确率是扰动级（激活 X → 行为 Y），从未与记录到的神经活动对比。Li et al. 2026（bioRxiv 10.64898/2026.08.21.745055）只做了 rate 模型 × 静息态 × neuropil 级（held-out FC r=0.423，天花板 0.473）。**脉冲 × 单细胞/细胞类型 × 诱发活动无人做**。
2. **权重=突触数、符号=递质预测**：谷氨酸正负性在果蝇有争议；±50% I:E 比例扫描只做过一次。重跑 164 条预测套件做敏感性审计 = 便宜、可预注册。
3. **零基线放电**：作者点名的失败原因（抑制性细胞 Tentacular/Phantom 预测失败）。加每细胞类型基线驱动 + 对比静息态统计（雪崩指数等）= 具体可证伪。
4. 无 gap junction（数据不存在，别碰）、无可塑性/神经调制（全脑尺度无数据约束，单独做会不可证伪）。
5. 具身闭环已被 Eon（2026.03，NeuroMechFly v2）和 FlyGM 占据；Digital Sphinx（Brunton et al. 2026，bioRxiv 10.64898/2026.03.20.713233）证明"行为像苍蝇"无信息量（线虫脑也能驱动蝇身）。

## 贯穿性方法论要求（2026 年后的入场券）

- 一切结论必须内置 degree-matched 重连 null 对照（Digital Sphinx / Zhao et al. arXiv 2609.38665 之后这是审稿标准）。
- 引用任何"准确率"数字前追溯到原始分母（Eon 的 ~95% 是 Shiu 扰动级统计的回收）。
- 避免"upload/emulation"措辞（LessWrong/Carboncopies 批评后此词在果蝇神经科学社区引发正当敌意），用 connectome-constrained modeling + 明确验证目标。

## 修正后路线图

- **Phase 0**：`validate_matrix.py` 引擎验证矩阵（进行中）。
- **Phase 1（主力，5080 到位后开跑）**：脉冲全脑模型 vs 公开钙成像数据的首次动力学级基准 + Shiu 结论敏感性/null 审计。
- **Phase 2（Phase 1 后二选一）**：BO 刺激逆向设计（对口光遗传实验设计社区）或 MB 学习（先孤立 MB 复现条件反射 → 再嵌全脑做背景消融）。
- **放弃**：冻结连接组 RL 控制（业余 swarm + 预期 null）、纯速度叙事。

## MB 学习路线的硬前提（若选）

1. APL 是 graded 非脉冲神经元，塞进 LIF 需专门处理以维持 KC ~5–10% 稀疏激活（clawRxiv 2604.01561 曾手动调参到 7.8%）。
2. 模型无多巴胺机制，三因子规则需外加；注意生物学主导机制是 KC 同步诱发的 depression（DopR1）。
3. 行为读出只能先靠 MBON 效价解析解码器；MBON→下行神经元通路动力学未验证。
4. 顺序：孤立 MB 复现条件反射（20fly8 证明数据可行）→ 全脑嵌入做背景消融对比。
