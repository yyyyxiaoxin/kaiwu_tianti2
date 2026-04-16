# agent_diy 改进说明（基于 agent_ppo_y_1 第1步改进）

## 版本信息

| 项目 | 内容 |
|------|------|
| 版本 | 1.0 |
| 改进日期 | 2026-04-10 |
| 基于 | agent_ppo_y_1 第1步改进 |

---

## 第1步改进：扩展动作空间（支持闪现技能）✅ 已完成

### 改动原因

- 原版 diy 是空模板，没有实现
- 现在基于 agent_ppo_y_1 实现完整 PPO 算法
- 基线版本只使用移动动作(0-7)，忽略了闪现技能(8-15)
- 闪现是最有效的逃生手段，可大幅提升生存能力

### 具体改动

| 文件 | 改动 | 说明 |
|------|------|------|
| `conf/conf.py` | `ACTION_SHAPE: (8,) → (16,)` | 支持 8 个移动 + 8 个闪现 |
| `feature/definition.py` | `legal_actions/probs: 8 → 16` | 适配 16 维 |
| `feature/preprocessor.py` | `legal_act 改为 16 维` | 使用全部 legal_act_raw |
| `model/model.py` | `Actor 输出: 8 → 16` | 输出 16 维动作 logit |
| `algorithm/algorithm.py` | 完整 PPO 实现 | 适配 16 维 |
| `workflow/train_workflow.py` | 完整训练流程 | PPO 训练循环 |
| `agent.py` | 完整 Agent 实现 | PPO 推理+训练 |

### 改动标记

- 日志标识: `[diy_y1]` 便于区分
- 模型标识: `gorge_chase_lite_y1`

### 验证方法

1. 选择 **diy** 算法启动训练
2. 观察日志中 `[diy_y1]` 开头的输出
3. 检查 `flash_count` 是否增加（表示使用了闪现）
4. 对比改进前后生存步数

### 动作空间

| 动作 | 类型 | 说明 |
|------|------|------|
| 0-7 | 移动 | 8 方向移动 |
| 8-15 | 闪现 | 8 方向闪烁（距离 8-10 格）|

---

## 使用方法

1. 在腾讯开悟客户端选择 **diy** 算法（不是 ppo）
2. 训练会自动使用本改进版
3. 模型保存在 `agent_diy/ckpt/` 目录

---

## 目录结构

```
agent_diy/
├── IMPROVEMENTS.md      # 本文件
├── __init__.py
├── agent.py             # Agent 主类（完整实现）
├── algorithm/
│   ├── algorithm.py      # PPO 算法（16维）
│   └── __init__.py
├── conf/
│   ├── conf.py          # 配置（ACTION_SHAPE=16）
│   ├── train_env_conf.toml
│   ├── monitor_builder.py
│   └── __init__.py
├── feature/
│   ├── definition.py    # 数据定义（16维）
│   ├── preprocessor.py  # 特征处理（16维legal_act）
│   └── __init__.py
├── model/
│   ├── model.py         # 神经网络（Actor输出16）
│   └── __init__.py
└── workflow/
    ├── train_workflow.py # 训练流程
    └── __init__.py
```

---

## 第2步改进计划（待实施）

### 2.1 添加宝箱收集奖励

**预期效果**：智能体学会主动收集宝箱

**改动点**：
- `workflow/train_workflow.py`: 每收集一个宝箱 +1.0 奖励
- 跟踪 `treasures_collected` 增量

### 2.2 添加 Buff 收集奖励

**预期效果**：智能体学会拾取加速 buff

**改动点**：
- `workflow/train_workflow.py`: 每收集一个 buff +0.5 奖励
- 跟踪 `collected_buff` 增量

---

## 常见问题排查

### Q: 训练报错 dimension mismatch
A: 检查 `conf/conf.py` 中 `ACTION_SHAPE` 是否为 `(16,)`

### Q: 如何切换算法？
A: 在腾讯开悟客户端选择 **diy** 算法

### Q: 模型保存在哪里？
A: `agent_diy/ckpt/` 目录
