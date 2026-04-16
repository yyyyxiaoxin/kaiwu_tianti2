# 腾讯开悟比赛 - 模型改进规则与方法指南

> 本文档是对官方规则的归纳总结，用于指导 DIY 智能体的模型改进。

---

## 一、核心约束（不可违反）

### ⚠️ 关键限制

| 约束项 | 规则要求 | 违反后果 |
|--------|----------|----------|
| **地图视野维度** | 必须保持 **4×4=16D** | 违反规则 |
| **特征向量维度** | 基线 40D，当前实现 54D | 需保持一致 |
| **动作空间维度** | 16（8移动+8闪现） | 不可更改 |

### 📌 任务目标
- **核心目标**：在 1000 步内生存，同时躲避 2 个怪物，收集宝箱
- **计分规则**：步数得分 + 宝箱得分（每个宝箱 100 分）

---

## 二、特征工程规则

### ✅ 允许扩展的特征（加在基线40D基础上）

| 特征类型 | 建议维度 | 说明 |
|----------|----------|------|
| 宝箱信息 | 4-6D | 从 `organs` 获取位置、距离、是否可拾取 |
| 怪物朝向 | 2D | 怪物相对英雄的方向向量 |
| 历史距离 | 2D | t-1 步怪物距离（用于计算距离变化） |
| buff 状态 | 2D | 加速增益剩余时间、是否可用 |

### ❌ 禁止修改的特征
- **map_local（4×4=16D）**：必须保持原样，不可扩展视野
- **legal_action（8D）**：动作合法性掩码
- **hero_self（4D）**：英雄自身状态

### 📊 当前实现的特征结构（54D）

```
hero_self (4D)     - 位置、闪现冷却、加速buff
monster_1 (5D)     - 是否可见、位置、速度、距离
monster_2 (5D)     - 同上
treasure (6D)      - 视野内宝箱数量、方向、距离【增强】
map_local (16D)    - 4×4 地图通行性 ⚠️ 不可改
legal_action (16D) - 合法动作掩码（8移动+8闪现）
progress (2D)      - 步数、存活比例
```

**宝箱特征（6D）明细**：
| 维度 | 含义 | 归一化 |
|------|------|--------|
| 0 | 视野内是否有宝箱 | 0/1 |
| 1 | 视野内宝箱数量 | 0~1 |
| 2 | 最近宝箱距离 | 0~1 |
| 3 | 宝箱方向 x | -1~1 |
| 4 | 宝箱方向 z | -1~1 |
| 5 | 已收集宝箱数 | 0~1 |

---

## 三、奖励设计规则

### 📈 当前实现的奖励结构（v1.7）

| 奖励类型 | 当前权重 | 说明 |
|----------|----------|------|
| survive_reward | +0.01 | 基础生存奖励（降低） |
| dist_shaping | ±0.1 × Δdist | 怪物距离塑形（远离怪物） |
| **treasure_reward** | **+10.0** | **拾取宝箱奖励** |
| **treasure_shaping** | **+0.1~0.3** | **接近宝箱塑形（含固定奖励）** |
| **treasure_view_bonus** | **+0.01** | **视野内有宝箱时奖励** |
| **explore_bonus** | **+0.02~0.03** | **探索新区域奖励【新增】** |
| safe_bonus | +0.02 | 安全距离奖励 |
| danger_penalty | -0.02 | 危险距离惩罚 |

### ⚡ 奖励设计原则
1. **稀疏奖励问题**：基础奖励稀疏，需添加塑形奖励
2. **多目标平衡**：生存 > 收集 > 探索
3. **阶段调整**：500步后怪物加速，需加强躲避权重

---

## 四、模型结构改进规则

### ✅ 允许的改进方向

| 改进方向 | 实施建议 | 预期效果 |
|----------|----------|----------|
| 网络加深 | 54→128→64 → 54→256→128→64 | 提升表达能力 |
| **Actor/Critic 分离** | 共享骨干 + 独立头 | 稳定训练 |
| 注意力机制 | 添加 self-attention 层 | 聚焦关键信息 |
| 残差连接 | ResNet 风格 | 缓解梯度消失 |

### 📐 推荐架构（已实现）

```python
# 【第3步改进】加深后的网络结构
# 共享特征提取层: 54 → 256 → 128 (+ LayerNorm)
# Actor 专用层: 128 → 128 → 64 (+ LayerNorm)
# Critic 专用层: 128 → 128 → 64 → 32 (+ LayerNorm)
# Actor head: 64 → 16
# Critic head: 32 → 1
```

**架构对比**：

| 组件 | 原架构 | 新架构（加深后） |
|------|--------|------------------|
| shared_net | 54→128 | 54→256→128 |
| actor_private | 128→64 | 128→128→64 |
| critic_private | 128→64 | 128→128→64→32 |
| LayerNorm | 无 | 每层后添加 |
| 权重初始化 | 部分 | 全面正交初始化 |

---

## 五、训练稳定性规则

### 🎯 超参数推荐

| 参数 | 推荐范围 | 说明 |
|------|----------|------|
| 学习率 | 1e-4 ~ 5e-4 | 推荐 1e-4 |
| 学习率衰减 | 0.99 ~ 0.999 | 逐步降低 |
| 梯度裁剪 | 0.5 ~ 1.0 | 防止梯度爆炸 |
| 熵系数 | 0.01 ~ 0.1 | 鼓励探索 |
| GAE λ | 0.95 ~ 0.99 | 平衡偏差/方差 |
| PPO clip | 0.1 ~ 0.2 | 限制策略更新幅度 |

### 🔧 训练技巧
1. **学习率调度**：使用 cosine annealing 或 step decay
2. **早停机制**：监控价值损失，异常时降低学习率
3. **多地图训练**：避免过拟合单一地图

### 📈 当前实现的 LR 调度器（已完成）

采用 **warmup + 余弦衰减** 策略：

| 阶段 | 步数 | 学习率变化 |
|------|------|-----------|
| Warmup | 0 ~ 1000 | 5e-6 → 5e-5（线性增长） |
| 余弦衰减 | 1000 ~ 100000 | 5e-5 → 1e-5（余弦曲线） |

**配置参数**（conf/conf.py）：
```python
START_LR = 5e-5           # 目标学习率（降低）
MIN_LR = 1e-5             # 最小学习率
WARMUP_STEPS = 1000       # warmup 步数
WARMUP_START_LR = 1e-5    # warmup 起始学习率
TOTAL_TRAINING_STEPS = 100000  # 总训练步数
ENTROPY_LOSS_COEFF = 0.01 # 熵系数（提高探索）
```

**实现方式**（agent/agent.py）：
```python
# warmup 调度器：线性提升
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=WARMUP_START_LR / START_LR,
    end_factor=1.0,
    total_iters=WARMUP_STEPS,
)

# 余弦衰减调度器
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=TOTAL_TRAINING_STEPS - WARMUP_STEPS,
    eta_min=MIN_LR,
)

# 组合调度器
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_STEPS],
)
```

---

## 六、allowed 优化方向总结

### 📋 优先级排序

```
1. [高] 奖励函数优化      → ✅ 已完成
2. [高] Actor/Critic 分离 → ✅ 已完成
3. [中] 特征工程扩展      → ✅ 已完成
4. [中] 学习率调度        → ✅ 已完成（warmup + cosine）
5. [中] 网络结构加深      → ✅ 已完成（256+128+64+32）
6. [低] 注意力机制        → 待定
```

### ❌ 禁止的优化
- 扩大地图视野（4×4 不可改）
- 修改动作空间维度
- 重写日志系统
- 破坏分布式架构

---

## 七、代码目录规范

```
agent_diy/
├── algorithm/    # PPO 算法实现
├── conf/         # 配置文件（超参数、维度）
├── feature/      # 特征处理（preprocessor.py）
├── model/        # 神经网络模型
├── workflow/     # 训练流程
└── agent.py      # 智能体核心接口
```

---

## 八、关键接口规范

### feature/preprocessor.py
```python
def feature_process(self, env_obs, last_action):
    # 输入: env_obs (环境观测), last_action (上一步动作)
    # 输出: feature (54D), legal_action (16D), reward (float)
```

### model/model.py
```python
class Model(nn.Module):
    def __init__(self, state_shape=(54,), action_shape=16):
        # 输入: state_shape (特征维度)
        # 输出: action_logits, value
```

### algorithm/ppo.py
```python
def learn(self, list_sample_data):
    # 输入: 采样数据列表
    # 输出: loss, metrics
```

---

## 九、常见错误规避

| 错误类型 | 原因 | 解决方案 |
|----------|------|----------|
| 维度不匹配 | 特征维度变化 | 保持 54D，更新 conf.py |
| shape 错误 | numpy 数组未 flatten | 使用 `.flatten()` 确保 1D |
| 训练崩溃 | 学习率过高 | 降至 1e-4 |
| reward_sum 缺失 | SampleData 未初始化 | 添加 `reward_sum=np.zeros(1)` |

---

## 十、改进检查清单

进行任何改进前，请确认：

- [ ] 特征向量维度仍为 54D
- [ ] map_local 仍为 4×4=16D
- [ ] 动作空间仍为 16 维
- [ ] 配置文件已同步更新
- [ ] 单元测试通过（若有）

---

*文档版本：v1.8*
*最后更新：2026-04-11*

---

## 十一、改进日志

### v1.8 (2026-04-11)
- **【BUG修复】exploit 方法输入格式不兼容**：
  - `agent.py`：`exploit` 方法只支持列表输入，但评估流程传入字典
  - 错误：`KeyError: 0` - `list_obs_data[0]` 在字典上失败
  - 修复：同时支持字典输入（评估流程）和列表输入（训练流程）

### v1.7 (2026-04-10)
- **【P0调试】添加 organs 数据验证日志**：
  - preprocessor.py：每100步打印一次 `organs_count`, `total_treasure`, `in_view`, `collected`
  - 用于排查宝箱数据是否正确获取

- **【P1奖励】增加固定接近奖励**：
  - preprocessor.py：treasure_shaping 增加固定值
    - 全局宝箱存在时：`+0.1 + 0.2 × Δdist`
    - 视野内有宝箱时：`+0.15 + 0.3 × Δdist`
  - 解决"接近但距离不变时无奖励"问题

- **【P1奖励】降低生存奖励**：
  - preprocessor.py：survive_reward 从 0.02 降至 0.01
  - 平衡生存与收集的目标优先级

- **【P3奖励】添加探索奖励**：
  - preprocessor.py：新增 explore_bonus
    - 访问新格子：+0.02
    - 移动超过2格：+0.01
  - 鼓励智能体主动探索地图寻找宝箱

- **【P2超参】降低学习率**：
  - conf.py：START_LR 从 1e-4 降至 5e-5
  - 提高训练稳定性

- **【P2超参】提高熵系数**：
  - conf.py：ENTROPY_LOSS_COEFF 从 0.001 增至 0.01
  - 增加探索度，避免过早收敛

### v1.6 (2026-04-10)
- **【BUG修复】特征维度配置错误**：
  - `conf/conf.py`：`FEATURE_VECTOR_SHAPE` 从 48D 修正为 54D
  - 正确维度构成：4 (hero) + 5 (m1) + 5 (m2) + 6 (treasure) + 16 (map) + 16 (legal) + 2 (progress) = 54D
  - 修复错误：`RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x54 and 48x128)`

### v1.5 (2026-04-10)
- **【规则允许】添加全局宝箱信息支持（从 organs 获取）**：
  - preprocessor.py：新增 organs 字段解析
    - 从 `frame_state["organs"]` 获取**视野外**宝箱位置
    - 计算全局最近宝箱距离和方向向量
    - 保持 4×4 地图视野约束不变（仅用于决策参考）
  - **规则依据**：`rules.md` 中"十个选手优化方向"明确允许"宝箱特征：从 organs 获取位置/距离"
  - **奖励更新**：使用全局距离计算接近奖励塑形，提高导航效率

### v1.4 (2026-04-10)
- **【核心改进】添加独立宝箱特征（6D）**：
  - preprocessor.py：新增 treasure_feat(6D)
    - 视野内宝箱数量
    - 最近宝箱方向向量
    - 最近宝箱距离
    - 已收集宝箱数

- **【核心改进】大幅提高宝箱奖励**：
  - preprocessor.py：
    - treasure_reward: 2.0 → 10.0（5倍提升）
    - treasure_shaping: 0.05 → 0.3（6倍提升）
    - 新增 treasure_view_bonus: +0.01（视野内有宝箱时）

### v1.3 (2026-04-10)
- **奖励函数优化**：基于监控面板分析，调整奖励结构
  - survive_reward: 0.01 → 0.05
  - dist_shaping: 0.3 → 0.1
  - 新增 safe_bonus、danger_penalty

### v1.2 (2026-04-10)
- **网络结构简化**：256→128→64 改为 128→64

### v1.1 (2026-04-10)
- **学习率调度器优化**：warmup + 余弦衰减

### v1.0 (2026-04-10)
- 初始版本，定义核心约束和优化方向
