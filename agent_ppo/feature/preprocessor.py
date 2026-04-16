#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。

特征维度（共53D）：
- hero_self: 4D
- monster_1: 7D (5D基础 + 方向2D)
- monster_2: 7D (5D基础 + 方向2D)
- wall_dist_8dir: 8D (8方向到墙距离)
- treasure: 5D (方向2D + 距离1D + 是否极近1D + 收集数1D)
- flash_available: 1D (闪现技能是否可用)
- legal_action: 16D (8移动 + 8闪现)
- exploration: 3D (已探索比例 + 当前格访问次数 + 停滞步数)
- progress: 2D

修改要点：
1. 英雄模型为1×1（非3×3），移除错误的3×3碰撞检测
2. 增加宝箱距离特征，解决斜向走绕宝箱问题
3. 增加绕宝箱惩罚和接近宝箱奖励
4. 动作空间扩展为16维（8移动+8闪现），支持闪现技能
5. 怪物近距离时，朝怪物方向闪现给予高额奖励（穿越怪物逃生）
"""

import numpy as np

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
# 宝箱"极近"距离阈值（格数，1×1英雄时宝箱需要踩上去）
TREASURE_VERY_CLOSE_DIST = 2.0

# 闪现触发距离阈值：怪物在2-3格内时鼓励朝怪物闪现
FLASH_MONSTER_DIST_MIN = 0.5   # 太近时闪现可能来不及，不鼓励
FLASH_MONSTER_DIST_MAX = 3.5   # 3.5格以内开始鼓励闪现

# 动作方向映射（8移动+8闪现）
# 移动0-7: 右、下、左、上、右下、左下、左上、右上
# 闪现8-15: 对应移动0-7的闪现版本
# 方向向量 (dx, dz)，dx=col方向，dz=row方向
ACTION_DIRECTIONS = {
    0: (1, 0),    # 右
    1: (0, 1),    # 下
    2: (-1, 0),   # 左
    3: (0, -1),   # 上
    4: (1, 1),    # 右下
    5: (-1, 1),   # 左下
    6: (-1, -1),  # 左上
    7: (1, -1),   # 右上
}
# 闪现方向与移动方向一一对应
FLASH_DIRECTIONS = {
    8: (1, 0),    # 闪现右
    9: (0, 1),    # 闪现下
    10: (-1, 0),  # 闪现左
    11: (0, -1),  # 闪现上
    12: (1, 1),   # 闪现右下
    13: (-1, 1),  # 闪现左下
    14: (-1, -1), # 闪现左上
    15: (1, -1),  # 闪现右上
}


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1]."""
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _get_direction_to_target(from_pos, to_pos):
    """计算从起点到目标的方向向量（归一化）。"""
    dx = to_pos["x"] - from_pos["x"]
    dz = to_pos["z"] - from_pos["z"]
    dist = np.sqrt(dx ** 2 + dz ** 2)
    if dist < 0.01:
        return (0.0, 0.0)
    return (dx / dist, dz / dist)


def _count_wall_in_direction(map_info, center_row, center_col, direction, max_dist=5):
    """计算某个方向上到墙壁的距离。"""
    dr, dc = direction
    for step in range(1, max_dist + 1):
        row = center_row + dr * step
        col = center_col + dc * step
        if not (0 <= row < len(map_info) and 0 <= col < len(map_info[0])):
            return float(step - 1)
        if map_info[row][col] != 0:
            return float(step - 1)
    return float(max_dist)


def _is_path_blocked_to_treasure(map_info, hero_pos, treasure_pos):
    """检测从英雄到宝箱的直线路径上是否有墙阻挡。

    使用 Bresenham 风格的射线检测，沿英雄→宝箱方向逐步检查是否遇到墙。
    返回 True 表示路径被阻挡。
    """
    if map_info is None or len(map_info) < 3:
        return False

    map_h = len(map_info)
    map_w = len(map_info[0]) if map_h > 0 else 0
    center = map_h // 2

    # 英雄在 map_info 中的行列
    hero_row = center
    hero_col = center

    # 宝箱在 map_info 中的行列（坐标转换：x→col, z→row，需要用相对偏移）
    # map_info 是以英雄为中心的局部地图，每格对应1个单位
    dx = treasure_pos["x"] - hero_pos["x"]
    dz = treasure_pos["z"] - hero_pos["z"]
    treasure_col = center + int(round(dx))
    treasure_row = center + int(round(dz))

    # 限制检测范围在 map_info 内
    max_row = map_h - 1
    max_col = map_w - 1

    # Bresenham 射线：从英雄到宝箱方向逐步检测
    dr = treasure_row - hero_row
    dc = treasure_col - hero_col
    dist = max(abs(dr), abs(dc))
    if dist == 0:
        return False

    # 只检测到 map_info 边界内的部分
    for step in range(1, min(dist, center) + 1):
        # 使用浮点插值取最近整数格
        row = hero_row + int(round(dr * step / dist))
        col = hero_col + int(round(dc * step / dist))

        # 超出地图范围视为阻挡
        if row < 0 or row > max_row or col < 0 or col > max_col:
            return True

        # 遇到墙（非0值）
        if map_info[row][col] != 0:
            return True

    return False


def _is_flash_direction_blocked(map_info, action_id, check_dist=2):
    """检测闪现方向前方是否被墙阻挡（闪现2格，需检查路径上所有格子）。

    Args:
        map_info: 局部地图信息
        action_id: 闪现动作ID (8-15)
        check_dist: 检查距离（闪现2格）

    Returns:
        True 表示闪现方向被阻挡，不建议闪现
    """
    if map_info is None or action_id not in FLASH_DIRECTIONS:
        return False

    if len(map_info) < 5:
        return False

    center = len(map_info) // 2
    dx, dz = FLASH_DIRECTIONS[action_id]
    is_diagonal = (dx != 0 and dz != 0)

    # 检查闪现路径上的所有格子
    for step in range(1, check_dist + 1):
        # 对角线移动需要检查经过的格子
        if is_diagonal:
            # 斜向闪现：检查直角边经过的格子（避免穿墙角）
            for sub_step in range(1, step + 1):
                # 水平方向
                r_h = center + dz * sub_step // step if step > 0 else center
                c_h = center + dx * step // max(sub_step, 1)
                # 垂直方向
                r_v = center + dz * step // max(sub_step, 1)
                c_v = center + dx * sub_step // step if step > 0 else center
                for r, c in [(center + dz * step, center + dx * sub_step),
                              (center + dz * sub_step, center + dx * step)]:
                    if 0 <= r < len(map_info) and 0 <= c < len(map_info[0]):
                        if map_info[r][c] != 0:
                            return True
        # 直线方向格子
        row = center + dz * step
        col = center + dx * step
        if not (0 <= row < len(map_info) and 0 <= col < len(map_info[0])):
            return True  # 超出地图=阻挡
        if map_info[row][col] != 0:
            return True  # 遇到墙

    return False


def _get_action_direction_match(action_id, target_dir_x, target_dir_z):
    """计算动作方向与目标方向的一致性（点积）。

    返回值范围 [-1, 1]，1=完全对准，-1=反方向。
    用于判断闪现动作是否朝向怪物方向。
    """
    if action_id in ACTION_DIRECTIONS:
        dx, dz = ACTION_DIRECTIONS[action_id]
    elif action_id in FLASH_DIRECTIONS:
        dx, dz = FLASH_DIRECTIONS[action_id]
    else:
        return 0.0

    # 归一化方向向量
    norm = np.sqrt(dx ** 2 + dz ** 2)
    if norm < 1e-6:
        return 0.0
    act_dir_x = dx / norm
    act_dir_z = dz / norm

    # 点积
    return act_dir_x * target_dir_x + act_dir_z * target_dir_z


class Preprocessor:
    def __init__(self):
        self.reset()

    # 探索网格分辨率：将128×128地图划分为 GRID_RES×GRID_RES 的网格
    GRID_RES = 8  # 8×8=64个网格单元，每格16×16游戏单位
    GRID_CELL_SIZE = MAP_SIZE / 8  # 每格16单位

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.last_hero_pos = None
        self.consecutive_wall_hits = 0
        self.last_action = -1
        self.last_treasure_count = 0
        self.last_treasure_dist = None      # 上一步最近宝箱距离
        self.last_treasure_pos = None        # 上一步最近宝箱位置
        self.near_treasure_orbit_steps = 0   # 在宝箱附近绕圈的步数
        self.treasure_blocked_steps = 0      # 宝箱路径被墙阻挡的连续步数
        self.last_monster_dist_when_approaching = None  # 上次靠近宝箱时的怪物距离

        # ===== 探索追踪 =====
        self.visit_grid = np.zeros((self.GRID_RES, self.GRID_RES), dtype=np.int32)  # 访问计数
        self.total_visited_cells = 0          # 已访问的不同网格数
        self.idle_steps = 0                   # 连续停滞步数（在同一网格内不动）
        self.current_grid_cell = None         # 当前所在网格
        self.last_grid_cell = None            # 上一步所在网格

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        特征维度（共41D）。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        # =========================================================================
        # 1. Hero self features (4D) / 英雄自身特征
        # =========================================================================
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero["flash_cooldown"], MAX_FLASH_CD)
        buff_remain_norm = _norm(hero["buff_remaining_time"], MAX_BUFF_DURATION)
        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # =========================================================================
        # 2. Monster features + 方向 (7D x 2)
        # =========================================================================
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]
                if is_in_view:
                    m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                    m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                    dir_x, dir_z = _get_direction_to_target(hero_pos, m_pos)
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                    dir_x, dir_z = 0.0, 0.0

                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm, dir_x, dir_z], dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(7, dtype=np.float32))

        # =========================================================================
        # 3. 8方向到墙距离 (8D) / 帮助判断该往哪里走
        # =========================================================================
        wall_dist_8dir = np.zeros(8, dtype=np.float32)
        if map_info is not None and len(map_info) >= 21:
            center = len(map_info) // 2
            directions = [
                (0, 1),   # 0: 右
                (1, 0),   # 1: 下
                (0, -1),  # 2: 左
                (-1, 0),  # 3: 上
                (1, 1),   # 4: 右下
                (1, -1),  # 5: 左下
                (-1, -1), # 6: 左上
                (-1, 1),  # 7: 右上
            ]
            for i, (dr, dc) in enumerate(directions):
                dist = _count_wall_in_direction(map_info, center, center, (dr, dc), max_dist=10)
                wall_dist_8dir[i] = dist / 10.0

        # =========================================================================
        # 4. 宝箱特征 (5D) / 方向2D + 距离1D + 是否极近1D + 收集数1D
        #    关键改进：增加距离信息，让模型感知"绕圈"问题
        # =========================================================================
        treasure_dir_x = 0.0
        treasure_dir_z = 0.0
        treasure_dist_norm = 1.0      # 默认：没有宝箱信息
        treasure_very_close = 0.0     # 是否非常接近宝箱
        treasure_collected_norm = 0.0

        nearest_treasure_dist = float('inf')
        nearest_treasure_pos = None

        # 从 organs 获取宝箱位置（视野内+视野外）
        organs = frame_state.get("organs", [])
        if organs and len(organs) > 0:
            for treasure in organs:
                t_pos = treasure.get("pos", {})
                if t_pos:
                    t_x = t_pos.get("x", hero_pos["x"])
                    t_z = t_pos.get("z", hero_pos["z"])
                    raw_dist = np.sqrt((hero_pos["x"] - t_x) ** 2 + (hero_pos["z"] - t_z) ** 2)
                    if raw_dist < nearest_treasure_dist:
                        nearest_treasure_dist = raw_dist
                        nearest_treasure_pos = {"x": t_x, "z": t_z}

        # 如果 organs 没有数据，尝试从 env_info 获取
        if nearest_treasure_pos is None:
            treasure_info = env_info.get("treasures", [])
            for t in treasure_info:
                t_pos = t.get("pos", {})
                if t_pos:
                    raw_dist = np.sqrt((hero_pos["x"] - t_pos.get("x", 0)) ** 2 + (hero_pos["z"] - t_pos.get("z", 0)) ** 2)
                    if raw_dist < nearest_treasure_dist:
                        nearest_treasure_dist = raw_dist
                        nearest_treasure_pos = t_pos

        if nearest_treasure_pos is not None:
            dir_x, dir_z = _get_direction_to_target(hero_pos, nearest_treasure_pos)
            treasure_dir_x = dir_x
            treasure_dir_z = dir_z
            treasure_dist_norm = _norm(nearest_treasure_dist, MAP_SIZE * 1.41)
            # 英雄1×1，宝箱需要踩上去，所以"极近"阈值设小
            if nearest_treasure_dist < TREASURE_VERY_CLOSE_DIST:
                treasure_very_close = 1.0

        treasures_collected = env_info.get("treasures_collected", 0)
        treasure_collected_norm = _norm(treasures_collected, 10.0)

        treasure_feat = np.array([
            treasure_dir_x,         # 宝箱方向 x
            treasure_dir_z,         # 宝箱方向 z
            treasure_dist_norm,     # 宝箱距离（归一化）
            treasure_very_close,    # 是否极近宝箱
            treasure_collected_norm,# 已收集宝箱数
        ], dtype=np.float32)

        # =========================================================================
        # 5. Legal action mask (16D) / 合法动作掩码 (8移动 + 8闪现)
        # =========================================================================
        legal_action = [0] * 16  # 初始化为全0

        # 闪现是否可用（CD为0时可用）
        flash_available = 1.0 if hero.get("flash_cooldown", MAX_FLASH_CD) <= 0 else 0.0

        # --- 移动动作 (0-7) ---
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(8, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 16}
                for j in range(8):
                    legal_action[j] = 1 if j in valid_set else 0
        else:
            legal_action[:8] = [1] * 8

        # --- 闪现动作 (8-15) ---
        # 闪现需要满足：1) 闪现CD为0  2) 环境允许该动作  3) 闪现方向不被墙阻挡
        if flash_available > 0.5:
            for j in range(8, 16):
                # 检查环境是否允许该闪现动作
                env_legal = False
                if isinstance(legal_act_raw, list) and j < len(legal_act_raw):
                    if isinstance(legal_act_raw[0], bool):
                        env_legal = bool(legal_act_raw[j])
                    else:
                        env_legal = j in {int(a) for a in legal_act_raw}

                if not env_legal:
                    # 环境不允许，跳过
                    legal_action[j] = 0
                    continue

                # 检查闪现方向是否被墙完全阻挡
                if _is_flash_direction_blocked(map_info, j):
                    legal_action[j] = 0  # 闪现会被墙挡住，不推荐
                else:
                    legal_action[j] = 1

        # 安全检查：如果所有动作都不可用，至少保证有一个安全移动方向
        if sum(legal_action) == 0:
            legal_action[:8] = [1] * 8

        # =========================================================================
        # 6. 探索特征 (3D) / 已探索区域比例 + 当前格访问次数 + 停滞步数
        # =========================================================================
        # 将英雄位置映射到网格
        grid_x = int(np.clip(hero_pos["x"] / self.GRID_CELL_SIZE, 0, self.GRID_RES - 1))
        grid_z = int(np.clip(hero_pos["z"] / self.GRID_CELL_SIZE, 0, self.GRID_RES - 1))
        self.current_grid_cell = (grid_x, grid_z)

        # 更新访问计数
        if self.visit_grid[grid_z, grid_x] == 0:
            self.total_visited_cells += 1
        self.visit_grid[grid_z, grid_x] += 1

        # 计算停滞步数：在同一网格内未移动
        if self.current_grid_cell == self.last_grid_cell:
            self.idle_steps += 1
        else:
            self.idle_steps = 0

        # 已探索区域比例
        max_visitable_cells = self.GRID_RES * self.GRID_RES  # 64
        visited_ratio = min(1.0, self.total_visited_cells / max_visitable_cells)

        # 当前格访问次数（归一化，高频=重复访问=停滞）
        current_cell_visits_norm = min(1.0, self.visit_grid[grid_z, grid_x] / 50.0)

        # 停滞步数归一化
        idle_steps_norm = min(1.0, self.idle_steps / 30.0)

        exploration_feat = np.array([
            visited_ratio,              # 已探索比例
            current_cell_visits_norm,    # 当前格重复访问度
            idle_steps_norm,             # 停滞程度
        ], dtype=np.float32)

        # 保存当前网格
        self.last_grid_cell = self.current_grid_cell

        # =========================================================================
        # 7. Progress features (2D)
        # =========================================================================
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # =========================================================================
        # 拼接特征向量 (53D)
        # =========================================================================
        feature = np.concatenate([
            hero_feat,                    # 4D
            monster_feats[0],             # 7D
            monster_feats[1],             # 7D
            wall_dist_8dir,              # 8D
            treasure_feat,               # 5D
            np.array([flash_available], dtype=np.float32),  # 1D
            np.array(legal_action, dtype=np.float32),  # 16D
            exploration_feat,            # 3D
            progress_feat,               # 2D
        ])

        # =========================================================================
        # 7. 奖励计算
        # =========================================================================
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[4])

        # 7.1 存活奖励
        survive_reward = 0.01

        # 7.2 怪物距离塑形（远离怪物加分，靠近不扣分）
        delta = cur_min_dist_norm - self.last_min_monster_dist_norm
        dist_shaping = 0.1 * delta if delta > 0 else 0
        self.last_min_monster_dist_norm = cur_min_dist_norm

        # 7.3 撞墙检测（仅记录，不惩罚，避免影响存活步数）
        is_wall_hit = False
        wall_penalty = 0.0
        if self.last_hero_pos is not None:
            move_x = hero_pos["x"] - self.last_hero_pos[0]
            move_z = hero_pos["z"] - self.last_hero_pos[1]
            move_dist = np.sqrt(move_x ** 2 + move_z ** 2)

            if move_dist < 0.5 and last_action >= 0:
                is_wall_hit = True
                self.consecutive_wall_hits += 1
            else:
                self.consecutive_wall_hits = 0

        # 保存当前位置
        self.last_hero_pos = (hero_pos["x"], hero_pos["z"])

        # 7.4 卡住不动惩罚（仅记录，不惩罚）
        stuck_penalty = 0.0

        # 7.5 宝箱收集奖励（大幅提高）
        treasure_reward = 0.0
        if treasures_collected > self.last_treasure_count:
            new_treasures = treasures_collected - self.last_treasure_count
            treasure_reward = new_treasures * 15.0  # 10→15，强化宝箱收集动机
        self.last_treasure_count = treasures_collected

        # 7.6 宝箱接近奖励塑形（核心改进：解决绕宝箱走+被墙吸附+怪物威胁问题）
        treasure_shaping = 0.0
        orbiting_penalty = 0.0  # 绕宝箱惩罚
        wall_magnet_penalty = 0.0  # 被墙吸附惩罚

        # ===== 路径阻挡检测 =====
        is_path_blocked = False
        if (nearest_treasure_pos is not None
                and np.isfinite(nearest_treasure_dist)
                and nearest_treasure_dist < 15.0
                and map_info is not None):
            is_path_blocked = _is_path_blocked_to_treasure(map_info, hero_pos, nearest_treasure_pos)

        # 更新被阻挡连续步数
        if is_path_blocked:
            self.treasure_blocked_steps += 1
        else:
            self.treasure_blocked_steps = max(0, self.treasure_blocked_steps - 2)  # 快速恢复

        # 被阻挡时的奖励衰减系数（阻挡越久衰减越大，最低0.1）
        if self.treasure_blocked_steps > 0:
            blocked_decay = max(0.1, 1.0 - 0.15 * self.treasure_blocked_steps)
        else:
            blocked_decay = 1.0

        # ===== 怪物威胁检测 =====
        # 1. 怪物在宝箱旁边 → 宝箱是危险的
        # 2. 靠近宝箱时离怪物越来越近 → 宝箱吸引力降低
        monster_threat_decay = 1.0  # 怪物威胁衰减系数，1.0=无威胁

        if nearest_treasure_pos is not None and np.isfinite(nearest_treasure_dist):
            # 计算每个可见怪物到宝箱的距离
            monster_near_treasure = False
            closest_monster_dist_to_hero = float('inf')

            for m in monsters:
                if m.get("is_in_view", 0):
                    m_pos = m["pos"]
                    # 怪物到宝箱的距离
                    monster_to_treasure = np.sqrt(
                        (m_pos["x"] - nearest_treasure_pos["x"]) ** 2 +
                        (m_pos["z"] - nearest_treasure_pos["z"]) ** 2
                    )
                    # 怪物到英雄的距离
                    monster_to_hero = np.sqrt(
                        (m_pos["x"] - hero_pos["x"]) ** 2 +
                        (m_pos["z"] - hero_pos["z"]) ** 2
                    )
                    closest_monster_dist_to_hero = min(closest_monster_dist_to_hero, monster_to_hero)

                    # 怪物在宝箱旁边（距离<6格视为"守着宝箱"）
                    if monster_to_treasure < 6.0:
                        monster_near_treasure = True

            # 情况1：怪物就在宝箱旁边 → 大幅衰减宝箱奖励
            if monster_near_treasure:
                monster_threat_decay = 0.2  # 宝箱旁边有怪，奖励降到20%

            # 情况2：靠近宝箱的同时离怪物越来越近 → 逐步衰减
            # 判断方式：比较当前怪物距离和上次靠近宝箱时的怪物距离
            if (np.isfinite(closest_monster_dist_to_hero)
                    and nearest_treasure_dist < 10.0):  # 只在宝箱较近时检测
                if self.last_monster_dist_when_approaching is not None:
                    # 怪物距离变化：正值=怪物在靠近我们，负值=怪物在远离
                    monster_delta = self.last_monster_dist_when_approaching - closest_monster_dist_to_hero
                    if monster_delta > 0:
                        # 靠近宝箱的同时怪物也在靠近 → 额外衰减
                        # 怪物越近衰减越大
                        approach_danger = min(0.7, 0.1 * monster_delta)
                        monster_threat_decay = min(monster_threat_decay, 1.0 - approach_danger)
                        # 怪物非常近（<5格）时，不管是否在宝箱旁都大幅衰减
                        if closest_monster_dist_to_hero < 5.0:
                            monster_threat_decay = min(monster_threat_decay, 0.15)

                # 更新记录
                self.last_monster_dist_when_approaching = closest_monster_dist_to_hero
            else:
                # 宝箱较远时，逐步恢复怪物距离记录
                self.last_monster_dist_when_approaching = None

        # 综合衰减 = 墙阻挡衰减 × 怪物威胁衰减
        total_decay = blocked_decay * monster_threat_decay

        # 数值安全：只在上一步和当前步都有有限距离时计算塑形
        has_valid_dist = (nearest_treasure_pos is not None
                          and self.last_treasure_dist is not None
                          and np.isfinite(nearest_treasure_dist)
                          and np.isfinite(self.last_treasure_dist))

        if has_valid_dist:
            # 接近宝箱给奖励（被阻挡/怪物威胁时大幅衰减）
            dist_delta = self.last_treasure_dist - nearest_treasure_dist
            if dist_delta > 0:
                # 正在靠近宝箱：基础奖励 + 距离越近奖励越大
                proximity_bonus = 1.0 if nearest_treasure_dist < 5.0 else 0.3
                treasure_shaping = (0.3 * dist_delta + proximity_bonus * dist_delta) * total_decay
            else:
                # 正在远离宝箱：轻微惩罚（衰减时降低惩罚，避免被墙/怪吸住后远离时反而不惩罚）
                treasure_shaping = 0.1 * dist_delta * total_decay

            # ===== 绕宝箱惩罚（核心修改）=====
            # 检测"在宝箱附近但距离不减少"的模式
            if nearest_treasure_dist < 5.0:
                # 检查是否在绕圈：距离几乎不变
                dist_change = abs(self.last_treasure_dist - nearest_treasure_dist)
                if dist_change < 0.5:  # 距离几乎没变
                    self.near_treasure_orbit_steps += 1
                else:
                    self.near_treasure_orbit_steps = max(0, self.near_treasure_orbit_steps - 1)

                # 绕圈超过3步就惩罚
                if self.near_treasure_orbit_steps > 3:
                    orbiting_penalty = -1.0  # 绕圈惩罚
                    # 距离越近绕圈惩罚越大
                    if nearest_treasure_dist < 3.0:
                        orbiting_penalty = -2.0

                    # ===== 被墙吸附额外惩罚 =====
                    # 如果绕圈同时路径被阻挡，说明是卡在墙上了，加大惩罚
                    if is_path_blocked and self.treasure_blocked_steps > 2:
                        wall_magnet_penalty = -2.0
                        if self.treasure_blocked_steps > 5:
                            wall_magnet_penalty = -4.0  # 卡得越久惩罚越重
            else:
                self.near_treasure_orbit_steps = 0

        # 更新宝箱距离记录（只保存有限值）
        if np.isfinite(nearest_treasure_dist):
            self.last_treasure_dist = nearest_treasure_dist
        else:
            self.last_treasure_dist = None
        self.last_treasure_pos = nearest_treasure_pos

        # 7.7 极近宝箱时的直线移动奖励（鼓励最后走直线吃宝箱，受威胁衰减）
        straight_approach_bonus = 0.0
        if (nearest_treasure_pos is not None
                and np.isfinite(nearest_treasure_dist)
                and nearest_treasure_dist < 3.0
                and last_action >= 0):
            # 直线动作: 0(右), 1(下), 2(左), 3(上)
            if last_action in [0, 1, 2, 3]:
                straight_approach_bonus = 0.5 * total_decay
            # 斜向动作在极近宝箱时轻微惩罚，鼓励走直线
            elif last_action in [4, 5, 6, 7]:
                straight_approach_bonus = -0.3

        # 7.8 宝箱视野内存在奖励（受威胁衰减）
        treasure_view_bonus = 0.0
        if (nearest_treasure_pos is not None
                and np.isfinite(nearest_treasure_dist)
                and nearest_treasure_dist < 10.0):
            treasure_view_bonus = 0.02 * total_decay

        # =========================================================================
        # 7.9 朝怪物方向闪现奖励（核心新增）
        # 思路：当怪物在2-3格内时，朝怪物方向闪现可以穿越怪物逃生
        # 闪现2格 → 怪物在身后 → 拉开距离 → 生存
        # =========================================================================
        flash_through_monster_bonus = 0.0

        if last_action >= 8 and flash_available > 0.5:
            # 上一步执行了闪现动作
            # 找到最近的可见怪物
            closest_monster_raw_dist = float('inf')
            closest_monster_dir_x = 0.0
            closest_monster_dir_z = 0.0
            for m in monsters:
                if m.get("is_in_view", 0):
                    m_pos = m["pos"]
                    raw_dist = np.sqrt(
                        (hero_pos["x"] - m_pos["x"]) ** 2 +
                        (hero_pos["z"] - m_pos["z"]) ** 2
                    )
                    if raw_dist < closest_monster_raw_dist:
                        closest_monster_raw_dist = raw_dist
                        # 怪物相对英雄的方向（英雄→怪物）
                        closest_monster_dir_x, closest_monster_dir_z = _get_direction_to_target(hero_pos, m_pos)

            # 只在怪物距离2-3格时触发（太远没意义，太近可能已被抓）
            if (FLASH_MONSTER_DIST_MIN < closest_monster_raw_dist < FLASH_MONSTER_DIST_MAX):
                # 计算闪现动作方向与怪物方向的对齐度
                direction_match = _get_action_direction_match(last_action, closest_monster_dir_x, closest_monster_dir_z)

                if direction_match > 0.3:
                    # 朝怪物方向闪现（方向匹配度>0.3），给予高额奖励
                    # 奖励与匹配度和危险程度成正比
                    danger_factor = 1.0 - (closest_monster_raw_dist - FLASH_MONSTER_DIST_MIN) / (FLASH_MONSTER_DIST_MAX - FLASH_MONSTER_DIST_MIN)
                    danger_factor = max(0.0, min(1.0, danger_factor))  # 越近越危险=奖励越大
                    flash_through_monster_bonus = 8.0 * direction_match * danger_factor
                elif direction_match < -0.3:
                    # 闪现方向与怪物方向相反（远离怪物方向闪现），轻微奖励（也不错但不如穿越）
                    flash_through_monster_bonus = 1.0 * abs(direction_match)

        # =========================================================================
        # 7.10 探索奖励 + 停滞惩罚
        # 核心思路：鼓励智能体探索未访问区域，惩罚长时间停留在同一区域
        # =========================================================================
        exploration_reward = 0.0
        idle_penalty = 0.0

        # --- 新区域探索奖励 ---
        # 进入新的网格单元时给奖励（首次访问）
        if self.visit_grid[grid_z, grid_x] == 1:
            # 首次访问该网格，给探索奖励
            # 探索比例越低（早期），奖励越大；探索比例越高（后期），奖励递减
            exploration_bonus_factor = max(0.3, 1.0 - visited_ratio)
            exploration_reward = 2.0 * exploration_bonus_factor

        # --- 重访次数衰减奖励 ---
        # 即使不是首次访问，只要不是频繁重访也给微小奖励（鼓励移动）
        elif self.visit_grid[grid_z, grid_x] <= 3:
            exploration_reward = 0.2

        # --- 停滞惩罚 ---
        # 在同一网格内停滞超过一定步数就惩罚
        if self.idle_steps > 10:
            # 停滞10步以上开始惩罚，越久惩罚越重
            idle_penalty = -0.05 * (self.idle_steps - 10)
            # 封顶惩罚
            idle_penalty = max(idle_penalty, -3.0)

        # --- 无怪物威胁时加强探索激励 ---
        # 当没有怪物在视野内时，加大探索奖励和停滞惩罚
        any_monster_visible = any(m.get("is_in_view", 0) for m in monsters)
        if not any_monster_visible:
            # 无怪物威胁：探索奖励翻倍，停滞惩罚加重
            exploration_reward *= 2.0
            if self.idle_steps > 5:
                # 5步以上就开始惩罚（比有怪物时更严格）
                idle_penalty = -0.1 * (self.idle_steps - 5)
                idle_penalty = max(idle_penalty, -5.0)

        # =========================================================================
        # 7.11 500步后内侧贴墙奖励
        # 思路：怪物加速后，英雄贴道路内边缘走可以让怪物走更长的外圈路径
        # 地图左侧→贴道路右边(内侧)，地图右侧→贴道路左边(内侧)
        # 地图上方→贴道路下方(内侧)，地图下方→贴道路上方(内侧)
        # =========================================================================
        inner_wall_bonus = 0.0

        if self.step_no > 500 and last_action >= 0 and last_action < 8 and map_info is not None and len(map_info) >= 21:
            center = len(map_info) // 2

            # 判断英雄在地图的大致位置（0~1，0=最左/最上，1=最右/最下）
            hero_pos_x_ratio = hero_pos["x"] / MAP_SIZE  # 水平位置比例
            hero_pos_z_ratio = hero_pos["z"] / MAP_SIZE  # 垂直位置比例

            # 计算各方向到墙的距离（已有wall_dist_8dir，但这里需要更精确的左右距离对比）
            # 方向映射: 0右, 1下, 2左, 3上, 4右下, 5左下, 6左上, 7右上
            dist_right = _count_wall_in_direction(map_info, center, center, (0, 1), max_dist=5)
            dist_left = _count_wall_in_direction(map_info, center, center, (0, -1), max_dist=5)
            dist_down = _count_wall_in_direction(map_info, center, center, (1, 0), max_dist=5)
            dist_up = _count_wall_in_direction(map_info, center, center, (-1, 0), max_dist=5)

            # 判断当前移动方向是否朝内侧贴墙
            # 动作方向: 0右, 1下, 2左, 3上, 4右下, 5左下, 6左上, 7右上
            is_moving_inner = False

            # 英雄在地图左侧 (x < 40%) → 内侧是右 → 应贴右侧墙走
            # 贴右侧墙走 = 右边墙近 + 往上或往下走
            if hero_pos_x_ratio < 0.4:
                # 在左侧，内侧是右。贴右侧墙 = dist_right小(1~2) 且 向下/上移动
                if dist_right <= 2 and last_action in [1, 3]:  # 下或上
                    is_moving_inner = True

            # 英雄在地图右侧 (x > 60%) → 内侧是左 → 应贴左侧墙走
            elif hero_pos_x_ratio > 0.6:
                # 在右侧，内侧是左。贴左侧墙 = dist_left小(1~2) 且 向下/上移动
                if dist_left <= 2 and last_action in [1, 3]:  # 下或上
                    is_moving_inner = True

            # 英雄在地图上方 (z < 40%) → 内侧是下 → 应贴下方墙走
            if hero_pos_z_ratio < 0.4:
                # 在上方，内侧是下。贴下方墙 = dist_down小(1~2) 且 向右/左移动
                if dist_down <= 2 and last_action in [0, 2]:  # 右或左
                    is_moving_inner = True

            # 英雄在地图下方 (z > 60%) → 内侧是上 → 应贴上方墙走
            elif hero_pos_z_ratio > 0.6:
                # 在下方，内侧是上。贴上方墙 = dist_up小(1~2) 且 向右/左移动
                if dist_up <= 2 and last_action in [0, 2]:  # 右或左
                    is_moving_inner = True

            if is_moving_inner:
                # 贴内侧墙走，给奖励。步数越多（怪物越快），奖励越大
                late_game_factor = min(1.0, (self.step_no - 500) / 500)  # 500步后逐步增大
                inner_wall_bonus = 1.5 * (0.3 + 0.7 * late_game_factor)

        # 保存上次动作
        self.last_action = last_action

        # 总奖励（数值安全：clamp 防止 inf/nan）
        raw_reward = (survive_reward + dist_shaping + wall_penalty + stuck_penalty +
                      treasure_reward + treasure_shaping + orbiting_penalty +
                      wall_magnet_penalty + straight_approach_bonus + treasure_view_bonus +
                      flash_through_monster_bonus + exploration_reward + idle_penalty +
                      inner_wall_bonus)
        if not np.isfinite(raw_reward):
            raw_reward = 0.0
        reward = [np.clip(raw_reward, -50.0, 50.0)]

        if self.step_no % 50 == 0:
            print(f"[INFO] step={self.step_no}, feature_dim={len(feature)}, "
                  f"treasure_dist={nearest_treasure_dist:.1f}, "
                  f"orbit_steps={self.near_treasure_orbit_steps}, "
                  f"blocked_steps={self.treasure_blocked_steps}, "
                  f"monster_decay={monster_threat_decay:.2f}, "
                  f"total_decay={total_decay:.2f}, "
                  f"treasure_reward={treasure_reward:.1f}, "
                  f"flash_bonus={flash_through_monster_bonus:.2f}, "
                  f"explore_reward={exploration_reward:.2f}, "
                  f"idle_penalty={idle_penalty:.2f}, "
                  f"inner_wall={inner_wall_bonus:.2f}, "
                  f"visited={visited_ratio:.0%}, "
                  f"idle_steps={self.idle_steps}")

        return feature, legal_action, reward
