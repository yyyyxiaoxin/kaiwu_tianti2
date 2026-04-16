#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。

特征维度（共42D）：
- hero_self: 4D
- monster_1: 7D (5D基础 + 方向2D)
- monster_2: 7D (5D基础 + 方向2D)
- wall_dist_8dir: 8D (8方向到墙距离)
- treasure: 5D (方向2D + 距离1D + 是否极近1D + 收集数1D)
- flash_available: 1D (闪现技能是否可用)
- legal_action: 16D (8移动 + 8闪现)
- progress: 2D

修改要点：
1. 英雄模型为1×1（非3×3），移除错误的3×3碰撞检测
2. 增加宝箱距离特征，解决斜向走绕宝箱问题
3. 增加绕宝箱惩罚和接近宝箱奖励
4. 动作空间扩展为16维（8移动+8闪现），支持闪现技能
5. 怪物近距离时，朝怪物反向闪现给予最高奖励（极危险时反向逃跑）
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

# 闪现触发距离阈值
FLASH_MONSTER_DIST_VERY_CLOSE = 2.0  # 2.0格以内：极危险，必须反向闪现
FLASH_MONSTER_DIST_CLOSE = 3.5        # 3.5格以内：较危险，反向或穿越都可以

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
    """计算某个方向上到墙壁的距离。

    对于斜向(1,1)类型的方向，斜走一格需要两步经过两个直角边格子，
    检测时需同时检查两步是否被堵（防止2x2角落误判为畅通）。
    """
    dr, dc = direction
    is_diagonal = (dr != 0 and dc != 0)

    for step in range(1, max_dist + 1):
        row = center_row + dr * step
        col = center_col + dc * step

        if not (0 <= row < len(map_info) and 0 <= col < len(map_info[0])):
            return float(step - 1)
        if map_info[row][col] != 0:
            return float(step - 1)

        # 斜向移动：还需检查直角边格子（斜走两步过一格）
        # 从(step-1, step-1)位置尝试向(dr, dc)方向斜走，
        # 需要检查(步数*dr, 0)和(0, 步数*dc)两个直角边位置
        if is_diagonal:
            row_side1 = center_row + dr * (step - 1) + 0   # (dr*(step-1), 0)
            col_side1 = center_col + dc * (step - 1) + dc  # (0, dc*(step-1)+dc)
            row_side2 = center_row + dr * (step - 1) + dr  # (dr*(step-1)+dr, 0)
            col_side2 = center_col + dc * (step - 1) + 0   # (0, dc*(step-1))

            # 只检查step=1时的主直角边（2x2角落的判断关键）
            if step == 1:
                for r, c in [(row_side1, col_side1), (row_side2, col_side2)]:
                    if not (0 <= r < len(map_info) and 0 <= c < len(map_info[0])):
                        return float(step - 1)
                    if map_info[r][c] != 0:
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

    if is_diagonal:
        # 斜向移动需要检查主路径和直角边格子
        # 英雄在局部地图中心(0,0)位置，向(dx,dz)方向移动2格
        # 主路径: (0,0)→(dx,dz)→(2*dx,2*dz)
        # 直角边格子（必须畅通才能斜走）：
        #   - 动作8/12(右移): 经过(0,1)和(1,0)时需要(dz,0)和(0,dx)畅通
        #   - 动作9/13(下移): 经过(0,1)和(1,0)时需要(dz,0)和(0,dx)畅通
        #   - 动作10/14(左移): 经过(0,-1)和(-1,0)时需要(dz,0)和(0,dx)畅通
        #   - 动作11/15(上移): 经过(0,-1)和(-1,0)时需要(dz,0)和(0,dx)畅通
        #
        # 通用的直角边检查点：
        # 斜向(1,1): 需检查(0,1)和(1,0)
        # 斜向(-1,1): 需检查(0,-1)和(-1,1)
        # 斜向(1,-1): 需检查(0,1)和(1,-1)  -- wait, let me recalculate

        # 简化：对每个斜向，检测"第一步"旁边的两个直角边格子
        # 从中心(0,0)向(dx,dz)方向，第一步到(dx,dz)
        # 直角边格子 = (0,dz)和(dx,0)
        side1_r = center + 0   # (0, dz) -> row = center + 0
        side1_c = center + dz  # (0, dz) -> col = center + dz
        side2_r = center + dx  # (dx, 0) -> row = center + dx
        side2_c = center + 0   # (dx, 0) -> col = center + 0

        # 检查直角边格子（step=1时）
        for r, c in [(side1_r, side1_c), (side2_r, side2_c)]:
            if not (0 <= r < len(map_info) and 0 <= c < len(map_info[0])):
                return True
            if map_info[r][c] != 0:
                return True

        # 检查主路径上的所有格子
        for step in range(1, check_dist + 1):
            row = center + dz * step
            col = center + dx * step
            if not (0 <= row < len(map_info) and 0 <= col < len(map_info[0])):
                return True
            if map_info[row][col] != 0:
                return True
    else:
        # 直线方向
        for step in range(1, check_dist + 1):
            row = center + dz * step
            col = center + dx * step
            if not (0 <= row < len(map_info) and 0 <= col < len(map_info[0])):
                return True
            if map_info[row][col] != 0:
                return True

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

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.last_hero_pos = None
        self.consecutive_wall_hits = 0
        self.diagonal_stuck_steps = 0      # 连续尝试斜向但被墙挡住的步数
        self.attempted_diagonal_when_blocked = False  # 上一步是否尝试了被堵的斜向
        self.last_action = -1
        self.last_treasure_count = 0
        self.last_treasure_dist = None      # 上一步最近宝箱距离
        self.last_treasure_pos = None        # 上一步最近宝箱位置
        self.near_treasure_orbit_steps = 0   # 在宝箱附近绕圈的步数
        self.treasure_blocked_steps = 0      # 宝箱路径被墙阻挡的连续步数
        self.last_monster_dist_when_approaching = None  # 上次靠近宝箱时的怪物距离

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
        # 6. Progress features (2D)
        # =========================================================================
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # =========================================================================
        # 拼接特征向量 (42D)
        # =========================================================================
        feature = np.concatenate([
            hero_feat,                    # 4D
            monster_feats[0],             # 7D
            monster_feats[1],             # 7D
            wall_dist_8dir,              # 8D
            treasure_feat,               # 5D
            np.array([flash_available], dtype=np.float32),  # 1D
            np.array(legal_action, dtype=np.float32),  # 16D
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

        # =========================================================================
        # 7.3b 角落后退惩罚（新增）
        # 场景：模型在角落/窄缝处反复尝试斜向移动但被堵
        # 诊断：如果上一步尝试斜向+被墙挡住，且直线方向有空间 → 惩罚
        # 使用已有的wall_dist_8dir（已在前面计算好）
        # =========================================================================
        corner_retreat_penalty = 0.0
        is_in_corner = False

        if (map_info is not None and len(map_info) >= 21
                and self.last_action >= 0 and self.last_action < 8):
            # wall_dist_8dir: 0右 1下 2左 3上 4右下 5左下 6左上 7右上
            # 直线距离
            cardinal_dists = [wall_dist_8dir[0], wall_dist_8dir[2], wall_dist_8dir[1], wall_dist_8dir[3]]
            # 斜向距离
            diagonal_dists = [wall_dist_8dir[4], wall_dist_8dir[5], wall_dist_8dir[6], wall_dist_8dir[7]]

            min_cardinal = min(cardinal_dists)
            is_diag_action = self.last_action >= 4  # 动作是斜向

            # 检测1：当前处于角落/窄缝（至少两个斜向被堵，且直线有空间）
            if sum(1 for d in diagonal_dists if d <= 0.15) >= 2 and min_cardinal >= 0.25:
                is_in_corner = True

            # 检测2：上一步尝试斜向+被墙挡住（纯斜向被堵场景）
            if (is_diag_action and is_wall_hit and min_cardinal >= 0.25):
                self.diagonal_stuck_steps += 1
                self.attempted_diagonal_when_blocked = True
            else:
                if not is_diag_action or not is_wall_hit:
                    self.diagonal_stuck_steps = 0

            # 惩罚：连续尝试被堵的斜向动作
            if self.diagonal_stuck_steps >= 2:
                corner_retreat_penalty = -1.0 * self.diagonal_stuck_steps
                corner_retreat_penalty = max(corner_retreat_penalty, -5.0)

            # 奖励：处于角落但选择了直线方向（正确选择）
            if is_in_corner and not is_diag_action and not is_wall_hit:
                corner_retreat_penalty = 1.0  # 正确选择直线，给奖励

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
        # 7.9 闪现逃生奖励（修复版）
        # 核心思路：
        #   - 怪物极近（<2格）：必须反向闪现逃跑，给最高奖励
        #   - 怪物较近（2~3.5格）：反向闪现或穿越都可以，给中等奖励
        #   - 反向闪现比穿越奖励更高，因为更安全
        # =========================================================================
        flash_escape_bonus = 0.0

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

            # 只在怪物距离3.5格以内时触发
            if closest_monster_raw_dist < FLASH_MONSTER_DIST_CLOSE:
                # 计算闪现动作方向与怪物方向的对齐度
                # > 0 表示朝怪物方向，< 0 表示远离怪物方向
                direction_match = _get_action_direction_match(last_action, closest_monster_dir_x, closest_monster_dir_z)

                # 危险程度：越近越危险
                if closest_monster_raw_dist < FLASH_MONSTER_DIST_VERY_CLOSE:
                    # ===== 极近距离（<2格）：必须反向闪现 = 最高奖励 =====
                    # 反向闪现（direction_match < -0.3）
                    if direction_match < -0.3:
                        # 越近+方向越正（完全反向）=奖励越高
                        reverse_factor = abs(direction_match)  # 0.3~1.0
                        proximity_factor = 1.0 - closest_monster_raw_dist / FLASH_MONSTER_DIST_VERY_CLOSE
                        proximity_factor = max(0.0, min(1.0, proximity_factor))
                        # 反向闪现最高奖励：10.0 × 反向程度 × 距离紧迫度
                        flash_escape_bonus = 10.0 * reverse_factor * (0.5 + 0.5 * proximity_factor)
                    # 朝怪物方向闪现（穿越）：只给微小奖励（不如反向安全）
                    elif direction_match > 0.3:
                        flash_escape_bonus = 1.0 * direction_match

                else:
                    # ===== 较近距离（2~3.5格）：反向闪现或穿越都可以 =====
                    if direction_match < -0.3:
                        # 反向闪现：中等奖励
                        reverse_factor = abs(direction_match)
                        flash_escape_bonus = 5.0 * reverse_factor
                    elif direction_match > 0.3:
                        # 穿越闪现（朝怪物方向）：较低奖励
                        # 因为怪物不太近时穿越不是最优解
                        flash_escape_bonus = 3.0 * direction_match

        # 保存上次动作
        self.last_action = last_action

        # 总奖励（数值安全：clamp 防止 inf/nan）
        raw_reward = (survive_reward + dist_shaping + wall_penalty + stuck_penalty +
                      treasure_reward + treasure_shaping + orbiting_penalty +
                      wall_magnet_penalty + straight_approach_bonus + treasure_view_bonus +
                      flash_escape_bonus + corner_retreat_penalty)
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
                  f"flash_escape={flash_escape_bonus:.2f}")

        return feature, legal_action, reward
