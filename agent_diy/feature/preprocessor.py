#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
=============================================================================
agent_diy 特征预处理（基于 agent_ppo_y_1 第1步改进）
=============================================================================
【第1步改进】扩展 legal_act 为 16 维（支持闪现）
"""

import numpy as np


# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0

# 【新增】墙壁惩罚参数（超大力惩罚！）
WALL_AHEAD_PENALTY = -5.0  # 向墙壁移动惩罚
FLASH_WALL_AHEAD_PENALTY = 0.0  # 闪现方向惩罚调为0（不鼓励乱用闪现）

# 动作方向偏移表（col=x轴向右, row=z轴向下）
# 0-7: 右、下、左、上、右下、左下、左上、右上
DIRECTION_OFFSETS = {
    0: (1, 0),   # 右
    1: (0, 1),   # 下
    2: (-1, 0),  # 左
    3: (0, -1),  # 上
    4: (1, 1),   # 右下
    5: (-1, 1),  # 左下
    6: (-1, -1), # 左上
    7: (1, -1),  # 右上
    # 闪现方向（8-15 对应 0-7）
    8: (2, 0),   # 闪现右
    9: (0, 2),   # 闪现下
    10: (-2, 0), # 闪现左
    11: (0, -2), # 闪现上
    12: (2, 2),  # 闪现右下
    13: (-2, 2), # 闪现左下
    14: (-2, -2),# 闪现左上
    15: (2, -2), # 闪现右上
}


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _is_wall_ahead(map_info, center, action_id):
    """检测动作方向前方是否有墙。

    Args:
        map_info: 地图信息（二维数组，0=空地, 1=障碍物, 2=宝箱）
        center: 地图中心索引
        action_id: 动作ID (0-15)

    Returns:
        True if the direction leads to a wall or is out of bounds
    """
    if map_info is None or action_id < 0 or action_id not in DIRECTION_OFFSETS:
        return False

    dx, dz = DIRECTION_OFFSETS[action_id]
    # 检查路径上所有格子（考虑斜向移动）
    if dx != 0 and dz != 0:
        # 斜向移动：检查直线上的格子是否可通行
        steps = max(abs(dx), abs(dz))
        for step in range(1, steps + 1):
            check_dx = dx * step // steps
            check_dz = dz * step // steps
            nr, nc = center + check_dz, center + check_dx
            if not (0 <= nr < len(map_info) and 0 <= nc < len(map_info[0])):
                return True  # 越界=墙
            if int(map_info[nr][nc]) == 1:
                return True  # 撞墙
    else:
        # 直行移动
        nr, nc = center + dz, center + dx
        if not (0 <= nr < len(map_info) and 0 <= nc < len(map_info[0])):
            return True  # 越界=墙
        if int(map_info[nr][nc]) == 1:
            return True  # 撞墙

    return False


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.last_treasure_dist_norm = 1.0  # 上一步宝箱距离
        self.last_treasure_collected = 0     # 上一步收集的宝箱数
        # 【新增】历史怪物信息
        self.last_monster_dists = [1.0, 1.0]  # 两个怪物的上一步距离
        # 【改进】历史位置（用于探索奖励）
        self.last_hero_pos = None  # 上一步英雄位置 (x, z)
        self.visited_cells = set()  # 已访问的格子集合
        # 【新增】死胡同状态追踪
        self.was_in_dead_end = False  # 上一步是否在死胡同

    def feature_process(self, env_obs, last_action=-1):
        """Process env_obs into feature vector, legal_act mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        # =========================================================================
        # Hero self features (4D) / 英雄自身特征
        # =========================================================================
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero["flash_cooldown"], MAX_FLASH_CD)
        buff_remain_norm = _norm(hero["buff_remaining_time"], MAX_BUFF_DURATION)

        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # =========================================================================
        # Monster features (5D x 2) / 怪物特征
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

                    # Euclidean distance / 欧式距离
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        # =========================================================================
        # 【改进】出口检测特征（识别死胡同）- 4D
        # 计算英雄周围4个方向的空格数量，帮助识别死胡同
        # =========================================================================
        exit_count = 0  # 周围空格数量
        exit_forward = 0.0  # 前方是否有出口
        has_escape_route = 1.0  # 是否有逃脱路线（≥2个方向有出口）
        dead_end_warning = 0.0  # 死胡同警告

        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            # 4个方向：上、下、左、右（相对于英雄朝向，假设z轴向下为正）
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (row_offset, col_offset)
            exit_dirs = 0
            for dr, dc in directions:
                nr, nc = center + dr, center + dc
                if 0 <= nr < len(map_info) and 0 <= nc < len(map_info[0]):
                    cell_value = int(map_info[nr][nc])
                    if cell_value == 0:  # 空地
                        exit_dirs += 1
            exit_count = exit_dirs
            exit_forward = 1.0 if exit_dirs > 0 else 0.0
            has_escape_route = 1.0 if exit_dirs >= 2 else 0.0
            dead_end_warning = 1.0 if exit_dirs == 1 else 0.0  # 只有1个出口=死胡同

        exit_feat = np.array([exit_forward, has_escape_route, dead_end_warning, exit_count / 4.0], dtype=np.float32)

        # =========================================================================
        # Local map features (16D) / 局部地图特征
        # map_info 中: 0=空地, 1=障碍物, 2=宝箱
        # =========================================================================
        map_feat = np.zeros(16, dtype=np.float32)
        treasure_dist_norm = 1.0  # 默认：视野内无宝箱
        treasure_count_in_view = 0  # 视野内宝箱数量
        treasure_dir_x = 0.0  # 最近宝箱方向 x
        treasure_dir_z = 0.0  # 最近宝箱方向 z
        treasure_collected = 0

        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            flat_idx = 0
            nearest_treasure_dist = float('inf')
            for row in range(center - 2, center + 2):
                for col in range(center - 2, center + 2):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        cell_value = int(map_info[row][col])
                        if cell_value == 1:
                            map_feat[flat_idx] = 1.0  # 障碍物
                        elif cell_value == 2:
                            map_feat[flat_idx] = 0.5  # 宝箱（标记为0.5）
                            treasure_count_in_view += 1
                            # 计算宝箱距离（以英雄位置为中心）
                            rel_row = row - center
                            rel_col = col - center
                            raw_dist = np.sqrt(rel_row ** 2 + rel_col ** 2)
                            if raw_dist < nearest_treasure_dist:
                                nearest_treasure_dist = raw_dist
                                treasure_dist_norm = _norm(raw_dist, 4.0)
                                # 归一化方向向量
                                if raw_dist > 0:
                                    treasure_dir_x = rel_col / raw_dist
                                    treasure_dir_z = rel_row / raw_dist
                    flat_idx += 1

        # 获取环境信息中的宝箱收集数
        treasure_collected = env_info.get("treasures_collected", 0)

        # =========================================================================
        # 【新增】Organs 全局宝箱信息（从 organs 获取视野外的宝箱位置）
        # 规则允许：从 organs 获取宝箱位置/距离，即使在视野外也可以
        # =========================================================================
        global_treasure_dist_norm = 1.0  # 默认：无法获取全局宝箱信息
        global_treasure_dir_x = 0.0     # 全局最近宝箱方向 x
        global_treasure_dir_z = 0.0     # 全局最近宝箱方向 z
        total_treasure_count = 0         # 场上总宝箱数

        # 尝试从 organs 获取宝箱信息
        organs = frame_state.get("organs", [])
        # 【DEBUG】每步打印宝箱数据（评估时启用）
        print(f"[DEBUG] step={self.step_no}, organs_count={len(organs)}, "
              f"total_treasure={total_treasure_count}, "
              f"in_view={treasure_count_in_view}, collected={treasure_collected}")
        if organs and len(organs) > 0:
            # 计算场上总宝箱数
            total_treasure_count = len(organs)

            # 找到最近的宝箱
            nearest_treasure_dist = float('inf')
            for treasure in organs:
                t_pos = treasure.get("pos", {})
                if t_pos:
                    t_x = t_pos.get("x", hero_pos["x"])
                    t_z = t_pos.get("z", hero_pos["z"])
                    raw_dist = np.sqrt((hero_pos["x"] - t_x) ** 2 + (hero_pos["z"] - t_z) ** 2)
                    if raw_dist < nearest_treasure_dist:
                        nearest_treasure_dist = raw_dist
                        global_treasure_dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                        # 归一化方向向量
                        if raw_dist > 0:
                            global_treasure_dir_x = (t_x - hero_pos["x"]) / (raw_dist * MAP_SIZE)
                            global_treasure_dir_z = (t_z - hero_pos["z"]) / (raw_dist * MAP_SIZE)
                        else:
                            global_treasure_dir_x = 0.0
                            global_treasure_dir_z = 0.0

        # =========================================================================
        # 【增强】Treasure features (6D) / 宝箱特征（支持视野外宝箱）
        # 规则允许：从 organs 获取视野外的宝箱位置和距离
        # =========================================================================
        treasure_feat = np.array([
            float(treasure_count_in_view > 0),   # 视野内是否有宝箱
            float(treasure_count_in_view) / 4.0,  # 视野内宝箱数量（归一化，最多4个）
            # 优先使用全局宝箱距离（视野外），其次使用视野内距离
            global_treasure_dist_norm if total_treasure_count > 0 else treasure_dist_norm,
            global_treasure_dir_x if total_treasure_count > 0 else treasure_dir_x,
            global_treasure_dir_z if total_treasure_count > 0 else treasure_dir_z,
            float(treasure_collected) / 10.0,   # 已收集宝箱数（归一化，最多10个）
        ], dtype=np.float32)

        # =========================================================================
        # 【核心改动点】Legal act mask (16D) / 合法动作掩码
        # 原版: 只处理 8 维移动 (0-7)
        # 改进: 处理全部 16 维 (0-7移动 + 8-15闪现)
        # 重要: 基于地图信息禁止向墙壁移动的动作！
        # =========================================================================
        legal_act = np.zeros(16, dtype=np.float32)

        # 获取地图中心位置（用于墙壁检测）
        if map_info is not None and len(map_info) >= 5:
            center = len(map_info) // 2
        else:
            center = 2  # 默认中心位置

        # 处理 8 维移动动作（根据墙壁检测设置）
        for action_id in range(8):
            # 检测该方向是否有墙
            if _is_wall_ahead(map_info, center, action_id):
                legal_act[action_id] = 0.0  # 向墙移动 = 禁止
            else:
                legal_act[action_id] = 1.0  # 安全方向 = 可用

        # 处理 8 维闪现动作（根据环境和墙壁检测）
        for j in range(8, 16):
            # 检测闪现方向是否有墙
            if _is_wall_ahead(map_info, center, j):
                # 闪现方向有墙，但规则说会退回到最后一格，所以可以允许
                # 但还是需要结合环境的 legal_act_raw
                env_legal = 0.0
                if isinstance(legal_act_raw, list) and j < len(legal_act_raw):
                    env_legal = float(legal_act_raw[j])
                # 如果环境允许且不是完全撞墙（在路径上有可通行格子），则允许
                if env_legal > 0:
                    legal_act[j] = 1.0
                else:
                    legal_act[j] = 0.0
            else:
                # 闪现方向安全
                if isinstance(legal_act_raw, list) and j < len(legal_act_raw):
                    legal_act[j] = float(legal_act_raw[j])
                else:
                    legal_act[j] = 1.0

        # 【安全检查】如果所有动作都不可用，至少保证有一个安全移动方向
        if legal_act.sum() == 0:
            # 找到任意一个非墙方向
            for action_id in range(8):
                if not _is_wall_ahead(map_info, center, action_id):
                    legal_act[action_id] = 1.0
                    break
            # 如果还是找不到，说明完全被墙包围，强制允许右移
            if legal_act.sum() == 0:
                legal_act[0] = 1.0

        # =========================================================================
        # Progress features (2D) / 进度特征
        # =========================================================================
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # Concatenate features / 拼接特征 (58D)
        # 4 + 10 + 6 + 4 + 16 + 16 + 2 = 58D
        feature = np.concatenate(
            [
                hero_feat,          # 4D
                monster_feats[0],   # 5D
                monster_feats[1],   # 5D
                treasure_feat,      # 6D
                exit_feat,          # 4D 【新增】出口检测特征
                map_feat,           # 16D
                legal_act,          # 16D
                progress_feat,      # 2D
            ]
        )

        # =========================================================================
        # Reward / 奖励（优化版：平衡生存与收集）
        # =========================================================================
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[4])

        # =========================================================================
        # 【v2.0改进】奖励结构调整（彻底解决"卡着不动"问题）
        # 核心思路：移除被动生存奖励，奖励必须通过移动获得
        # =========================================================================

        # 0. 【新增】方向性墙壁惩罚（在撞墙之前就检测并惩罚）
        wall_ahead_penalty = 0.0
        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            # 检测上一步动作方向是否有墙
            if last_action >= 0 and last_action < 16:
                if _is_wall_ahead(map_info, center, last_action):
                    wall_ahead_penalty = WALL_AHEAD_PENALTY  # 统一惩罚

        # 1. 【核心改进】有效移动奖励（只有真正移动了才给奖励）
        move_bonus = 0.0
        wall_bump_penalty = 0.0  # 撞墙惩罚（只有没移动成功时才触发）
        if self.last_hero_pos is not None:
            move_dist = np.sqrt((hero_pos["x"] - self.last_hero_pos[0]) ** 2 +
                                (hero_pos["z"] - self.last_hero_pos[1]) ** 2)

            if move_dist < 0.5:  # 几乎没移动（被挡住或卡住）
                # 检查是否在朝障碍物方向（有出口但没移动成功）
                if has_escape_route > 0.5:  # 有逃脱路线但没移动 → 超级大惩罚！
                    wall_bump_penalty = -5.0
                    move_bonus = -3.0
                else:
                    # 真正的死胡同，超大惩罚
                    move_bonus = -2.0
            else:
                # 成功移动，给予奖励
                move_bonus = 0.02 * min(move_dist / 5.0, 1.0)  # 移动距离越大，奖励越高

                # 【极端奖励】如果成功移动，给予惩罚（移动会撞墙）
                survive_reward = -5.0
        else:
            # 第一次调用或无历史位置，站着不动 = 高奖励
            move_dist = 0.0
            survive_reward = 20.0  # 站立 = +20分/步！

        # 1. 【完全移除】被动生存奖励（这是导致"卡着不动"的根源！）
        # 英雄必须通过移动来获得分数，而不是躺着不动
        survive_reward = 0.0

        # 2. 【改进】加强怪物躲避奖励（核心改动）
        dist_shaping = 0.3 * (cur_min_dist_norm - self.last_min_monster_dist_norm)  # 0.1 → 0.3

        # 3. 宝箱收集奖励（保持）
        treasure_reward = 0.0
        if treasure_collected > self.last_treasure_collected:
            treasure_reward = 10.0 * (treasure_collected - self.last_treasure_collected)

        # 4. 【改进】降低宝箱接近奖励，减少冒险收集
        if total_treasure_count > 0:
            current_treasure_dist = global_treasure_dist_norm
            treasure_shaping = 0.05 + 0.1 * (self.last_treasure_dist_norm - current_treasure_dist)  # 降低
        elif treasure_count_in_view > 0:
            treasure_shaping = 0.08 + 0.15 * (self.last_treasure_dist_norm - treasure_dist_norm)
        else:
            treasure_shaping = 0.02 * (self.last_treasure_dist_norm - treasure_dist_norm)

        # 5. 【改进】安全距离奖励（提高阈值）
        safe_bonus = 0.03 if cur_min_dist_norm > 0.4 else 0.0  # 0.5 → 0.4

        # 6. 【大幅加强】危险惩罚（核心改动 - 解决不躲避怪物问题）
        danger_penalty = -0.5 if cur_min_dist_norm < 0.15 else (-0.2 if cur_min_dist_norm < 0.25 else 0.0)
        # 极危险(<0.15): -0.5 | 危险(<0.25): -0.2 | 安全: 0

        # 7. 视野内宝箱存在奖励（降低）
        treasure_view_bonus = 0.005 if treasure_count_in_view > 0 else 0.0  # 0.01 → 0.005

        # 8. 【新增】死胡同惩罚（解决问题2）
        dead_end_penalty = -0.05 if dead_end_warning > 0.5 else 0.0
        # 如果在死胡同里且附近有怪物，双重惩罚
        if dead_end_warning > 0.5 and cur_min_dist_norm < 0.3:
            dead_end_penalty = -0.2  # 死胡同 + 危险 = 严重惩罚

        # 9. 【新增】逃脱奖励（解决问题2）
        escape_bonus = 0.0
        if self.last_hero_pos is not None:
            # 如果从死胡同逃出来了，给予奖励
            if self.was_in_dead_end and dead_end_warning < 0.5:
                escape_bonus = 0.1
        self.was_in_dead_end = dead_end_warning > 0.5

        # 10. 探索奖励（保持）
        explore_bonus = 0.0
        current_cell = (int(hero_pos["x"]), int(hero_pos["z"]))
        if self.last_hero_pos is not None:
            if current_cell not in self.visited_cells:
                explore_bonus = 0.015
                self.visited_cells.add(current_cell)
            last_dist = np.sqrt((current_cell[0] - self.last_hero_pos[0]) ** 2 +
                               (current_cell[1] - self.last_hero_pos[1]) ** 2)
            if last_dist > 2:
                explore_bonus += 0.01

        self.last_min_monster_dist_norm = cur_min_dist_norm
        # 使用全局距离更新历史距离（规则允许使用视野外的宝箱信息）
        if total_treasure_count > 0:
            self.last_treasure_dist_norm = global_treasure_dist_norm
        else:
            self.last_treasure_dist_norm = treasure_dist_norm
        self.last_treasure_collected = treasure_collected

        # 【改进】更新历史位置
        self.last_hero_pos = (int(hero_pos["x"]), int(hero_pos["z"]))
        if self.step_no == 0:  # 每局开始时清空访问记录
            self.visited_cells = set()

        # 汇总奖励（包含所有新奖励项）
        reward = [
            move_bonus + wall_bump_penalty + wall_ahead_penalty + survive_reward + dist_shaping + treasure_reward +
            treasure_shaping + safe_bonus + danger_penalty + treasure_view_bonus +
            dead_end_penalty + escape_bonus + explore_bonus
        ]


        # 【DEBUG】每步打印奖励分解（评估时启用）
        print(f"[REWARD] step={self.step_no}, action={last_action}, "
              f"move={move_bonus:.3f}, wall_bump={wall_bump_penalty:.3f}, wall_ahead={wall_ahead_penalty:.3f}, "
              f"danger={danger_penalty:.3f}, dist={dist_shaping:.3f}, "
              f"treasure={treasure_reward:.3f}, dead_end={dead_end_penalty:.3f}, "
              f"monster_dist={cur_min_dist_norm:.3f}, exit={exit_count}")

        return feature, legal_act, reward
