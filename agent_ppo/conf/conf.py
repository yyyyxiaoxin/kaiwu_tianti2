#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""


class Config:

    # 特征维度（共53D）
    # - hero_self: 4D
    # - monster_1: 7D (5D基础 + 方向2D)
    # - monster_2: 7D (5D基础 + 方向2D)
    # - wall_dist_8dir: 8D (8方向到墙距离)
    # - treasure: 5D (方向2D + 距离1D + 是否极近1D + 收集数1D)
    # - flash_available: 1D (闪现是否可用)
    # - legal_action: 16D (8移动 + 8闪现)
    # - exploration: 3D (已探索区域比例1D + 本格访问次数1D + 停滞步数1D)
    # - progress: 2D
    FEATURES = [
        4,   # hero_self
        7,   # monster_1
        7,   # monster_2
        8,   # wall_dist_8dir
        5,   # treasure (方向2 + 距离1 + 极近标志1 + 收集数1)
        1,   # flash_available
        16,  # legal_action (8移动 + 8闪现)
        3,   # exploration (已探索比例 + 当前格访问次数 + 停滞步数)
        2,   # progress
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：8个移动方向 + 8个闪现方向 = 16
    ACTION_NUM = 16

    # Value head / 价值头：单头生存奖励
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.01          # 提高熵系数 0.001→0.01，增加探索避免卡在斜向走
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
