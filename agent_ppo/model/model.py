#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。

改进：
1. 加深网络: 53→256→128→64（原 47→128→64）
2. 添加 LayerNorm 提升训练稳定性
3. Actor/Critic 独立层，减少策略和价值之间的干扰
4. 正交初始化
"""

import torch
import torch.nn as nn
import numpy as np

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization."""
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    """Deep MLP backbone + Actor/Critic dual heads with LayerNorm."""

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_lite"
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION  # 53
        action_num = Config.ACTION_NUM           # 16
        value_num = Config.VALUE_NUM             # 1

        # Shared backbone: 53 → 256 → 128
        self.backbone = nn.Sequential(
            make_fc_layer(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            make_fc_layer(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Actor head: 128 → 64 → 16
        self.actor_head = nn.Sequential(
            make_fc_layer(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            make_fc_layer(64, action_num),
        )

        # Critic head: 128 → 64 → 1
        self.critic_head = nn.Sequential(
            make_fc_layer(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            make_fc_layer(64, value_num),
        )

    def forward(self, obs, inference=False):
        hidden = self.backbone(obs)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
