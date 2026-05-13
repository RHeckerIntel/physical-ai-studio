# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference output field name constants.

Canonical key names for inference pipeline outputs, enabling IDE
autocomplete and safe refactoring across the inference module.
"""

IMAGES = "images"
ACTION = "action"
TASK = "task"
STATE = "state"

TOKENIZED_PROMPT = "tokenized_prompt"
TOKENIZED_PROMPT_MASK = "tokenized_prompt_mask"
IMAGE_MASKS = "image_masks"

# RTC-specific keys
PREV_CHUNK_LEFT_OVER = "prev_chunk_left_over"
INFERENCE_DELAY = "inference_delay"
NOISE = "noise"

__all__ = [
    "ACTION",
    "IMAGES",
    "IMAGE_MASKS",
    "INFERENCE_DELAY",
    "NOISE",
    "PREV_CHUNK_LEFT_OVER",
    "STATE",
    "TASK",
    "TOKENIZED_PROMPT",
    "TOKENIZED_PROMPT_MASK",
]
