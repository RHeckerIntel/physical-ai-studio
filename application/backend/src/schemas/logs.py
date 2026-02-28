# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import BaseModel


class LogSource(BaseModel):
    """Describes an available log source (worker log file or per-job log file)."""

    id: str
    name: str
    type: Literal["worker", "job"]
