# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numbered CLI adapter for the central Module 1 workshop script."""

import click

from experiments.module_1_python.workshop import main


@click.group()
def cli() -> None:
    """Generate and verify Python solutions from a sample file."""


cli.add_command(main, "run")
