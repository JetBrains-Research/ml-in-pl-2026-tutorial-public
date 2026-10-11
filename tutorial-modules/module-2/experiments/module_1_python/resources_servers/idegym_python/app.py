# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin Gym server adapter; the workshop owns the task and verification logic."""

import asyncio

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.sandbox import resolve_provider_config
from pydantic import PrivateAttr

from experiments.module_1_python.workshop import verify_solution
from experiments.module_1_python.workshop_support import (
    PythonResourcesConfig,
    PythonVerifyRequest,
    PythonVerifyResponse,
)


class PythonResourcesServer(SimpleResourcesServer):
    config: PythonResourcesConfig
    _semaphore: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.concurrency)

    async def verify(self, body: PythonVerifyRequest) -> PythonVerifyResponse:
        # Gym calls this endpoint after generation. Resolve the named sandbox
        # configuration to the IDEGym connection settings, then use the workshop verifier.
        provider = resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
        # Bound simultaneous verifications so each response gets its own sandbox
        # without exceeding the resources server's configured concurrency.
        async with self._semaphore:
            return await verify_solution(body, provider, self.config)


if __name__ == "__main__":
    PythonResourcesServer.run_webserver()
