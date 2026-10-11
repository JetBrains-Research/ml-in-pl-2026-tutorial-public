# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_utils import ServerClient

from experiments.module_1_python.resources_servers.idegym_python.app import (
    PythonResourcesConfig,
    PythonResourcesServer,
    PythonVerifyRequest,
)
from experiments.module_1_python.workshop_support import extract_code, output_matches


def make_request(text: str = "def solve(data): return sum(data)") -> PythonVerifyRequest:
    return PythonVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": "Sum the list"},
            "verifier_metadata": {
                "task_id": "sum-list",
                "test_cases": [{"input": [1, 2], "expected": 3}],
            },
            "response": {
                "id": "test-response",
                "created_at": 0,
                "object": "response",
                "model": "test",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "type": "message",
                        "id": "m1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
            },
        }
    )


@pytest.fixture
def server() -> PythonResourcesServer:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"sandbox": {"idegym": {}}}
    return PythonResourcesServer(
        config=PythonResourcesConfig(host="127.0.0.1", port=9000, name="test", entrypoint="app.py", image="test:1"),
        server_client=client,
    )


@pytest.mark.parametrize(
    ("result", "reward"),
    [
        (SandboxExecResult("3", "", 0), 1),
        (SandboxExecResult("4", "", 0), 0),
        (SandboxExecResult("3", "error", 1), 0),
        (SandboxExecResult("", "SyntaxError", 1), 0),
        (SandboxExecResult("", "timed out", -1, "timeout"), 0),
        (SandboxExecResult("not JSON", "", 0), 0),
    ],
)
async def test_reward_and_cleanup(server: PythonResourcesServer, result: SandboxExecResult, reward: int) -> None:
    sandbox = AsyncMock()
    sandbox.__aenter__.return_value = sandbox
    sandbox.exec.side_effect = [SandboxExecResult("Python 3.12", "", 0), result]
    with patch("experiments.module_1_python.workshop.AsyncSandbox", return_value=sandbox) as factory:
        response = await server.verify(make_request())
    assert response.reward == reward
    assert response.passed_tests == reward
    assert response.total_tests == 1
    sandbox.__aexit__.assert_awaited_once()
    spec = factory.call_args.args[1]
    assert spec.files == {"/tmp/solution.py": "def solve(data): return sum(data)"}


async def test_infrastructure_error_is_not_zero_reward(server: PythonResourcesServer) -> None:
    sandbox = AsyncMock()
    sandbox.__aenter__.return_value = sandbox
    sandbox.exec.side_effect = [SandboxExecResult("Python 3.12", "", 0), SandboxExecResult("", "gone", -1, "sandbox")]
    with patch("experiments.module_1_python.workshop.AsyncSandbox", return_value=sandbox):
        with pytest.raises(RuntimeError, match="IDEGym execution failed"):
            await server.verify(make_request())
    sandbox.__aexit__.assert_awaited_once()


async def test_empty_answer_does_not_create_sandbox(server: PythonResourcesServer) -> None:
    with patch("experiments.module_1_python.workshop.AsyncSandbox") as factory:
        result = await server.verify(make_request(""))
    assert result.reward == 0
    assert result.failure_reason == "empty_model_output"
    factory.assert_not_called()


def test_code_extraction_and_strict_output() -> None:
    assert extract_code("<think>plan</think>\n```python\ndef solve(x): return x\n```") == "def solve(x): return x"
    assert output_matches("true", True)
    assert not output_matches("1", True)
    assert not output_matches("debug\n3", 3)
