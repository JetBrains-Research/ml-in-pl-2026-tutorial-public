"""The workshop owns Ray while Gym and its components share the same address."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from experiments import gym_runner


@pytest.mark.parametrize("fails", [False, True])
def test_controlled_cluster_settings_are_forwarded_to_gym_and_cleaned_up(fails, monkeypatch):
    ray = SimpleNamespace(
        init=Mock(),
        get=Mock(return_value=True),
        remote=Mock(return_value=SimpleNamespace(remote=Mock(return_value="worker-result"))),
        get_runtime_context=Mock(return_value=SimpleNamespace(gcs_address="127.0.0.1:12345")),
        shutdown=Mock(),
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setattr(sys, "argv", ["original"])
    seen = []

    def dispatch():
        seen.append(sys.argv.copy())
        if fails:
            raise RuntimeError("Gym service failed")

    monkeypatch.setitem(sys.modules, "nemo_gym.cli.main", SimpleNamespace(main=dispatch))
    result = CliRunner().invoke(gym_runner.main, ["eval", "run", "--config", "example.yaml"])
    assert (result.exit_code != 0) == fails
    ray.init.assert_called_once_with(num_cpus=2, object_store_memory=256 * 1024 * 1024, include_dashboard=False)
    ray.get.assert_called_once_with("worker-result", timeout=90)
    assert seen == [["gym", "eval", "run", "--config", "example.yaml", "++ray_head_node_address=127.0.0.1:12345"]]
    ray.shutdown.assert_called_once()
