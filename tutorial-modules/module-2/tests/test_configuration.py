import importlib
import json
import os
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from omegaconf import OmegaConf

from experiments import common

ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted(path.name for path in (ROOT / "experiments").glob("module_*") if path.is_dir())


@pytest.fixture()
def environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    clean = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("LITELLM_", "OPENROUTER_", "POC_MODEL_", "POLICY_MODEL_", "IDEGYM_", "MODEL_PROVIDER"))
    }
    monkeypatch.setattr(os, "environ", clean)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(common.os, "umask", lambda value: None)
    return tmp_path / ".env"


def write_environment(path: Path, settings: str) -> None:
    path.write_text(
        "IDEGYM_ORCHESTRATOR_URL=https://idegym.example\n"
        "IDEGYM_AUTH_USERNAME=user\nIDEGYM_AUTH_PASSWORD=password\n"
        "IDEGYM_WORKSHOP_ORCHESTRATOR_URL=https://workshop.example\n"
        "IDEGYM_WORKSHOP_AUTH_USERNAME=workshop-user\nIDEGYM_WORKSHOP_AUTH_PASSWORD=workshop-password\n" + settings
    )


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("prefix", [None, "IDEGYM", "IDEGYM_STAGING"])
def test_selected_deployment_used_for_every_module(environment: Path, module: str, prefix: str | None) -> None:
    settings = "LITELLM_BASE_URL=https://proxy.example\nLITELLM_API_KEY=key\n"
    if prefix:
        settings += f"IDEGYM_ENV_PREFIX={prefix}\n"
    if prefix == "IDEGYM_STAGING":
        settings += (
            "IDEGYM_STAGING_ORCHESTRATOR_URL=https://staging.example\n"
            "IDEGYM_STAGING_AUTH_USERNAME=staging-user\nIDEGYM_STAGING_AUTH_PASSWORD=staging-password\n"
        )
    selected = prefix or "IDEGYM_WORKSHOP"
    settings += f"{selected}_NAMESPACE=selected-namespace\n"
    write_environment(environment, settings)
    common.configure(ROOT / "experiments" / module)
    deployment = {None: "workshop", "IDEGYM": "idegym", "IDEGYM_STAGING": "staging"}[prefix]
    assert os.environ["IDEGYM_ORCHESTRATOR_URL"] == f"https://{deployment}.example"
    assert os.environ["IDEGYM_AUTH_USERNAME"] == ("user" if prefix == "IDEGYM" else f"{deployment}-user")
    assert os.environ["IDEGYM_AUTH_PASSWORD"] == ("password" if prefix == "IDEGYM" else f"{deployment}-password")
    assert os.environ["IDEGYM_NAMESPACE"] == "selected-namespace"
    if module == "module_1_python":
        config = OmegaConf.load(ROOT / "experiments" / module / "config.yaml")
        assert OmegaConf.to_container(config.sandbox.idegym.connection, resolve=True)["orchestrator_url"] == (
            f"https://{deployment}.example"
        )


def test_unselected_namespace_does_not_override_default(environment: Path) -> None:
    write_environment(
        environment,
        "IDEGYM_NAMESPACE=another-namespace\nLITELLM_BASE_URL=https://proxy.example\nLITELLM_API_KEY=key\n",
    )
    common.configure(ROOT / "experiments/module_1_python")
    assert os.environ["IDEGYM_NAMESPACE"] == "idegym"


@pytest.mark.parametrize("suffix", ["ORCHESTRATOR_URL", "AUTH_USERNAME", "AUTH_PASSWORD"])
def test_missing_workshop_setting_does_not_fall_back(environment: Path, suffix: str) -> None:
    write_environment(
        environment,
        f"IDEGYM_WORKSHOP_{suffix}=\nLITELLM_BASE_URL=https://proxy.example\nLITELLM_API_KEY=key\n",
    )
    with pytest.raises(click.ClickException, match=f"IDEGYM_WORKSHOP_{suffix}"):
        common.configure(ROOT / "experiments/module_1_python")


def test_missing_custom_deployment_reports_selected_names(environment: Path) -> None:
    write_environment(environment, "IDEGYM_ENV_PREFIX=IDEGYM_STAGING\n")
    with pytest.raises(
        click.ClickException,
        match="IDEGYM_STAGING_ORCHESTRATOR_URL, IDEGYM_STAGING_AUTH_USERNAME, IDEGYM_STAGING_AUTH_PASSWORD",
    ):
        common.configure(ROOT / "experiments/module_1_python")


def test_empty_deployment_prefix_fails(environment: Path) -> None:
    write_environment(environment, "IDEGYM_ENV_PREFIX=\n")
    with pytest.raises(click.ClickException, match="IDEGYM_ENV_PREFIX must not be empty"):
        common.configure(ROOT / "experiments/module_1_python")


def test_switching_provider_selects_url_key_and_model(environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_environment(
        environment,
        "MODEL_PROVIDER=litellm\nLITELLM_BASE_URL=https://proxy.example/\nLITELLM_API_KEY=litellm-key\n"
        "LITELLM_MODEL_NAME=proxy-model\nOPENROUTER_BASE_URL=https://openrouter.ai/api/v1/\n"
        "OPENROUTER_API_KEY=openrouter-key\nOPENROUTER_MODEL_NAME=vendor/router-model\n",
    )
    module = ROOT / "experiments/module_1_python"
    common.configure(module)
    assert os.environ["POC_MODEL_BASE_URL"] == "https://proxy.example/v1"
    assert os.environ["POC_MODEL_API_KEY"] == "litellm-key"
    assert os.environ["POC_MODEL_NAME"] == "proxy-model"
    monkeypatch.setenv("MODEL_PROVIDER", "openrouter")
    common.configure(module)
    assert os.environ["POC_MODEL_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert os.environ["POC_MODEL_API_KEY"] == "openrouter-key"
    assert os.environ["POC_MODEL_NAME"] == "vendor/router-model"
    assert os.environ["LITELLM_API_KEY"] == "litellm-key"


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("override", [None, "vendor/explicit-model", "vertex_ai/google/gemma-4-26b-a4b-it-maas"])
def test_every_cli_uses_provider_model_and_supports_override(
    environment: Path, monkeypatch: pytest.MonkeyPatch, module: str, override: str | None
) -> None:
    write_environment(
        environment,
        "MODEL_PROVIDER=openrouter\nOPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n"
        "OPENROUTER_API_KEY=router-key\nOPENROUTER_MODEL_NAME=vendor/router-model\n"
        "POLICY_MODEL_NAME=old-litellm-model\n",
    )
    cli_module = importlib.import_module(f"experiments.{module}.cli")
    # Every run is owned by its own pedagogical workshop.
    owner_module = module
    owner = importlib.import_module(f"experiments.{owner_module}.workshop")

    def stop_before_services(components: tuple[str, ...]) -> None:
        raise RuntimeError("services intercepted")

    monkeypatch.setattr(
        common if module == "module_3_openenv_bash" else owner, "prepare_runtime", stop_before_services
    )
    args = ["run", "--output-dir", str(environment.parent / "output")]
    if override:
        args += ["--model", override]
    result = CliRunner().invoke(cli_module.cli, args)
    assert isinstance(result.exception, RuntimeError), result.output
    assert str(result.exception) == "services intercepted"
    expected = "google/gemma-4-26b-a4b-it" if override and override.startswith("vertex_ai/") else override
    expected = expected or "vendor/router-model"
    assert os.environ["POC_MODEL_NAME"] == expected
    config = OmegaConf.load(ROOT / "experiments" / module / "config.yaml")
    provider = OmegaConf.to_container(config.policy_model.responses_api_models.inference_provider, resolve=True)
    assert provider["base_url"] == "https://openrouter.ai/api/v1"
    assert provider["api_key"] == "router-key"
    assert provider["model"] == expected


def test_experiment_model_takes_precedence(environment: Path) -> None:
    write_environment(
        environment,
        "MODEL_PROVIDER=openrouter\nOPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n"
        "OPENROUTER_API_KEY=router-key\nOPENROUTER_MODEL_NAME=vendor/shared\n"
        "OPENROUTER_MODEL_NAME_MODULE_4_JETBRAINS_MCP=vendor/inspection-model\n",
    )
    common.configure(ROOT / "experiments/module_4_jetbrains_mcp")
    assert os.environ["POC_MODEL_NAME"] == "vendor/inspection-model"


@pytest.mark.parametrize("provider", ["litellm", "openrouter"])
@pytest.mark.parametrize(
    "module,shared,expected",
    [
        *((module, "", "gemma") for module in MODULES),
        ("module_2_direct_bash", "POLICY_MODEL_NAME=google/gemma-4-26b-a4b-it\n", "gemma"),
        (
            "module_4_jetbrains_mcp",
            "POLICY_MODEL_NAME=openai/gpt-4.1-nano\n"
            "POLICY_MODEL_NAME_MODULE_4_JETBRAINS_MCP=vertex_ai/google/gemma-4-26b-a4b-it-maas\n",
            "gemma",
        ),
    ],
)
def test_shared_models_keep_identity_when_provider_changes(
    environment: Path, provider: str, module: str, shared: str, expected: str
) -> None:
    prefix = provider.upper()
    write_environment(
        environment,
        f"MODEL_PROVIDER={provider}\n{prefix}_BASE_URL=https://provider.example/v1\n{prefix}_API_KEY=key\n" + shared,
    )
    common.configure(ROOT / "experiments" / module)
    if expected == "gemma":
        expected = "vertex_ai/google/gemma-4-26b-a4b-it-maas" if provider == "litellm" else "google/gemma-4-26b-a4b-it"
    assert os.environ["POC_MODEL_NAME"] == expected


@pytest.mark.parametrize(
    "module,legacy,expected",
    [
        ("module_1_python", "", "gemma"),
        ("module_4_jetbrains_mcp", "", "gemma"),
        ("module_1_python", "POLICY_MODEL_NAME=legacy-model\n", "legacy-model"),
    ],
)
def test_existing_litellm_configuration_still_works(
    environment: Path, module: str, legacy: str, expected: str
) -> None:
    write_environment(environment, "LITELLM_BASE_URL=https://proxy.example\nLITELLM_API_KEY=key\n" + legacy)
    common.configure(ROOT / "experiments" / module)
    if expected == "gemma":
        expected = "vertex_ai/google/gemma-4-26b-a4b-it-maas"
    assert os.environ["POC_MODEL_NAME"] == expected


def test_inspection_cli_omits_reasoning_effort_by_default() -> None:
    from experiments.module_4_jetbrains_mcp.cli import cli

    command = cli.commands["run"]
    option = next(parameter for parameter in command.params if parameter.name == "reasoning_effort")
    assert option.get_default(click.Context(command)) == "omit"


@pytest.mark.parametrize(
    "settings,message",
    [
        ("MODEL_PROVIDER=other\n", "MODEL_PROVIDER must be litellm or openrouter"),
        (
            "MODEL_PROVIDER=openrouter\nLITELLM_BASE_URL=https://proxy.example\nLITELLM_API_KEY=key\n",
            "OPENROUTER_BASE_URL, OPENROUTER_API_KEY",
        ),
    ],
)
def test_invalid_or_incomplete_provider_fails(environment: Path, settings: str, message: str) -> None:
    write_environment(environment, settings)
    with pytest.raises(click.ClickException, match=message):
        common.configure(ROOT / "experiments/module_1_python")


@pytest.mark.parametrize("module", MODULES)
def test_experiment_images_are_fixed(environment: Path, monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    from experiments.images import BOOKWORM_IMAGE, OPENENV_IMAGE, PYCHARM_IMAGE

    write_environment(environment, "LITELLM_BASE_URL=https://proxy.example\nLITELLM_API_KEY=key\n")
    for name in ("IDEGYM_IMAGE", "BASH_COMPARISON_IMAGE", "OPENENV_BASH_IMAGE", "IDE_INSPECTIONS_IMAGE"):
        monkeypatch.setenv(name, "ignored/override:1")
    record = environment.parent / ".runtime" / module / "image.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"status": "SUCCESS", "runtime": "pycharm-mcp", "image": "ignored/cache:1"}))
    cli = importlib.import_module(f"experiments.{module}.cli").cli
    owner = importlib.import_module(f"experiments.{module}.workshop")

    def stop_before_services(components: tuple[str, ...]) -> None:
        raise RuntimeError("services intercepted")

    monkeypatch.setattr(
        common if module == "module_3_openenv_bash" else owner, "prepare_runtime", stop_before_services
    )
    output = environment.parent / "output"
    result = CliRunner().invoke(cli, ["run", "--output-dir", str(output)])
    assert isinstance(result.exception, RuntimeError), result.output
    assert str(result.exception) == "services intercepted"
    expected = {
        "module_1_python": BOOKWORM_IMAGE,
        "module_2_direct_bash": BOOKWORM_IMAGE,
        "module_3_openenv_bash": OPENENV_IMAGE,
        "module_4_jetbrains_mcp": PYCHARM_IMAGE,
    }[module]
    config = OmegaConf.to_container(OmegaConf.load(output / "config.yaml"), resolve=True)
    if module == "module_1_python":
        assert config["idegym_python"]["resources_servers"]["idegym_python"]["image"] == expected
    else:
        assert json.loads((output / "run-settings.json").read_text())["image"] == expected
        agent, settings = {
            "module_2_direct_bash": ("bash_agent", "shell"),
            "module_3_openenv_bash": ("bash_agent", "environment"),
            "module_4_jetbrains_mcp": ("inspection_agent", "ide"),
        }[module]
        component = "ide_agent" if module == "module_4_jetbrains_mcp" else "bash_agent"
        assert config[component]["responses_api_agents"][agent][settings]["image"] == expected
    help_result = CliRunner().invoke(cli, ["run", "--help"])
    assert help_result.exit_code == 0
    assert "--image" not in help_result.output
