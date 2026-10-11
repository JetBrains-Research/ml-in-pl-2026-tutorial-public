import asyncio
import json
import logging
import os
import shlex

import structlog
from dotenv import load_dotenv
from idegym.client.client import IdeGYMClient, ServerCloseAction
from openai import AsyncOpenAI

load_dotenv()
# The IdeGYM client logs through structlog, which prints every level unless configured.
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

PROJECT_PATH = "/home/appuser/work"
PROMPT = (
    f"Explore the Python project in {PROJECT_PATH}: what does it do, how is it structured, and do its "
    "tests pass? Do not modify any files. End with a short summary."
)

# The sandbox shell's PATH does not include the image's Python and pytest; this puts them first.
SHELL = f"cd {PROJECT_PATH} && PATH=/opt/idegym/.venv/bin:/usr/local/bin:/usr/bin:/bin /bin/bash -c "

SYSTEM = (
    f"You are working on the Python project at {PROJECT_PATH} in a remote Linux sandbox. Use the bash tool "
    "to inspect and test the project. Python and pytest are on PATH. Each call starts a fresh shell in the "
    "project directory."
)
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash script in the sandbox and return its exit code, stdout, and stderr.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


async def main() -> None:
    # OpenRouter speaks the OpenAI API, so the standard client works with a different base URL.
    llm = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4.1-nano")

    async with IdeGYMClient(
        orchestrator_url=os.environ["IDEGYM_ORCHESTRATOR_URL"],
        name=os.environ.get("IDEGYM_CLIENT_NAME", "ml-in-pl"),
        namespace=os.environ.get("IDEGYM_NAMESPACE", "idegym"),
    ) as client:
        async with client.with_server(
            image_tag=os.environ["BASH_IMAGE"],
            server_name="bash-agent",
            close_action=ServerCloseAction.STOP,
        ) as server:
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT}]
            for _ in range(20):
                response = await llm.chat.completions.create(model=model, messages=messages, tools=[BASH_TOOL])
                message = response.choices[0].message
                messages.append(message.model_dump(exclude_none=True))
                if not message.tool_calls:
                    break
                for call in message.tool_calls:
                    command = json.loads(call.function.arguments)["command"]
                    print(f"$ {command}")
                    result = await server.execute_bash(
                        script=SHELL + shlex.quote(command), command_timeout=60, max_output_bytes=24000
                    )
                    output = f"exit_code={result.exit_code}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
            print(message.content)


if __name__ == "__main__":
    asyncio.run(main())
