"""A bounded Gym agent whose model-visible tools all belong to the selected IDE MCP."""

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    accumulate_response_usage,
)
from pydantic import Field

from experiments.ide_mcp.transport import IdeSettings, ToolInfo, ToolResult


class IdeAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    ide: IdeSettings
    step_limit: int = Field(default=20, ge=1, le=100)


def model_tool_output(result: ToolResult) -> list[dict[str, Any]]:
    """Preserve screenshots as images instead of feeding base64 to a text model."""
    content = [{"type": "input_text", "text": json.dumps({"is_error": result.is_error})}]
    for block in result.content:
        if block.get("type") == "image":
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{block['mimeType']};base64,{block['data']}",
                    "detail": "auto",
                }
            )
        else:
            content.append({"type": "input_text", "text": str(block.get("text", json.dumps(block)))[:24000]})
    return content


class ToolConnection(Protocol):
    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


async def run_episode(
    params: NeMoGymResponseCreateParamsNonStreaming,
    tools: list[ToolInfo],
    connection: ToolConnection,
    model_call: Callable[[NeMoGymResponseCreateParamsNonStreaming], Awaitable[NeMoGymResponse]],
    step_limit: int,
    trace_path: Path,
) -> tuple[NeMoGymResponse, int, int]:
    from experiments.workshop_support import event

    # Keep the complete prompt, previous responses, and tool observations for each turn.
    inputs = (
        params.input
        if not isinstance(params.input, str)
        else [NeMoGymEasyInputMessage(role="user", content=params.input)]
    )
    outputs = []
    usage = None
    attempts = successes = 0
    allowed = {tool.name for tool in tools}
    with trace_path.open("w") as trace:
        for step in range(step_limit):
            payload = params.model_dump(exclude_none=True) | {
                "input": [*inputs, *outputs],
                "tools": [tool.function() for tool in tools],
                "parallel_tool_calls": False,
            }
            event(f"Model turn {step + 1}/{step_limit}: waiting for a response...")
            response = await model_call(NeMoGymResponseCreateParamsNonStreaming.model_validate(payload))
            usage = accumulate_response_usage(usage, response.usage)
            outputs.extend(response.output)
            trace.write(json.dumps({"step": step + 1, "response": response.model_dump(mode="json")}) + "\n")
            trace.flush()
            calls = [item for item in response.output if item.type == "function_call"]
            visible_reply = any(
                (getattr(part, "text", "") or getattr(part, "refusal", "")).strip()
                for item in response.output
                if item.type == "message"
                for part in item.content
            )
            if not calls and not visible_reply and not response.incomplete_details:
                response.status = "incomplete"
                event("Model returned no text or tool calls; episode incomplete. Grading the current files.")
                break
            if response.incomplete_details or not calls:
                event("Model stopped; handing the submission to the independent grader.")
                break
            for call in calls:
                attempts += 1
                try:
                    if call.name not in allowed:
                        raise ValueError("Tool is not exposed by the selected interface")
                    arguments = json.loads(call.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                except ValueError as exc:
                    result = ToolResult(is_error=True, content=[{"type": "text", "text": str(exc)}])
                else:
                    # Transport failures abort evaluation; IDE tool errors go back to the model.
                    event(f"Model turn {step + 1}: calling {call.name}...")
                    result = await connection.call(call.name, arguments)
                    event(f"Tool {call.name}: {'error returned to model' if result.is_error else 'completed'}.")
                    successes += int(not result.is_error)
                outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output", call_id=call.call_id, output=model_tool_output(result)
                    )
                )
                trace.write(json.dumps({"tool": call.name, "arguments": call.arguments, **result.model_dump()}) + "\n")
                trace.flush()
            if getattr(connection, "done", False):
                event("Environment ended the episode; handing its submission to the grader.")
                break
        else:
            response.status = "incomplete"
    response.output = outputs
    response.usage = usage
    return response, attempts, successes
