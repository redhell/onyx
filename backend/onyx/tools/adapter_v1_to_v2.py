# create adapter from Tool to FunctionTool
import json
from typing import Any

from agents import FunctionTool
from agents import RunContextWrapper

from onyx.tools.tool import Tool


async def _tool_run_wrapper(
    tool: Tool, context: RunContextWrapper[Any], json_string: str
):
    """
    Wrapper function to adapt Tool.run() to FunctionTool.on_invoke_tool() signature.
    """
    # Parse the JSON string to get the arguments
    args = json.loads(json_string) if json_string else {}

    # Call the original tool's run method
    # The run method returns a generator, so we need to collect all results
    results = []
    for result in tool.run(**args):
        results.append(result)

    # Return the results (FunctionTool expects a string or something that can be converted to string)
    return results


def tool_to_function_tool(tool: Tool) -> FunctionTool:
    return FunctionTool(
        name=tool.name,
        description=tool.description,
        params_json_schema=tool.tool_definition()["function"]["parameters"],
        on_invoke_tool=lambda context, json_string: _tool_run_wrapper(
            tool, context, json_string
        ),
    )
