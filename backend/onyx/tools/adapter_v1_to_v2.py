# create adapter from Tool to FunctionTool
import json
from typing import Any

from agents import FunctionTool
from agents import RunContextWrapper

from onyx.tools.built_in_tools import BUILT_IN_TOOL_MAP_V2
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


def tools_to_function_tools(tools: list[Tool]) -> list[FunctionTool]:
    from onyx.tools.tool_implementations.mcp.mcp_tool import MCPTool
    from onyx.tools.tool_implementations.custom.custom_tool import CustomTool

    onyx_tools: list[list[FunctionTool]] = [
        BUILT_IN_TOOL_MAP_V2[type(tool).__name__]
        for tool in tools
        if type(tool).__name__ in BUILT_IN_TOOL_MAP_V2
    ]
    flattened_builtin_tools: list[FunctionTool] = [
        onyx_tool for sublist in onyx_tools for onyx_tool in sublist
    ]
    custom_and_mcp_tools: list[FunctionTool] = [
        tool_to_function_tool(tool)
        for tool in tools
        if isinstance(tool, CustomTool) or isinstance(tool, MCPTool)
    ]

    return flattened_builtin_tools + custom_and_mcp_tools
