import uuid
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from onyx.db.enums import MCPAuthenticationType
from onyx.db.enums import MCPTransport
from onyx.db.models import MCPServer
from onyx.tools.adapter_v1_to_v2 import tool_to_function_tool
from onyx.tools.models import DynamicSchemaInfo
from onyx.tools.tool_implementations.custom.custom_tool import (
    build_custom_tools_from_openapi_schema_and_headers,
)
from onyx.tools.tool_implementations.mcp.mcp_tool import MCPTool


@pytest.fixture
def openapi_schema() -> dict[str, Any]:
    """OpenAPI schema for testing."""
    return {
        "openapi": "3.0.0",
        "info": {
            "version": "1.0.0",
            "title": "Test API",
            "description": "A test API for adapter testing",
        },
        "servers": [
            {"url": "http://localhost:8080/CHAT_SESSION_ID/test/MESSAGE_ID"},
        ],
        "paths": {
            "/test/{test_id}": {
                "GET": {
                    "summary": "Get a test item",
                    "operationId": "getTestItem",
                    "parameters": [
                        {
                            "name": "test_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                },
                "POST": {
                    "summary": "Create a test item",
                    "operationId": "createTestItem",
                    "parameters": [
                        {
                            "name": "test_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    },
                },
            }
        },
    }


@pytest.fixture
def dynamic_schema_info() -> DynamicSchemaInfo:
    """Dynamic schema info for testing."""
    return DynamicSchemaInfo(chat_session_id=uuid.uuid4(), message_id=42)


@pytest.fixture
def mcp_server() -> MCPServer:
    """MCP server for testing."""
    return MCPServer(
        id=1,
        name="test_mcp_server",
        server_url="http://localhost:8080/mcp",
        auth_type=MCPAuthenticationType.NONE,
        transport=MCPTransport.STREAMABLE_HTTP,
    )


@pytest.fixture
def mcp_tool(mcp_server: MCPServer) -> MCPTool:
    """MCP tool for testing."""
    tool_definition = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query"}},
        "required": ["query"],
    }

    return MCPTool(
        tool_id=1,
        mcp_server=mcp_server,
        tool_name="search",
        tool_description="Search for information",
        tool_definition=tool_definition,
        connection_config=None,
        user_email="test@example.com",
    )


def test_tool_to_function_tool_post_method(
    openapi_schema: dict[str, Any], dynamic_schema_info: DynamicSchemaInfo
) -> None:
    """
    Test conversion of a POST method tool.
    Verifies that the adapter works with different HTTP methods.
    """
    tools = build_custom_tools_from_openapi_schema_and_headers(
        tool_id=-1,  # dummy tool id
        openapi_schema=openapi_schema,
        dynamic_schema_info=dynamic_schema_info,
    )

    # Get the second tool (POST method)
    v1_tool = tools[1]
    v2_tool = tool_to_function_tool(v1_tool)

    # Verify the conversion works for POST method
    assert v2_tool.name == v1_tool.name
    assert v2_tool.description == v1_tool.description
    assert (
        v2_tool.params_json_schema
        == v1_tool.tool_definition()["function"]["parameters"]
    )
    assert v2_tool.on_invoke_tool is not None


@patch("onyx.tools.tool_implementations.custom.custom_tool.requests.request")
def test_function_tool_invocation(
    mock_request: MagicMock,
    openapi_schema: dict[str, Any],
    dynamic_schema_info: DynamicSchemaInfo,
) -> None:
    """
    Test that the converted FunctionTool can be invoked correctly.
    Verifies that the on_invoke_tool method works as expected.
    """
    tools = build_custom_tools_from_openapi_schema_and_headers(
        tool_id=-1,  # dummy tool id
        openapi_schema=openapi_schema,
        dynamic_schema_info=dynamic_schema_info,
    )

    v1_tool = tools[0]
    v2_tool = tool_to_function_tool(v1_tool)

    # Mock the tool context (required by FunctionTool)
    mock_context = MagicMock()

    # Test the on_invoke_tool method
    # The FunctionTool expects a JSON string as input
    test_args = '{"test_id": "123"}'

    # This should call the original tool's run method
    # We need to handle the async nature of the FunctionTool
    import asyncio

    async def test_invoke():
        return await v2_tool.on_invoke_tool(mock_context, test_args)

    # Run the async function
    result = asyncio.run(test_invoke())

    # Verify the tool was called with the correct arguments
    expected_url = f"http://localhost:8080/{dynamic_schema_info.chat_session_id}/test/{dynamic_schema_info.message_id}/test/123"
    mock_request.assert_called_once_with("GET", expected_url, json=None, headers={})

    # The result should be a list of ToolResponse objects
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].id == "custom_tool_response"


def test_tool_to_function_tool_mcp_tool(mcp_tool: MCPTool) -> None:
    """
    Test conversion of an MCP tool to FunctionTool.
    Verifies that the adapter works with MCP tools.
    """
    v2_tool = tool_to_function_tool(mcp_tool)

    # Verify the conversion works for MCP tool
    assert v2_tool.name == mcp_tool.name
    assert v2_tool.description == mcp_tool.description
    assert (
        v2_tool.params_json_schema
        == mcp_tool.tool_definition()["function"]["parameters"]
    )
    assert v2_tool.on_invoke_tool is not None


@patch("onyx.tools.tool_implementations.mcp.mcp_tool.call_mcp_tool")
def test_mcp_tool_invocation(mock_call_mcp_tool: MagicMock, mcp_tool: MCPTool) -> None:
    """
    Test that the converted MCP FunctionTool can be invoked correctly.
    Verifies that the on_invoke_tool method works as expected for MCP tools.
    """
    # Mock the MCP tool call response
    mock_call_mcp_tool.return_value = "Search results: test query"

    v2_tool = tool_to_function_tool(mcp_tool)

    # Mock the tool context (required by FunctionTool)
    mock_context = MagicMock()

    # Test the on_invoke_tool method
    # The FunctionTool expects a JSON string as input
    test_args = '{"query": "test search"}'

    # This should call the original tool's run method
    # We need to handle the async nature of the FunctionTool
    import asyncio

    async def test_invoke():
        return await v2_tool.on_invoke_tool(mock_context, test_args)

    # Run the async function
    result = asyncio.run(test_invoke())

    # Verify the MCP tool was called with the correct arguments
    mock_call_mcp_tool.assert_called_once_with(
        mcp_tool.mcp_server.server_url,
        mcp_tool.name,
        {"query": "test search"},
        connection_headers={},
        transport=mcp_tool.mcp_server.transport,
    )

    # The result should be a list of ToolResponse objects
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].id == "custom_tool_response"
