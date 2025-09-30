"""Integration tests for the MCP server endpoint.

Tests the MCP server implementation to ensure it properly authenticates requests
and executes the internal_search tool.
"""

import json
from typing import Any
from typing import Dict

import requests

from tests.integration.common_utils.constants import API_SERVER_URL
from tests.integration.common_utils.managers.api_key import APIKeyManager
from tests.integration.common_utils.managers.llm_provider import LLMProviderManager
from tests.integration.common_utils.test_models import DATestUser


class TestMCPServer:
    """Test suite for the MCP server endpoint."""

    def test_mcp_server_authentication_required(self) -> None:
        """Test that MCP server requires authentication."""
        url = f"{API_SERVER_URL}/mcp/tools/list"
        
        # Request without authentication should fail
        response = requests.post(url, json={"jsonrpc": "2.0", "method": "tools/list", "id": 1})
        assert response.status_code == 401
        assert "Authentication required" in response.json()["detail"]

    def test_mcp_server_tools_list(self, admin_user: DATestUser) -> None:
        """Test that MCP server returns available tools list."""
        # Create LLM provider
        LLMProviderManager.create(user_performing_action=admin_user)
        
        # Create API key for authentication
        api_key = APIKeyManager.create(user_performing_action=admin_user)
        
        url = f"{API_SERVER_URL}/mcp/tools/list"
        headers = {"Authorization": f"Bearer {api_key.api_key}"}
        
        response = requests.post(
            url, 
            json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate JSON-RPC response structure
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert "result" in data
        assert "tools" in data["result"]
        
        # Validate internal_search tool is present
        tools = data["result"]["tools"]
        assert len(tools) == 1
        
        tool = tools[0]
        assert tool["name"] == "internal_search"
        assert "Search Onyx's internal knowledge base" in tool["description"]
        assert "inputSchema" in tool
        
        # Validate tool schema
        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "question" in schema["properties"]
        assert schema["required"] == ["question"]

    def test_mcp_server_initialize(self, admin_user: DATestUser) -> None:
        """Test MCP server initialization."""
        # Create API key for authentication
        api_key = APIKeyManager.create(user_performing_action=admin_user)
        
        url = f"{API_SERVER_URL}/mcp/initialize"
        headers = {"Authorization": f"Bearer {api_key.api_key}"}
        
        response = requests.post(
            url,
            json={"jsonrpc": "2.0", "method": "initialize", "id": 1},
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate JSON-RPC response structure
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert "result" in data
        
        result = data["result"]
        assert "protocolVersion" in result
        assert "capabilities" in result
        assert "serverInfo" in result
        
        # Validate server info
        server_info = result["serverInfo"]
        assert server_info["name"] == "Onyx Internal MCP Server"
        assert "version" in server_info

    def test_mcp_server_tool_call_missing_question(self, admin_user: DATestUser) -> None:
        """Test tool call with missing question parameter."""
        # Create API key for authentication
        api_key = APIKeyManager.create(user_performing_action=admin_user)
        
        url = f"{API_SERVER_URL}/mcp/tools/call"
        headers = {"Authorization": f"Bearer {api_key.api_key}"}
        
        response = requests.post(
            url,
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "id": 1,
                "params": {
                    "name": "internal_search",
                    "arguments": {}  # Missing question
                }
            },
            headers=headers
        )
        
        assert response.status_code == 200  # Should return 200 with error in JSON-RPC
        data = response.json()
        
        # Validate error response
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert "error" in data
        assert data["error"]["code"] == -32602  # Invalid params
        assert "question is required" in data["error"]["message"]

    def test_mcp_server_tool_call_invalid_tool(self, admin_user: DATestUser) -> None:
        """Test tool call with invalid tool name."""
        # Create API key for authentication
        api_key = APIKeyManager.create(user_performing_action=admin_user)
        
        url = f"{API_SERVER_URL}/mcp/tools/call"
        headers = {"Authorization": f"Bearer {api_key.api_key}"}
        
        response = requests.post(
            url,
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "id": 1,
                "params": {
                    "name": "invalid_tool",
                    "arguments": {"question": "test"}
                }
            },
            headers=headers
        )
        
        assert response.status_code == 200  # Should return 200 with error in JSON-RPC
        data = response.json()
        
        # Validate error response
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert "error" in data
        assert data["error"]["code"] == -32601  # Method not found
        assert "Unknown tool: invalid_tool" in data["error"]["message"]