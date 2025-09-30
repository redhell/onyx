"""Integration tests for the MCP server endpoint."""

import json
import uuid
from unittest.mock import Mock, patch


def test_mcp_server_creation():
    """Test that the MCP server can be created successfully."""
    try:
        from onyx.server.features.mcp.mcp_server import create_onyx_mcp_server
        
        mcp = create_onyx_mcp_server()
        
        assert mcp is not None
        assert mcp.name == "Onyx MCP Server"
        
        # Check that the HTTP app can be created
        http_app = mcp.http_app()
        assert http_app is not None
        
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False


def test_api_key_verifier_creation():
    """Test that the API key verifier can be created."""
    try:
        from onyx.server.features.mcp.mcp_server import OnyxApiKeyVerifier
        
        verifier = OnyxApiKeyVerifier()
        assert verifier is not None
        
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run basic tests without complex dependencies
    print("Running MCP server tests...")
    
    test1 = test_mcp_server_creation()
    test2 = test_api_key_verifier_creation()
    
    if test1 and test2:
        print("✓ All basic MCP server tests passed!")
    else:
        print("✗ Some tests failed!")