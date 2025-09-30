# Onyx MCP Server

The Onyx MCP (Model Context Protocol) Server provides a standardized interface for LLMs to access Onyx's internal search functionality. The server is accessible at `[onyxdomain]/mcp` and requires authentication via Onyx API tokens.

## Features

- **Authentication**: Requires valid Onyx API token via `Authorization: Bearer <token>` header
- **Single Tool**: Exposes `internal_search` tool for searching Onyx's knowledge base
- **JSON-RPC 2.0**: Implements MCP protocol over HTTP with JSON-RPC 2.0
- **Error Handling**: Comprehensive error responses for invalid requests

## Endpoints

### `POST /mcp/initialize`
Initialize the MCP session.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {}
    },
    "serverInfo": {
      "name": "Onyx Internal MCP Server",
      "version": "1.0.0"
    }
  }
}
```

### `POST /mcp/tools/list`
List available tools.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "internal_search",
        "description": "Search Onyx's internal knowledge base for information...",
        "inputSchema": {
          "type": "object",
          "properties": {
            "question": {
              "type": "string",
              "description": "The search query or question to find relevant information"
            }
          },
          "required": ["question"]
        }
      }
    ]
  }
}
```

### `POST /mcp/tools/call`
Execute a tool.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "id": 1,
  "params": {
    "name": "internal_search",
    "arguments": {
      "question": "What is Onyx?"
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\n  \"query\": \"What is Onyx?\",\n  \"rephrased_query\": \"What is Onyx?\",\n  \"documents\": [...],\n  \"document_count\": 5\n}"
      }
    ]
  }
}
```

## Authentication

All requests must include a valid Onyx API token:

```bash
curl -X POST https://your-onyx-domain.com/api/mcp/tools/list \
  -H "Authorization: Bearer your-api-token" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'
```

## Error Responses

The server returns JSON-RPC 2.0 compliant error responses:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid params: question is required"
  }
}
```

Common error codes:
- `-32700`: Parse error (invalid JSON)
- `-32601`: Method not found
- `-32602`: Invalid params
- `-32603`: Internal error

## Integration

The MCP server integrates seamlessly with Onyx's existing infrastructure:

- Uses `current_user` dependency for authentication
- Leverages `SearchTool` for search functionality
- Respects user permissions and persona settings
- Follows existing error handling patterns

## Testing

Integration tests are available in `backend/tests/integration/tests/mcp/test_mcp_server.py` and cover:

- Authentication requirements
- Tool listing functionality
- Tool execution with valid/invalid parameters
- Error handling scenarios