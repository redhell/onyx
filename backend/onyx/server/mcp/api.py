"""API router for the MCP server endpoint.

This module integrates the MCP server into the main FastAPI application,
making it accessible at the /mcp endpoint.
"""

import json
from typing import Any
from typing import Dict

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi import status
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.db.engine.sql_engine import get_session
from onyx.db.models import User
from onyx.server.mcp.server import internal_search_sync
from onyx.utils.logger import setup_logger

logger = setup_logger()

router = APIRouter(prefix="/mcp")


@router.post("/tools/list")
async def list_tools(
    request: Request,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """List available MCP tools."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Onyx API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Get the request JSON
        body = await request.json()
        request_id = body.get("id")

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "internal_search",
                        "description": (
                            "Search Onyx's internal knowledge base for information. "
                            "Use this tool to find relevant documents, answers, and "
                            "information from the connected data sources."
                        ),
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
    except Exception as e:
        logger.error(f"Error in list_tools: {str(e)}", exc_info=True)
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }


@router.post("/tools/call")
async def call_tool(
    request: Request,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """Execute a tool call."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Onyx API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Get the request JSON
        body = await request.json()
        request_id = body.get("id")
        params = body.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name != "internal_search":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}"
                }
            }

        question = arguments.get("question", "")
        if not question:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params: question is required"
                }
            }

        # Execute the search
        result = internal_search_sync(question, user, db_session)

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }
                ]
            }
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in call_tool: {str(e)}", exc_info=True)
        return {
            "jsonrpc": "2.0",
            "id": request_id if 'request_id' in locals() else None,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }


@router.post("/initialize")
async def initialize(
    request: Request,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """Initialize the MCP session."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Onyx API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Get the request JSON
        body = await request.json()
        request_id = body.get("id")

        return {
            "jsonrpc": "2.0",
            "id": request_id,
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
    except Exception as e:
        logger.error(f"Error in initialize: {str(e)}", exc_info=True)
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }


@router.options("/{path:path}")
async def options_handler(path: str) -> Response:
    """Handle CORS OPTIONS requests."""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
            "Access-Control-Max-Age": "86400",
        }
    )