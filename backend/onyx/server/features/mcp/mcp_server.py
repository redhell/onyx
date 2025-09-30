"""MCP Server implementation that exposes Onyx's internal search functionality.

This module implements a Model Context Protocol (MCP) server endpoint that:
1. Uses Onyx API token authentication
2. Implements the streamable HTTP protocol  
3. Exposes only the internal_search tool
4. Leverages existing SearchTool functionality
"""

import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict
from collections.abc import Callable

from fastapi import HTTPException, Request
from fastmcp import FastMCP
from fastmcp.server.auth.auth import AccessToken, TokenVerifier
from sqlalchemy.orm import Session

from onyx.auth.api_key import get_hashed_api_key_from_request, hash_api_key
from onyx.db.api_key import get_api_key_by_hash
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import User
from onyx.db.users import get_user_by_id
from onyx.tools.tool_implementations.search.search_tool import SearchTool, QUERY_FIELD
from onyx.tools.models import ToolResponse
from onyx.utils.logger import setup_logger

logger = setup_logger()


class OnyxApiKeyVerifier(TokenVerifier):
    """Token verifier that validates Onyx API keys."""
    
    async def verify_token(self, token: str) -> AccessToken | None:
        """Verify an Onyx API key and return access token."""
        try:
            # Hash the provided token to match against stored hashes
            hashed_token = hash_api_key(token)
            
            with get_session_with_current_tenant() as db_session:
                # Look up the API key in the database
                api_key_descriptor = get_api_key_by_hash(db_session, hashed_token)
                
                if not api_key_descriptor:
                    return None
                
                # Get the user associated with this API key
                user = get_user_by_id(db_session, api_key_descriptor.user_id)
                if not user:
                    return None
                
                # Return an access token with user information
                return AccessToken(
                    token=token,
                    client_id=str(user.id),
                    scopes=["mcp:use"],  # Standard MCP scope
                    expires_at=None,  # Onyx API keys don't expire by default
                    resource=None,
                    claims={
                        "user_id": str(user.id),
                        "email": user.email,
                        "role": api_key_descriptor.api_key_role.value,
                    },
                )
                
        except Exception as e:
            logger.warning(f"Failed to verify API key: {e}")
            return None


def create_search_tool_for_user(user_id: str, db_session: Session) -> SearchTool:
    """Create a SearchTool instance for a specific user."""
    from onyx.db.persona import get_default_persona
    from onyx.configs.model_configs import GEN_AI_MODEL_VERSION
    from onyx.llm.factory import get_default_llm
    from onyx.chat.models import PromptConfig, DocumentPruningConfig, AnswerStyleConfig
    from onyx.context.search.enums import LLMEvaluationType
    from onyx.context.search.models import RetrievalDetails
    
    # Get the user
    user = get_user_by_id(db_session, uuid.UUID(user_id))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get the default persona for the user
    persona = get_default_persona(user.id, db_session)
    if not persona:
        raise HTTPException(status_code=404, detail="No default persona found for user")
    
    # Create basic configurations
    llm = get_default_llm()
    fast_llm = get_default_llm()  # Using same LLM for both for simplicity
    
    prompt_config = PromptConfig()
    document_pruning_config = DocumentPruningConfig()
    answer_style_config = AnswerStyleConfig()
    retrieval_options = RetrievalDetails()
    
    # Create the search tool
    return SearchTool(
        tool_id=0,  # Using 0 for MCP searches
        db_session=db_session,
        user=user,
        persona=persona,
        retrieval_options=retrieval_options,
        prompt_config=prompt_config,
        llm=llm,
        fast_llm=fast_llm,
        document_pruning_config=document_pruning_config,
        answer_style_config=answer_style_config,
        evaluation_type=LLMEvaluationType.BASIC,
    )


def create_onyx_mcp_server() -> FastMCP:
    """Create and configure the Onyx MCP server."""
    
    # Create MCP server with authentication
    mcp = FastMCP(
        "Onyx MCP Server",
        auth=OnyxApiKeyVerifier()
    )
    
    @mcp.tool(
        name="internal_search",
        description=(
            "Search Onyx's internal knowledge base using semantic search. "
            "This tool searches through all connected documents and data sources "
            "that the authenticated user has access to. Use this tool to find "
            "relevant information, documents, or answers to questions based on "
            "the organization's knowledge base."
        ),
    )
    async def internal_search(query: str) -> Dict[str, Any]:
        """
        Perform semantic search on Onyx's knowledge base.
        
        Args:
            query: The search query string
            
        Returns:
            Dict containing search results with documents and metadata
        """
        # Get the authenticated user from the access token
        from fastmcp.server.auth.auth import get_access_token
        
        access_token = get_access_token()
        if not access_token:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        user_id = access_token.client_id
        logger.info(f"Performing internal search for user {user_id}: {query}")
        
        try:
            with get_session_with_current_tenant() as db_session:
                # Create search tool for the user
                search_tool = create_search_tool_for_user(user_id, db_session)
                
                # Execute the search
                tool_responses = list(search_tool.run(**{QUERY_FIELD: query}))
                
                # Extract the final results
                final_result = search_tool.final_result(*tool_responses)
                
                # Format the response
                return {
                    "query": query,
                    "search_results": final_result,
                    "num_results": len(final_result) if final_result else 0,
                    "user_id": user_id,
                }
                
        except Exception as e:
            logger.error(f"Search failed for user {user_id}: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Internal search failed: {str(e)}"
            )
    
    return mcp


# Global MCP server instance
_mcp_server: FastMCP | None = None


def get_mcp_server() -> FastMCP:
    """Get or create the global MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = create_onyx_mcp_server()
    return _mcp_server