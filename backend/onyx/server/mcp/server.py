"""MCP server implementation for Onyx.

This module provides an MCP (Model Context Protocol) server that exposes Onyx's 
internal search functionality through a standardized protocol. The server is 
accessible at [onyxdomain]/mcp and requires authentication via Onyx API tokens.

The server exposes a single tool: internal_search, which allows LLMs to search
Onyx's internal knowledge base using the same functionality as the search tool.
"""

from typing import Any
from typing import Dict

from fastapi import HTTPException
from fastapi import status
from fastmcp import FastMCP
from sqlalchemy.orm import Session

from onyx.db.models import User
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.context.search.models import RetrievalDetails
from onyx.chat.models import PromptConfig
from onyx.chat.models import DocumentPruningConfig
from onyx.chat.models import AnswerStyleConfig
from onyx.context.search.enums import LLMEvaluationType
from onyx.llm.factory import get_llms_for_persona
from onyx.db.persona import get_best_persona_id_for_user
from onyx.db.persona import get_persona_by_id
from onyx.utils.logger import setup_logger

logger = setup_logger()


def internal_search_sync(question: str, user: User, db_session: Session) -> Dict[str, Any]:
    """Execute an internal search synchronously and return results."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Onyx API token."
        )

    try:
        # Get best persona for the user (defaulting to None which gets the default)
        persona_id = get_best_persona_id_for_user(db_session, user, None)
        if not persona_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No accessible persona found for user"
            )

        persona = get_persona_by_id(persona_id, user, db_session)
        if not persona:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve persona"
            )

        # Get LLMs for the persona
        llm, fast_llm = get_llms_for_persona(persona=persona)

        # Create search tool with required parameters
        search_tool = SearchTool(
            tool_id=0,  # Using 0 for MCP internal tool
            db_session=db_session,
            user=user,
            persona=persona,
            retrieval_options=RetrievalDetails(
                run_search="always",
                real_time=True,
                enable_auto_detect_filters=False,
                filters={}
            ),
            prompt_config=PromptConfig(),
            llm=llm,
            fast_llm=fast_llm,
            document_pruning_config=DocumentPruningConfig(),
            answer_style_config=AnswerStyleConfig(),
            evaluation_type=LLMEvaluationType.BASIC,
            bypass_acl=False
        )

        # Execute the search
        tool_responses = list(search_tool.run(query=question))

        # Extract and format results
        final_result = search_tool.final_result(*tool_responses)
        
        # Get the first response which contains search summary
        search_summary = None
        for response in tool_responses:
            if hasattr(response.response, 'rephrased_query'):
                search_summary = response.response
                break

        return {
            "query": question,
            "rephrased_query": search_summary.rephrased_query if search_summary else question,
            "documents": final_result,
            "document_count": len(final_result)
        }

    except Exception as e:
        logger.error(f"Error in internal_search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


