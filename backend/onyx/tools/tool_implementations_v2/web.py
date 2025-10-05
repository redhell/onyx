from typing import List
from typing import Optional

from agents import function_tool
from agents import RunContextWrapper
from pydantic import BaseModel

from onyx.agents.agent_search.dr.models import IterationAnswer
from onyx.agents.agent_search.dr.models import IterationInstructions
from onyx.agents.agent_search.dr.sub_agents.web_search.providers import (
    get_default_provider,
)
from onyx.agents.agent_search.dr.sub_agents.web_search.providers import (
    WebSearchProvider,
)
from onyx.agents.agent_search.dr.sub_agents.web_search.utils import (
    dummy_inference_section_from_internet_content,
)
from onyx.chat.turn.models import ChatTurnContext
from onyx.configs.constants import DocumentSource
from onyx.server.query_and_chat.streaming_models import FetchToolStart
from onyx.server.query_and_chat.streaming_models import Packet
from onyx.server.query_and_chat.streaming_models import SavedSearchDoc
from onyx.server.query_and_chat.streaming_models import SearchToolDelta
from onyx.server.query_and_chat.streaming_models import SearchToolStart
from onyx.tools.tool_implementations_v2.tool_accounting import tool_accounting


class WebSearchResult(BaseModel):
    tag: str
    title: str
    link: str
    snippet: str
    author: Optional[str] = None
    published_date: Optional[str] = None


class WebSearchResponse(BaseModel):
    results: List[WebSearchResult]


class WebFetchResult(BaseModel):
    tag: str
    title: str
    link: str
    full_content: str
    published_date: Optional[str] = None


class WebFetchResponse(BaseModel):
    results: List[WebFetchResult]


def short_tag(link: str, i: int) -> str:
    return f"S{i+1}"


@tool_accounting
def _web_search_core(
    run_context: RunContextWrapper[ChatTurnContext],
    query: str,
    search_provider: WebSearchProvider,
) -> WebSearchResponse:
    # TODO: Find better way to track index that isn't so implicit
    # based on number of tool calls
    index = run_context.context.current_run_step
    run_context.context.run_dependencies.emitter.emit(
        Packet(
            ind=index,
            obj=SearchToolStart(
                type="internal_search_tool_start", is_internet_search=True
            ),
        )
    )
    run_context.context.run_dependencies.emitter.emit(
        Packet(
            ind=index,
            obj=SearchToolDelta(
                type="internal_search_tool_delta", queries=[query], documents=None
            ),
        )
    )
    run_context.context.iteration_instructions.append(
        IterationInstructions(
            iteration_nr=index,
            plan="plan",
            purpose="Searching the web for information",
            reasoning=f"I am now using Web Search to gather information on {query}",
        )
    )
    hits = search_provider.search(query)
    results = []
    for i, r in enumerate(hits):
        results.append(
            WebSearchResult(
                tag=short_tag(r.link, i),
                title=r.title,
                link=r.link,
                snippet=r.snippet,
                author=r.author,
                published_date=(
                    r.published_date.isoformat() if r.published_date else None
                ),
            )
        )
    run_context.context.aggregated_context.global_iteration_responses.append(
        IterationAnswer(
            tool="web_search",
            tool_id=18,
            iteration_nr=index,
            parallelization_nr=0,
            question=query,
            reasoning=f"I am now using Web Search to gather information on {query}",
            answer="Cool",
            cited_documents={},
            claims=["web_search"],
        )
    )
    return WebSearchResponse(results=results)


@function_tool
def web_search_tool(run_context: RunContextWrapper[ChatTurnContext], query: str) -> str:
    """
    Tool for searching the public internet. Useful for up to date information on PUBLIC knowledge.
    ---
    ## Decision boundary
    - You MUST call `web_search_tool` to discover sources when the request involves:
      - Fresh/unstable info (news, prices, laws, schedules, product specs, scores, exchange rates).
      - Recommendations, or any query where the specific sources matter.
      - Verifiable claims, quotes, or citations.
    - After ANY successful `web_search_tool` call that yields candidate URLs, you MUST call
      `web_fetch_tool` on the selected URLs BEFORE answering. Do NOT answer from snippets.

    ## When NOT to use
    - Casual chat, rewriting/summarizing user-provided text, or translation.
    - When the user already provided URLs (go straight to `web_fetch_tool`).

    ## Usage hints
    - Use ONE focused natural-language `query` per call.
    - Prefer 1–3 searches for distinct intents; then batch-fetch 3–8 best URLs.
    - Deduplicate domains/near-duplicates. Prefer recent, authoritative sources.

    ## Args
    - query (str): The search query.

    ## Returns (JSON string)
    {
      "results": [
        {
          "tag": "short_ref",
          "title": "...",
          "link": "https://...",
          "author": "...",
          "published_date": "2025-10-01T12:34:56Z"
          // intentionally NO full content
        }
      ]
    }
    """
    search_provider = get_default_provider()
    response = _web_search_core(run_context, query, search_provider)
    return response.model_dump_json()


@tool_accounting
def _web_fetch_core(
    run_context: RunContextWrapper[ChatTurnContext],
    urls: List[str],
    search_provider: WebSearchProvider,
) -> WebFetchResponse:
    # TODO: Find better way to track index that isn't so implicit
    # based on number of tool calls
    index = run_context.context.current_run_step

    # Create SavedSearchDoc objects from URLs for the FetchToolStart event
    saved_search_docs = [
        SavedSearchDoc(
            db_doc_id=0,
            document_id=url,
            chunk_ind=0,
            semantic_identifier=url,
            link=url,
            blurb="",  # Will be populated after fetching
            source_type=DocumentSource.WEB,
            boost=1,
            hidden=False,
            metadata={},
            score=0.0,
            is_relevant=None,
            relevance_explanation=None,
            match_highlights=[],
            updated_at=None,
            primary_owners=None,
            secondary_owners=None,
            is_internet=True,
        )
        for url in urls
    ]

    run_context.context.run_dependencies.emitter.emit(
        Packet(
            ind=index,
            obj=FetchToolStart(type="fetch_tool_start", documents=saved_search_docs),
        )
    )

    docs = search_provider.contents(urls)
    out = []
    for i, d in enumerate(docs):
        out.append(
            WebFetchResult(
                tag=short_tag(d.link, i),  # <-- add a tag
                title=d.title,
                link=d.link,
                full_content=d.full_content,
                published_date=(
                    d.published_date.isoformat() if d.published_date else None
                ),
            )
        )
    run_context.context.iteration_instructions.append(
        IterationInstructions(
            iteration_nr=index,
            plan="plan",
            purpose="Fetching content from URLs",
            reasoning=f"I am now using Web Fetch to gather information on {', '.join(urls)}",
        )
    )

    inference_sections = [
        dummy_inference_section_from_internet_content(d) for d in docs
    ]
    run_context.context.aggregated_context.global_iteration_responses.append(
        IterationAnswer(
            tool="web_fetch",
            tool_id=18,
            iteration_nr=index,
            parallelization_nr=0,
            question=f"Fetch content from URLs: {', '.join(urls)}",
            reasoning=f"I am now using Web Fetch to gather information on {', '.join(urls)}",
            answer="Cool",
            cited_documents={
                i: inference_section
                for i, inference_section in enumerate(inference_sections)
            },
            claims=["web_fetch"],
        )
    )

    return WebFetchResponse(results=out)


@function_tool
def web_fetch_tool(
    run_context: RunContextWrapper[ChatTurnContext], urls: List[str]
) -> str:
    """
    Tool for fetching and extracting full content from web pages.

    ---
    ## Decision boundary
    - You MUST use `web_fetch_tool` before quoting, citing, or relying on page content.
    - Use it whenever you already have URLs (from the user or from `web_search_tool`).
    - Do NOT answer questions based on search snippets alone.

    ## When NOT to use
    - If you do not yet have URLs (search first).
    - Avoid many tiny calls; batch URLs (1–20) in one request.

    ## Usage hints
    - Batch 3–8 high-quality, deduplicated URLs per topic.
    - Prefer primary, recent, and reputable sources.
    - If PDFs/long docs appear, still fetch; you may summarize sections explicitly.

    ## Args
    - urls (List[str]): Absolute URLs to retrieve.

    ## Returns (JSON string)
    {
      "results": [
        {
          "tag": "short_ref",
          "title": "...",
          "link": "https://...",
          "full_content": "...",
          "published_date": "2025-10-01T12:34:56Z"
        }
      ]
    }
    """
    search_provider = get_default_provider()
    response = _web_fetch_core(run_context, urls, search_provider)
    return response.model_dump_json()
