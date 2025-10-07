import json
import re
from collections.abc import Generator
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import urlparse
from urllib.parse import urlunparse
from uuid import UUID

from slack_sdk import WebClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource
from onyx.configs.constants import FederatedConnectorSource
from onyx.connectors.factory import identify_connector_class
from onyx.connectors.highspot.utils import scrape_url_content
from onyx.connectors.models import InputType
from onyx.context.search.enums import LLMEvaluationType
from onyx.context.search.enums import SearchType
from onyx.context.search.models import IndexFilters
from onyx.context.search.models import SearchQuery
from onyx.db.federated import get_federated_connector_oauth_token
from onyx.db.models import Document
from onyx.db.models import FederatedConnector
from onyx.federated_connectors.federated_retrieval import (
    get_federated_retrieval_functions,
)
from onyx.llm.interfaces import LLM
from onyx.llm.models import PreviousMessage
from onyx.tools.base_tool import BaseTool
from onyx.tools.models import DocumentResult
from onyx.tools.models import DocumentRetrievalType
from onyx.tools.models import ToolResponse
from onyx.tools.utils import get_full_document_by_id
from onyx.tools.utils import process_chunks_to_document_result
from onyx.utils.logger import setup_logger

logger = setup_logger()

URL_FIELD = "url"
DOCUMENT_RESULT_ID = "document_result"


class FetchUrlTool(BaseTool):
    _NAME = "fetch_url"
    _DESCRIPTION = (
        "Fetch content from a specific URL. ALWAYS use this tool when the user provides a URL "
        "in their message. Examples: 'What's on https://reddit.com/r/news?' → use this tool with "
        "url='https://reddit.com/r/news'. "
        "'Check www.example.com' → use this tool with url='https://www.example.com'. "
        "'Show me http://docs.google.com/file' → use this tool with url='http://docs.google.com/file'. "
        "Extract the URL from the user's message and fetch its content. Works with web pages, "
        "Google Drive links, Slack threads, Notion pages, Reddit posts, and any other URL."
    )
    _DISPLAY_NAME = "Fetch URL"

    def __init__(self, tool_id: int, db_session: Session, user_id: UUID | None = None):
        self.tool_id = tool_id
        self.db_session = db_session
        self.user_id = user_id

    @property
    def id(self) -> int:
        return self.tool_id

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def description(self) -> str:
        return self._DESCRIPTION

    @property
    def display_name(self) -> str:
        return self._DISPLAY_NAME

    @classmethod
    def is_available(cls, db_session: Session) -> bool:
        """Always available"""
        return True

    def tool_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        URL_FIELD: {
                            "type": "string",
                            "description": "The URL to fetch content from",
                        },
                    },
                    "required": [URL_FIELD],
                },
            },
        }

    def get_args_for_non_tool_calling_llm(
        self,
        query: str,
        history: list[PreviousMessage],
        llm: LLM,
        force_run: bool = False,
    ) -> dict[str, Any] | None:
        """Extract URL from query for non-tool-calling LLMs"""
        # Extract URL from query
        extracted_url = self._extract_url_from_query(query)
        if not extracted_url:
            return None

        return {URL_FIELD: extracted_url}

    def build_tool_message_content(
        self, *args: ToolResponse
    ) -> str | list[str | dict[str, Any]]:
        document_result_response = next(
            response for response in args if response.id == DOCUMENT_RESULT_ID
        )
        document_result = document_result_response.response

        if isinstance(document_result, DocumentResult):
            return json.dumps(
                {
                    "title": document_result.title,
                    "content": document_result.content,
                    "source": document_result.source,
                    "url": document_result.url,
                    "confidence": document_result.confidence,
                }
            )
        return json.dumps({"error": "No document found"})

    def run(
        self, override_kwargs: None = None, **llm_kwargs: Any
    ) -> Generator[ToolResponse, None, None]:
        url = llm_kwargs.get(URL_FIELD, "")
        if not url:
            raise ValueError("URL is required")

        logger.info(f"Fetching URL: {url}")

        # Parse URL to determine source type and access method
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        # Determine the source type and access method
        source_type, access_method = self._get_source_type_and_method(domain)
        logger.info(
            f"Domain '{domain}' -> source_type: {source_type}, access_method: {access_method}"
        )

        # Route to appropriate fetch method based on access method
        if access_method == "federated" and source_type:
            logger.info(f"Routing to federated source: {url}")
            result = self._fetch_from_federated_source(url, source_type)
        elif access_method == "indexed" and source_type:
            logger.info(f"Routing to indexed source: {url}")
            result = self._fetch_from_indexed_source(url, source_type)
        else:
            logger.info(f"Routing to web scraping: {url}")
            result = self._fetch_from_web(url)

        yield ToolResponse(id=DOCUMENT_RESULT_ID, response=result)

    def final_result(self, *args: ToolResponse) -> dict[str, Any]:
        document_result_response = next(
            arg for arg in args if arg.id == DOCUMENT_RESULT_ID
        )
        document_result = document_result_response.response

        if isinstance(document_result, DocumentResult):
            return {
                "title": document_result.title,
                "content": document_result.content,
                "source": document_result.source,
                "url": document_result.url,
                "confidence": document_result.confidence,
                "metadata": document_result.metadata,
            }

        return {}

    def _get_universal_ignore_params(self) -> set[str]:
        """Universal query parameters that should be ignored for all sources

        These parameters don't affect content and are commonly added by:
        - Analytics/tracking (UTM parameters)
        - Social media sharing
        - Browser navigation
        - UI state (tabs, views, etc.)
        """
        return {
            # Analytics & Tracking
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "utm_term",
            "utm_content",
            "fbclid",
            "gclid",
            "msclkid",
            "twclid",
            "li_fat_id",
            # Social Media Sharing
            "ref",
            "referrer",
            "source",
            "campaign",
            # UI State & Navigation
            "tab",
            "view",
            "mode",
            "display",
            "layout",
            "format",
            "pli",
            "usp",
            "authuser",  # Google-specific UI params
            # Browser & Client
            "dl",
            "raw",
            "download",
            "preview",
            "embed",
            "web",
            "mobile",
            "app",
            # Pagination & Sorting (usually don't affect content)
            "page",
            "sort",
            "order",
            "limit",
            "offset",
            # Common platform-specific UI params
            "atlOrigin",  # Atlassian
            "e",  # SharePoint
            "p",
            "v",
            "r",  # Notion
        }

    def _get_domain_patterns_for_source(self, source_type: DocumentSource) -> list[str]:
        """Auto-generate domain patterns from DocumentSource enum name

        Converts DocumentSource enum values to likely domain patterns:
        - GOOGLE_DRIVE -> ['google.com', 'docs.google.com', 'drive.google.com']
        - LINEAR -> ['linear.app']
        - GITHUB -> ['github.com']
        - etc.
        """
        source_name = source_type.value.lower()

        # Special cases that don't follow the simple pattern
        special_cases = {
            "google_drive": [
                "docs.google.com",
                "drive.google.com",
                "sheets.google.com",
                "slides.google.com",
                "forms.google.com",
            ],
            "google_sites": ["sites.google.com"],
            "linear": ["linear.app"],
            "notion": ["notion.so", "notion.site"],
            "confluence": ["confluence.atlassian.com"],
            "jira": ["jira.atlassian.com"],
            "sharepoint": ["sharepoint.com"],
            "dropbox": ["dropbox.com"],
            "zendesk": ["zendesk.com"],
            "github": ["github.com"],
            "gitlab": ["gitlab.com"],
            "gmail": ["gmail.com"],
            "slack": ["slack.com"],
            "web": ["http", "https"],  # Special case for web scraping
        }

        if source_name in special_cases:
            return special_cases[source_name]

        # Default pattern: convert underscore to dot and add .com
        # LINEAR -> linear.com, ASANA -> asana.com, etc.
        domain = source_name.replace("_", ".")
        return [f"{domain}.com"]

    def _get_auto_extract_patterns(
        self, source_type: DocumentSource, url: str
    ) -> list[str]:
        """Auto-generate extract patterns based on common URL structures

        Returns a list of regex patterns to extract IDs for LIKE matching
        """
        patterns = []

        # Common patterns that work across many platforms
        common_patterns = [
            # Extract IDs after common path segments
            r"/d/([a-zA-Z0-9_-]+)",  # /d/ID pattern (Google Docs, etc.)
            r"/file/d/([a-zA-Z0-9_-]+)",  # /file/d/ID pattern (Google Drive)
            r"/document/([a-zA-Z0-9_-]+)",  # /document/ID pattern
            r"/page/([a-zA-Z0-9_-]+)",  # /page/ID pattern
            r"/item/([a-zA-Z0-9_-]+)",  # /item/ID pattern
            # Extract IDs from query parameters
            r"[?&]id=([a-zA-Z0-9_-]+)",  # ?id=ID or &id=ID
            r"[?&]pageId=(\d+)",  # ?pageId=123
            r"[?&]documentId=([a-zA-Z0-9_-]+)",  # ?documentId=ID
            # Extract IDs from path segments (generic)
            r"/([a-zA-Z0-9_-]{8,})",  # Long alphanumeric IDs (8+ chars)
            r"/([a-f0-9]{32})",  # 32-char hex IDs (Notion style)
            r"/([A-Z]+-\d+)",  # Ticket-style IDs (JIRA, Linear)
            # Extract file paths (for GitHub, GitLab, etc.)
            r"/blob/[^/]+/(.+)",  # GitHub blob paths
            r"/tree/[^/]+/(.+)",  # GitHub tree paths
        ]

        # Test each pattern against the URL
        for pattern in common_patterns:
            if re.search(pattern, url):
                patterns.append(pattern)

        return patterns

    def _normalize_url_for_indexed_search(
        self, url: str, source_type: DocumentSource
    ) -> list[str]:
        """Generate normalized URL variations for indexed source lookup using configuration-driven approach

        Returns a list of URL variations to try, ordered by preference:
        1. Original URL (exact match)
        2. Base URL without query parameters
        3. URL with filtered query parameters
        4. Extracted ID patterns for LIKE matching
        """

        variations = [url]  # Start with original URL

        # Parse the URL
        parsed = urlparse(url)

        # Add base URL without query parameters (universal fallback)
        base_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
        if base_url != url:
            variations.append(base_url)

        # Apply universal query parameter filtering
        universal_ignore_params = self._get_universal_ignore_params()

        if universal_ignore_params and parsed.query:
            query_params = parse_qs(parsed.query)
            filtered_params = {
                k: v
                for k, v in query_params.items()
                if k not in universal_ignore_params
            }
            if filtered_params != query_params:
                filtered_query = "&".join(
                    [f"{k}={v[0]}" for k, v in filtered_params.items()]
                )
                filtered_url = urlunparse(
                    (
                        parsed.scheme,
                        parsed.netloc,
                        parsed.path,
                        parsed.params,
                        filtered_query,
                        parsed.fragment,
                    )
                )
                variations.append(filtered_url)

        # Apply ID extraction patterns for LIKE matching
        # Handle special cases first (sources with very specific requirements)
        extract_patterns = []
        if source_type == DocumentSource.CONFLUENCE:
            extract_patterns = [r"/pages/viewpage\.action\?pageId=(\d+)"]
        elif source_type == DocumentSource.JIRA:
            extract_patterns = [r"/browse/([A-Z]+-\d+)"]
        else:
            # Use auto-generated patterns for all other sources
            extract_patterns = self._get_auto_extract_patterns(source_type, url)

        for pattern in extract_patterns:
            match = re.search(pattern, url)
            if match:
                extracted_id = match.group(1)
                # Add LIKE search pattern
                variations.append(f"%{extracted_id}%")

        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for variation in variations:
            if variation not in seen:
                seen.add(variation)
                unique_variations.append(variation)

        return unique_variations

    def _get_source_type_and_method(
        self, domain: str
    ) -> tuple[DocumentSource | None, str]:
        """Get the DocumentSource type and access method for a given domain

        Returns:
            tuple: (DocumentSource, access_method) where access_method is:
            - 'federated': Real-time API access (Slack, etc.)
            - 'indexed': Content is indexed and searchable (Google Drive, etc.)
            - 'web': Regular web scraping
        """
        logger.info(f"Checking domain '{domain}' for source type and method")

        # First check federated sources (fewer, so faster)
        for federated_source in FederatedConnectorSource:
            non_federated_source = federated_source.to_non_federated_source()
            logger.info(
                f"Checking federated source: {federated_source} -> {non_federated_source}"
            )
            if non_federated_source:
                try:
                    # For federated sources, use POLL input type (most common for real-time access)
                    identify_connector_class(non_federated_source, InputType.POLL)
                    domain_patterns = self._get_domain_patterns_for_source(
                        non_federated_source
                    )
                    logger.info(
                        f"Domain patterns for {non_federated_source}: {domain_patterns}"
                    )
                    for pattern in domain_patterns:
                        logger.info(
                            f"Checking pattern '{pattern}' against domain '{domain}'"
                        )
                        if (
                            domain.endswith(pattern)
                            or pattern in domain
                            or domain.endswith(f".{pattern}")
                        ):
                            logger.info(
                                f"MATCH! Domain '{domain}' matches pattern '{pattern}' -> federated"
                            )
                            return non_federated_source, "federated"
                except Exception as e:
                    logger.info(f"Skipping {non_federated_source} due to error: {e}")
                    continue  # Skip sources without connectors

        # Then check indexed sources (skip federated ones to avoid duplicates)
        federated_document_sources = {
            federated_source.to_non_federated_source()
            for federated_source in FederatedConnectorSource
            if federated_source.to_non_federated_source()
        }

        for source in DocumentSource:
            if source in federated_document_sources:
                continue  # Skip federated sources (already checked above)

            try:
                identify_connector_class(source)
                domain_patterns = self._get_domain_patterns_for_source(source)
                for pattern in domain_patterns:
                    if (
                        domain.endswith(pattern)
                        or pattern in domain
                        or domain.endswith(f".{pattern}")
                    ):
                        return source, "indexed"
            except Exception:
                continue  # Skip sources without connectors

        # Default to web scraping for unknown domains
        return None, "web"

    def _parse_slack_url(self, url: str) -> tuple[str, str] | None:
        """Parse Slack URL to extract channel_id and message_ts

        URL format: https://workspace.slack.com/archives/CHANNEL_ID/pMESSAGE_TS
        Example: https://onyx-company.slack.com/archives/C0771QKDBPE/p1759254460309479
        Returns: (channel_id, message_ts) or None if invalid
        """
        try:
            parsed = urlparse(url)
            path_parts = parsed.path.split("/")

            # Expected: ['', 'archives', 'CHANNEL_ID', 'pMESSAGE_TS']
            if len(path_parts) >= 4 and path_parts[1] == "archives":
                channel_id = path_parts[2]
                message_ts_with_p = path_parts[3].split("?")[0]  # Remove query params

                if message_ts_with_p.startswith("p"):
                    # Convert p1759254460309479 to 1759254460.309479
                    message_ts_str = message_ts_with_p[1:]  # Remove 'p'
                    # Insert dot before last 6 digits
                    if len(message_ts_str) > 6:
                        message_ts = f"{message_ts_str[:-6]}.{message_ts_str[-6:]}"
                        return channel_id, message_ts

            return None
        except Exception as e:
            logger.warning(f"Failed to parse Slack URL {url}: {e}")
            return None

    def _fetch_from_federated_source(
        self, url: str, source_type: DocumentSource
    ) -> DocumentResult:
        try:
            # Slack direct fetch path
            if source_type == DocumentSource.SLACK:
                parsed = self._parse_slack_url(url)
                if parsed:
                    channel_id, message_ts = parsed
                    logger.info(
                        f"Parsed Slack URL: channel={channel_id}, ts={message_ts}"
                    )

                    # Get Slack OAuth token
                    token = None
                    try:
                        fc_id_stmt = select(FederatedConnector.id).where(
                            FederatedConnector.source
                            == FederatedConnectorSource.FEDERATED_SLACK
                        )
                        fc_id = self.db_session.scalar(fc_id_stmt)

                        if fc_id and self.user_id:
                            oauth = get_federated_connector_oauth_token(
                                db_session=self.db_session,
                                federated_connector_id=fc_id,
                                user_id=self.user_id,
                            )
                            token = oauth.token if oauth and oauth.token else None

                    except Exception as e:
                        logger.info(f"Slack token lookup failed: {e}")

                    if token:
                        logger.info(
                            f"Fetching Slack message: channel={channel_id}, ts={message_ts}"
                        )
                        try:
                            client = WebClient(token=token)
                            resp = client.conversations_history(
                                channel=channel_id,
                                latest=message_ts,
                                inclusive=True,
                                limit=1,
                            )
                            resp.validate()

                            messages: list[dict] = resp.get("messages", [])
                            if messages:
                                message = messages[0]
                                text = message.get("text", "")

                                # Extract from blocks if no plain text
                                if not text:
                                    blocks = message.get("blocks", [])
                                    text = next(
                                        (
                                            block.get("text").get("text")
                                            for block in blocks
                                            if block.get("text", "").get("text", "")
                                        ),
                                        "",
                                    )

                                content = text or url
                                return DocumentResult(
                                    title=f"Slack message in #{channel_id}",
                                    content=content,
                                    source=DocumentRetrievalType.FEDERATED,
                                    url=url,
                                    metadata={
                                        "channel_id": channel_id,
                                        "message_ts": message_ts,
                                    },
                                    confidence=95,
                                )
                            else:
                                logger.info(
                                    "No messages found for given channel/timestamp"
                                )

                        except Exception as e:
                            logger.info(f"Slack API call failed: {e}")
                    else:
                        logger.info(
                            "No Slack token available; using federated fallback"
                        )

            # Fallback: existing federated retrieval
            federated_retrieval_infos = get_federated_retrieval_functions(
                db_session=self.db_session,
                user_id=self.user_id,
                source_types=[source_type],  # Only fetch for the specific source type
                document_set_names=[],  # No specific document sets
            )
            if federated_retrieval_infos:
                # Build a human-meaningful search query from the URL
                human_query = self._build_query_from_url(url)
                # Use first available (typically Slack)
                search_query = self._create_minimal_search_query(human_query)
                chunks = federated_retrieval_infos[0].retrieval_function(search_query)
                content = (
                    "\n\n".join([chunk.content for chunk in chunks if chunk.content])
                    if chunks
                    else url
                )
                return DocumentResult(
                    title="Content from Federated Source",
                    content=content,
                    source=DocumentRetrievalType.FEDERATED,
                    url=url,
                    metadata={"chunk_count": str(len(chunks))},
                    confidence=95,
                )

            return self._fetch_from_web(url)

        except Exception as e:
            logger.warning(f"Failed to fetch from federated sources: {e}")
            return self._fetch_from_web(url)

    def _fetch_from_indexed_source(
        self, url: str, source_type: DocumentSource
    ) -> DocumentResult:
        """Fetch content from indexed sources by finding document directly by URL"""
        logger.info(f"Fetching from indexed source: {url} (source: {source_type})")

        try:
            # Generate URL variations for more robust document lookup
            url_variations = self._normalize_url_for_indexed_search(url, source_type)
            logger.info(
                f"Trying {len(url_variations)} URL variations: {url_variations}"
            )

            document = None
            for i, url_variation in enumerate(url_variations):
                if i == 0:
                    # First try: exact match
                    stmt = select(Document).where(Document.link == url_variation)
                    document = self.db_session.scalar(stmt)
                    if document:
                        logger.info(
                            f"Found document with exact URL match: {document.semantic_id}"
                        )
                        break
                else:
                    # Subsequent tries: LIKE match for patterns (e.g., Google Docs ID)
                    if url_variation.startswith("%") and url_variation.endswith("%"):
                        stmt = select(Document).where(Document.link.like(url_variation))
                        document = self.db_session.scalar(stmt)
                        if document:
                            logger.info(
                                f"Found document with LIKE pattern match: {document.semantic_id}"
                            )
                            break
                    else:
                        # Try exact match for other variations
                        stmt = select(Document).where(Document.link == url_variation)
                        document = self.db_session.scalar(stmt)
                        if document:
                            logger.info(
                                f"Found document with normalized URL match: {document.semantic_id}"
                            )
                            break

            if document:
                logger.info(f"Found indexed document: {document.semantic_id}")

                # Use the generic method to get full document content
                return get_full_document_by_id(
                    document_id=document.id,
                    url=url,
                    source_type=source_type,
                    source_method="indexed",
                    db_session=self.db_session,
                    user=None,  # No user context needed for URL-based access
                    use_access_filters=False,  # Skip access filters for direct URL access
                )

            # If no document found by URL, fall back to web scraping
            logger.info(
                f"No indexed document found for {url}, falling back to web scraping"
            )
            return self._fetch_from_web(url)

        except Exception as e:
            logger.warning(f"Failed to fetch from indexed source: {e}")
            return self._fetch_from_web(url)

    def _fetch_from_web(self, url: str) -> DocumentResult:
        """Fetch content from external web URLs using scrape_url_content"""
        logger.info(f"Fetching from web: {url}")

        try:
            # Use the real scrape_url_content function
            content = scrape_url_content(url)

            if content is None:
                return DocumentResult(
                    title="Failed to fetch content",
                    content=f"Could not scrape content from {url}. The page might be inaccessible or require authentication.",
                    source=DocumentRetrievalType.INTERNAL,
                    url=url,
                    metadata={"error": "scraping_failed", "source_type": "web"},
                    confidence=0,
                )

            # Extract title from content (first line or first 100 chars)
            title = self._extract_title_from_content(content, url)

            return DocumentResult(
                title=title,
                content=content,
                source=DocumentRetrievalType.EXTERNAL,
                url=url,
                metadata={
                    "fetched_at": "2025-01-01T00:00:00Z",
                    "source_type": "web",
                    "content_length": str(len(content)),
                },
                confidence=95,
            )

        except Exception as e:
            logger.error(f"Failed to fetch web content from {url}: {e}")
            return DocumentResult(
                title="Error fetching content",
                content=f"Failed to fetch content from {url}: {str(e)}",
                source=DocumentRetrievalType.INTERNAL,
                url=url,
                metadata={"error": str(e)},
                confidence=0,
            )

    def _extract_title_from_content(self, content: str, url: str) -> str:
        """Extract a title from the scraped content"""
        if not content:
            return f"Web Document from {url}"

        # Try to find the first line that looks like a title
        lines = content.strip().split("\n")
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line) < 200:  # Reasonable title length
                return line

        # Fallback: use first 100 characters
        return content[:100].strip() + "..." if len(content) > 100 else content.strip()

    def _create_minimal_search_query(self, query: str) -> SearchQuery:
        """Create a minimal but valid SearchQuery for federated retrieval"""

        return SearchQuery(
            query=query,
            processed_keywords=[query],  # Use the query as keywords
            search_type=SearchType.SEMANTIC,
            evaluation_type=LLMEvaluationType.BASIC,
            filters=IndexFilters(
                access_control_list=[],
                user_file_ids=[],
                source_type=None,
                document_set=None,
                time_cutoff=None,
                tags=None,
                tenant_id=None,
                kg_entities=None,
                kg_relationships=None,
                kg_terms=None,
                kg_sources=None,
                kg_chunk_id_zero_only=None,
            ),
            chunks_above=0,
            chunks_below=0,
            rerank_settings=None,
            hybrid_alpha=0.5,
            recency_bias_multiplier=1.0,
            max_llm_filter_sections=0,
            original_query=query,
        )

    def _build_query_from_url(self, url: str) -> str:
        """Derive a human-meaningful query string from a URL for federated search.

        - For Slack URLs, extract channel and timestamp hints
        - Otherwise, use URL path and last path segment as keywords
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path or ""
            last_seg = path.rstrip("/").split("/")[-1]
            base_keywords: list[str] = []
            if last_seg:
                base_keywords.append(last_seg.replace("-", " ").replace("_", " "))

            # Slack-specific hinting
            source_type, method = self._get_source_type_and_method(domain)
            if source_type == DocumentSource.SLACK and method == "federated":
                parsed_slack = self._parse_slack_url(url)
                if parsed_slack:
                    channel_id, message_ts = parsed_slack
                    hint = f"channel:{channel_id} ts:{message_ts}"
                    base_keywords.append(hint)

            # Fallback to domain keywords if nothing else
            if not base_keywords:
                base_keywords.append(domain)

            # Join and trim
            query = " ".join(k for k in base_keywords if k).strip()
            return query or url
        except Exception:
            return url

    def _process_chunks_to_document_result(
        self,
        chunks: list,
        url: str,
        source_type: DocumentSource,
        source_method: str,
        document_id: str | None = None,
    ) -> DocumentResult:
        """Process chunks into full document content using shared utility"""
        return process_chunks_to_document_result(
            chunks=chunks,
            url=url,
            source_type=source_type,
            source_method=source_method,
            document_id=document_id,
        )

    def _extract_url_from_query(self, query: str) -> str | None:
        """Extract URL from query text using regex patterns"""
        # Regex pattern to match URLs (http/https)
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'

        # Find all URLs in the query
        urls = re.findall(url_pattern, query)

        if not urls:
            return None

        # Return the first URL found
        url = urls[0]

        # Clean up the URL (remove trailing punctuation)
        url = re.sub(r"[.,;:!?]+$", "", url)

        return url
