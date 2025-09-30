import json
from concurrent.futures import ThreadPoolExecutor

import requests

from onyx.agents.agent_search.dr.sub_agents.web_search.models import (
    InternetContent,
)
from onyx.agents.agent_search.dr.sub_agents.web_search.models import (
    InternetSearchProvider,
)
from onyx.agents.agent_search.dr.sub_agents.web_search.models import (
    InternetSearchResult,
)
from onyx.configs.chat_configs import SERPER_API_KEY
from onyx.connectors.cross_connector_utils.miscellaneous_utils import time_str_to_utc
from onyx.utils.retry_wrapper import retry_builder


SERPER_SEARCH_URL = "https://google.serper.dev/search"
SERPER_CONTENTS_URL = "https://scrape.serper.dev"


class SerperClient(InternetSearchProvider):
    def __init__(self, api_key: str | None = SERPER_API_KEY) -> None:
        self.api_key = api_key

    @retry_builder(tries=3, delay=1, backoff=2)
    def search(self, query: str) -> list[InternetSearchResult]:
        headers = self._create_header()

        payload = {
            "q": query,
        }

        response = requests.post(
            SERPER_SEARCH_URL,
            headers=headers,
            data=json.dumps(payload),
        )

        results = response.json()
        organic_results = results["organic"]

        return [
            InternetSearchResult(
                title=result["title"],
                link=result["link"],
                snippet=result["snippet"],
                author=None,
                published_date=None,
            )
            for result in organic_results
        ]

    def contents(self, urls: list[str]) -> list[InternetContent]:
        if not urls:
            return []

        with ThreadPoolExecutor(max_workers=min(4, len(urls))) as e:
            return list(e.map(self._get_webpage_content, urls))

    @retry_builder(tries=3, delay=1, backoff=2)
    def _get_webpage_content(self, url: str) -> InternetContent:
        headers = self._create_header()

        payload = {
            "url": url,
        }

        response = requests.post(
            SERPER_CONTENTS_URL,
            headers=headers,
            data=json.dumps(payload),
        )

        response_json = response.json()

        # Response contains at a minimum text and metadata
        text = response_json["text"]
        metadata = response_json["metadata"]

        # jsonld is not guaranteed to be present
        jsonld = response_json["jsonld"] if "jsonld" in response_json else {}

        title = extract_title_from_metadata(metadata)

        # Serper does not provide an easy mechanism to extract the url
        response_url = url
        published_date = extract_published_date_from_jsonld(jsonld)

        if published_date:
            try:
                published_date = time_str_to_utc(published_date)
            except ValueError:
                published_date = None

        return InternetContent(
            title=title or "",
            link=response_url or url,
            full_content=text or "",
            published_date=published_date,
        )

    def _create_header(self) -> dict[str, str]:
        return {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }


def extract_title_from_metadata(metadata: dict[str, str]) -> str | None:
    keys = ["title", "og:title"]
    return extract_value_from_dict(metadata, keys)


def extract_published_date_from_jsonld(jsonld: dict[str, str]) -> str | None:
    keys = ["dateModified"]
    return extract_value_from_dict(jsonld, keys)


def extract_value_from_dict(data: dict[str, str], keys: list[str]) -> str | None:
    for key in keys:
        if key in data:
            return data[key]
    return None
