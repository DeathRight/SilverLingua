import json
import os

import requests

from SilverLingua.core.atoms.tool import Tool

url = "https://google.serper.dev/"

search_types = ["search", "images", "maps", "news", "shopping", "scholar"]


@Tool
def google_search(query: str, type: str = "search"):
    """
    Searches Google for the given query.

    Args:
        query: The query to search for.
        type: Can be one of "search", "images", "maps", "news", "shopping", "scholar". Defaults to "search".
    """
    payload = {
        "q": query,
        "location": "Dalles, Texas, United States",
    }

    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }

    response = requests.post(url + type, headers=headers, data=json.dumps(payload))

    return response.text
