"""
Fetching Napari Plugin data from Napari GitHub API.
"""

import json
from typing import Dict, List, NamedTuple, Optional
from urllib import request


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

NAPARI_INDEX_URL = "https://api.napari-hub.org/plugins"


NAPARI_PLUGIN_URL_TEMPLATE = "https://api.napari-hub.org/plugins/{}"

# Maps each Napari Hub plugin Category to corresponding EDAM ClassID
CATEGORY_TERM_MAP = {
    "Supported data": "http://edamontology.org/data_Image",
    "Image modality": "http://edamontology.org/topic_3382",
    "Workflow step": "http://edamontology.org/operation_0004"
}


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class NapariPlugin(NamedTuple):
    name: str
    summary: str
    description: Optional[str] = None
    # From the 'category' field of plugin-data
    categories: Optional[Dict[str, List[str]]] = None

    def get_combined_description(self):
        descr = self.summary
        if self.description:
            if descr:
                descr += "\n"
            descr += self.description

        return descr
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def fetch_all_plugins() -> Dict[str, str]:
    """
    :return: Dict[ Plugin-Name [str] => Plugin-Version [str] ]
    """
    with request.urlopen(NAPARI_INDEX_URL) as f:
        response = f.read()
        plugin_index = json.loads(response)

    return plugin_index


def fetch_plugin(plugin_name: str) -> Optional[NapariPlugin]:
    """
    Returns `None` if plugin with provided name not found.
    """
    with request.urlopen(NAPARI_PLUGIN_URL_TEMPLATE.format(plugin_name)) as f:
        response = f.read()
        plugin_data = json.loads(response)

    if plugin_data:
        return NapariPlugin(name=plugin_data["name"], summary=plugin_data["summary"],
                            description=plugin_data.get("description_text"),
                            categories=plugin_data.get("category"))
