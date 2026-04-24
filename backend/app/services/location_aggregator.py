"""
Location Aggregator Service
============================

Extracts geographic references from Zep knowledge-graph nodes and aggregates
them into a nested hierarchy::

    country → state → city → neighborhood

Each level exposes an ``entity_count`` and a list of the entities found
there.  Callers can also filter the output by country, entity_type, or any
combination of the two.

How locations are resolved
--------------------------
The service tries four sources in priority order for each node:

1. **Structured node attributes** — any attribute whose key is one of
   ``country``, ``state``, ``city``, ``neighborhood``, ``location``.
2. **Ontology-specific location attributes** — broader scan of all
   attributes for keys that look like they contain location data
   (``location``, ``address``, ``place``, ``locale``, ``region``).
3. **Node summary text** — lightweight regex patterns that recognise
   common phrases such as *"located in"*, *"based in"*, or
   *"from <City>, <State>"*.
4. **Related edge facts** — same regex patterns applied to the
   ``fact`` strings of edges incident to the node.

All extracted values are normalised (title-cased, stripped) before
being stored.  Nodes with no detectable location are placed under the
``"Unknown"`` bucket at every level so they are still counted in
totals but can be excluded by callers that only care about known
locations.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger('mirofish.location_aggregator')

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches phrases like "located in São Paulo", "based in Rio de Janeiro, SP"
_LOCATED_IN_RE = re.compile(
    r'(?:located|based|from|in|of|at)\s+([A-ZÁÉÍÓÚÂÊÔÃÕÀÜÇÑ][A-Za-záéíóúâêôãõàüçñ\s\-]{1,40})'
    r'(?:,\s*([A-ZÁÉÍÓÚÂÊÔÃÕÀÜÇÑ][A-Za-záéíóúâêôãõàüçñ\s\-]{1,40}))?',
    re.IGNORECASE
)

# Attribute key patterns considered "location-like"
_LOCATION_ATTR_KEYS = frozenset({
    'country', 'state', 'city', 'neighborhood', 'neighbourhood',
    'location', 'address', 'place', 'locale', 'region', 'district',
    'province', 'municipality', 'borough', 'cidade', 'estado', 'pais',
    'bairro', 'municipio',
})

# Canonical map from Portuguese/Spanish geo terms to the structured field name
_ATTR_KEY_NORMALISE = {
    'pais': 'country', 'país': 'country',
    'estado': 'state', 'provincia': 'state', 'província': 'state',
    'cidade': 'city', 'municipio': 'city', 'município': 'city',
    'bairro': 'neighborhood', 'neighbourhood': 'neighborhood',
    'distrito': 'neighborhood', 'borough': 'neighborhood',
}

_UNKNOWN = "Unknown"


def _get_node_uuid(node: Any) -> str:
    """Return the UUID string for a Zep node regardless of attribute name variant."""
    return getattr(node, 'uuid_', None) or getattr(node, 'uuid', '') or ''


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(value: Any) -> str:
    """Strip, title-case and return a string; empty input returns ''."""
    if not value:
        return ''
    return str(value).strip().title()


def _extract_from_attributes(attrs: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract location fields from a node's attribute dict.

    Returns a dict with zero or more of the keys:
    ``country``, ``state``, ``city``, ``neighborhood``, ``location``.
    """
    result: Dict[str, str] = {}
    if not attrs:
        return result

    for raw_key, raw_val in attrs.items():
        key = raw_key.lower().strip()
        canonical = _ATTR_KEY_NORMALISE.get(key, key)
        if canonical not in _LOCATION_ATTR_KEYS:
            continue
        val = _normalise(raw_val)
        if val:
            result[canonical] = val

    return result


def _extract_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Apply regex to free-form text and return (city_or_place, region_hint).

    Both values may be None if nothing was found.
    """
    if not text:
        return None, None

    match = _LOCATED_IN_RE.search(text)
    if match:
        place = _normalise(match.group(1))
        region = _normalise(match.group(2)) if match.group(2) else None
        return place or None, region
    return None, None


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------

class LocationAggregatorService:
    """
    Aggregate location data from a Zep graph.

    Parameters
    ----------
    api_key:
        Zep API key.  Defaults to ``Config.ZEP_API_KEY``.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY is not configured")

        zep_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if Config.ZEP_BASE_URL:
            zep_kwargs["base_url"] = Config.ZEP_BASE_URL
        self.client = Zep(**zep_kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_location_stats(
        self,
        graph_id: str,
        entity_type_filter: Optional[str] = None,
        country_filter: Optional[str] = None,
        include_unknown: bool = False,
    ) -> Dict[str, Any]:
        """
        Build a nested location-statistics structure for a graph.

        Parameters
        ----------
        graph_id:
            The Zep graph to analyse.
        entity_type_filter:
            When provided, only nodes whose labels include this type are
            counted (case-insensitive).
        country_filter:
            When provided, the output is limited to this country
            (case-insensitive prefix match).
        include_unknown:
            If ``True``, the ``"Unknown"`` bucket is included in the
            response.  Default is ``False``.

        Returns
        -------
        dict
            ``{
              "graph_id": ...,
              "total_entities_analysed": int,
              "total_with_location": int,
              "hierarchy": {
                "<country>": {
                  "count": int,
                  "states": {
                    "<state>": {
                      "count": int,
                      "cities": {
                        "<city>": {
                          "count": int,
                          "neighborhoods": {
                            "<neighborhood>": {"count": int, "entities": [...]}
                          },
                          "entities": [...]
                        }
                      },
                      "entities": [...]
                    }
                  },
                  "entities": [...]
                }
              }
            }``
        """
        logger.info(f"Building location stats for graph {graph_id}...")

        nodes = fetch_all_nodes(self.client, graph_id)
        edges = fetch_all_edges(self.client, graph_id)

        # Build an edge-fact index keyed by node uuid for fast lookup
        edge_facts_by_node: Dict[str, List[str]] = defaultdict(list)
        for edge in edges:
            fact = getattr(edge, 'fact', None) or ''
            if fact:
                src = getattr(edge, 'source_node_uuid', None)
                tgt = getattr(edge, 'target_node_uuid', None)
                if src:
                    edge_facts_by_node[src].append(fact)
                if tgt:
                    edge_facts_by_node[tgt].append(fact)

        # Nested dict:  hierarchy[country][state][city][neighborhood] = entity list
        hierarchy: Dict[str, Any] = {}
        total_analysed = 0
        total_with_location = 0

        for node in nodes:
            labels: List[str] = getattr(node, 'labels', []) or []
            node_uuid: str = _get_node_uuid(node)
            node_name: str = (node.name or '').strip()

            # Entity-type filter
            if entity_type_filter:
                custom = [l for l in labels if l not in ('Entity', 'Node')]
                if not any(l.lower() == entity_type_filter.lower() for l in custom):
                    continue

            total_analysed += 1

            attrs: Dict[str, Any] = node.attributes or {}
            summary: str = node.summary or ''

            # --- Step 1: structured attributes ---
            loc = _extract_from_attributes(attrs)

            country = loc.get('country') or ''
            state = loc.get('state') or ''
            city = loc.get('city') or ''
            neighborhood = loc.get('neighborhood') or ''

            # If a generic "location" attribute exists and we are still missing
            # city/state, use its value as city.
            if not city and loc.get('location'):
                city = loc['location']

            # --- Step 2: summary text ---
            if not country and not city:
                place, region = _extract_from_text(summary)
                if place:
                    city = place
                if region:
                    state = region

            # --- Step 3: edge facts ---
            if not country and not city:
                for fact in edge_facts_by_node.get(node_uuid, []):
                    place, region = _extract_from_text(fact)
                    if place:
                        city = place
                        if region:
                            state = region
                        break

            # Normalise
            country = _normalise(country) or _UNKNOWN
            state = _normalise(state) or _UNKNOWN
            city = _normalise(city) or _UNKNOWN
            neighborhood = _normalise(neighborhood) or _UNKNOWN

            # Country filter
            if country_filter:
                if country.lower() != country_filter.lower():
                    continue

            has_location = not (
                country == _UNKNOWN
                and state == _UNKNOWN
                and city == _UNKNOWN
                and neighborhood == _UNKNOWN
            )
            if has_location:
                total_with_location += 1

            # Entity summary for embedding in the response
            entity_types = [l for l in labels if l not in ('Entity', 'Node')]
            entity_entry = {
                "uuid": node_uuid,
                "name": node_name,
                "entity_type": entity_types[0] if entity_types else "Entity",
                "country": country,
                "state": state,
                "city": city,
                "neighborhood": neighborhood,
            }

            # --- Insert into hierarchy ---
            if country not in hierarchy:
                hierarchy[country] = {"count": 0, "states": {}, "entities": []}
            hierarchy[country]["count"] += 1
            hierarchy[country]["entities"].append(entity_entry)

            states_h = hierarchy[country]["states"]
            if state not in states_h:
                states_h[state] = {"count": 0, "cities": {}, "entities": []}
            states_h[state]["count"] += 1
            states_h[state]["entities"].append(entity_entry)

            cities_h = states_h[state]["cities"]
            if city not in cities_h:
                cities_h[city] = {"count": 0, "neighborhoods": {}, "entities": []}
            cities_h[city]["count"] += 1
            cities_h[city]["entities"].append(entity_entry)

            nbhd_h = cities_h[city]["neighborhoods"]
            if neighborhood not in nbhd_h:
                nbhd_h[neighborhood] = {"count": 0, "entities": []}
            nbhd_h[neighborhood]["count"] += 1
            nbhd_h[neighborhood]["entities"].append(entity_entry)

        # Optionally strip the Unknown bucket
        if not include_unknown:
            hierarchy.pop(_UNKNOWN, None)
            for c_data in hierarchy.values():
                c_data["states"].pop(_UNKNOWN, None)
                for s_data in c_data["states"].values():
                    s_data["cities"].pop(_UNKNOWN, None)
                    for city_data in s_data["cities"].values():
                        city_data["neighborhoods"].pop(_UNKNOWN, None)

        logger.info(
            f"Location stats complete: {total_analysed} entities analysed, "
            f"{total_with_location} with location data, "
            f"{len(hierarchy)} countries found"
        )

        return {
            "graph_id": graph_id,
            "total_entities_analysed": total_analysed,
            "total_with_location": total_with_location,
            "filters_applied": {
                "entity_type": entity_type_filter,
                "country": country_filter,
                "include_unknown": include_unknown,
            },
            "hierarchy": hierarchy,
        }

    def get_entities_by_location(
        self,
        graph_id: str,
        country: Optional[str] = None,
        state: Optional[str] = None,
        city: Optional[str] = None,
        neighborhood: Optional[str] = None,
        entity_type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a flat list of entities matching the given location components.

        At least one of *country*, *state*, *city*, or *neighborhood* must be
        provided; otherwise a ``ValueError`` is raised.

        Matching is case-insensitive.  Providing only ``country`` returns
        all entities in that country; providing ``country`` + ``city``
        narrows the result further.  Fields set to ``None`` are treated as
        wildcards.
        """
        if not any([country, state, city, neighborhood]):
            raise ValueError(
                "At least one location filter (country, state, city, neighborhood) must be provided"
            )

        stats = self.get_location_stats(
            graph_id=graph_id,
            entity_type_filter=entity_type_filter,
            country_filter=country,
            include_unknown=True,
        )

        results: List[Dict[str, Any]] = []
        for c_name, c_data in stats["hierarchy"].items():
            for s_name, s_data in c_data["states"].items():
                for city_name, city_data in s_data["cities"].items():
                    for nbhd_name, nbhd_data in city_data["neighborhoods"].items():
                        if state and s_name.lower() != state.lower():
                            continue
                        if city and city_name.lower() != city.lower():
                            continue
                        if neighborhood and nbhd_name.lower() != neighborhood.lower():
                            continue
                        results.extend(nbhd_data["entities"])

        # Deduplicate by uuid
        seen = set()
        unique: List[Dict[str, Any]] = []
        for e in results:
            if e["uuid"] not in seen:
                seen.add(e["uuid"])
                unique.append(e)

        return unique
