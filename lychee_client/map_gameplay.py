"""Map gameplay context derived from start message — no hardcoded node IDs in strategy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from lychee_client.map_graph import MapGraph
from lychee_client.state import GATE_CORRIDOR_NODES

# Legacy fallbacks when gameplay fields are missing (e.g. partial test payloads).
_DEFAULT_WATER_NODES = frozenset({"S04", "S05"})
_DEFAULT_OFFICIAL_MID = frozenset({"S03", "S07"})


class MapKind(str, Enum):
    PUBLIC = "public"
    VARIANT = "variant"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MapStrategyProfile:
    """Per-map tunables — strategy reads these instead of hard-coded constants."""

    map_kind: MapKind
    ice_box_freshness_threshold: float = 88.0
    ice_box_rush_use_threshold: float = 95.0
    water_route_task_min: int = 60
    require_ice_for_water: bool = False
    squad_scout_min_squad: int = 3
    force_delivery_slack_buffer: float = 28.0
    guard_min_task_score: int = 90
    near_gate_skip_permit_hops: int = 8
    intel_enabled: bool = True


PUBLIC_PROFILE = MapStrategyProfile(
    map_kind=MapKind.PUBLIC,
    ice_box_freshness_threshold=88.0,
    ice_box_rush_use_threshold=95.0,
    water_route_task_min=55,
    require_ice_for_water=False,
    squad_scout_min_squad=3,
    force_delivery_slack_buffer=28.0,
    guard_min_task_score=90,
)

VARIANT_PROFILE = MapStrategyProfile(
    map_kind=MapKind.VARIANT,
    ice_box_freshness_threshold=88.0,
    ice_box_rush_use_threshold=95.0,
    water_route_task_min=50,
    require_ice_for_water=False,
    squad_scout_min_squad=3,
    force_delivery_slack_buffer=32.0,
    guard_min_task_score=90,
    near_gate_skip_permit_hops=6,
)

DEFAULT_PROFILE = MapStrategyProfile(map_kind=MapKind.UNKNOWN)


@dataclass(frozen=True)
class MapGameplayContext:
    water_route_nodes: frozenset[str]
    official_mid_route_nodes: frozenset[str]
    obstacle_candidate_node_ids: frozenset[str]
    route_task_buckets: dict[str, tuple[str, ...]]
    static_resources: tuple[dict[str, Any], ...]
    map_id: str = ""
    map_kind: MapKind = MapKind.UNKNOWN
    profile: MapStrategyProfile = DEFAULT_PROFILE
    pass_token_nodes: frozenset[str] = frozenset()
    official_permit_nodes: frozenset[str] = frozenset()
    intel_nodes: frozenset[str] = frozenset()


def default_map_gameplay() -> MapGameplayContext:
    return MapGameplayContext(
        water_route_nodes=_DEFAULT_WATER_NODES,
        official_mid_route_nodes=_DEFAULT_OFFICIAL_MID,
        obstacle_candidate_node_ids=frozenset(),
        route_task_buckets={},
        static_resources=(),
        map_kind=MapKind.UNKNOWN,
        profile=DEFAULT_PROFILE,
    )


def _water_nodes_from_edges(graph: MapGraph) -> set[str]:
    nodes: set[str] = set()
    for (from_id, to_id), edge in graph.edge_info.items():
        if edge.get("routeType") == "WATER":
            nodes.add(from_id)
            nodes.add(to_id)
    return nodes


def _derive_official_mid_nodes(
    road_bucket: list[str],
    water_bucket: set[str],
    mountain_bucket: set[str],
    start_node_id: str,
) -> frozenset[str]:
    """ROAD task nodes that are not shared with WATER/MOUNTAIN buckets (官道锚点)."""
    road = set(road_bucket)
    pure_road = road - water_bucket - mountain_bucket - GATE_CORRIDOR_NODES
    if start_node_id:
        pure_road.discard(start_node_id)
    if pure_road:
        return frozenset(pure_road)
    fallback = road - GATE_CORRIDOR_NODES
    if start_node_id:
        fallback.discard(start_node_id)
    if fallback:
        return frozenset(fallback)
    return _DEFAULT_OFFICIAL_MID


def _process_type_at(
    node_id: str,
    gameplay: dict[str, Any],
    graph: MapGraph | None,
) -> str:
    for pn in gameplay.get("processNodes", []) or []:
        if pn.get("nodeId") == node_id:
            return pn.get("processType", "") or ""
    if graph:
        node = graph.get_node(node_id)
        if node:
            return node.get("processType", "") or ""
    return ""


def _detect_map_kind(
    gameplay: dict[str, Any],
    graph: MapGraph | None,
) -> MapKind:
    """Identify public vs variant map from process layout and obstacle candidates."""
    s04_pt = _process_type_at("S04", gameplay, graph)
    s05_pt = _process_type_at("S05", gameplay, graph)
    if s04_pt == "WATER_TRANSFER" and s05_pt == "BOARD":
        return MapKind.VARIANT
    if s04_pt == "BOARD" and s05_pt == "WATER_TRANSFER":
        return MapKind.PUBLIC

    obstacles = set(gameplay.get("obstacleCandidateNodeIds", []) or [])
    if "S07" in obstacles and "S08" not in obstacles:
        return MapKind.VARIANT
    if "S08" in obstacles and "S07" not in obstacles:
        return MapKind.PUBLIC

    if graph:
        for (_from, _to), edge in graph.edge_info.items():
            if edge.get("edgeId") == "E18":
                if "S02" in (_from, _to):
                    return MapKind.VARIANT
                if "S03" in (_from, _to):
                    return MapKind.PUBLIC
            dist = edge.get("distance", 0)
            nodes_pair = {_from, _to}
            if nodes_pair == {"S02", "S06"} or (dist == 80 and "S06" in nodes_pair and "S02" in nodes_pair):
                return MapKind.VARIANT
            if nodes_pair == {"S03", "S06"} or (dist == 38 and "S06" in nodes_pair and "S03" in nodes_pair):
                return MapKind.PUBLIC

    map_id = str(gameplay.get("mapId") or "")
    if "variant" in map_id.lower() or "变种" in map_id:
        return MapKind.VARIANT
    if "public" in map_id.lower() or "公开" in map_id:
        return MapKind.PUBLIC
    return MapKind.UNKNOWN


def _profile_for_kind(kind: MapKind) -> MapStrategyProfile:
    if kind == MapKind.PUBLIC:
        return PUBLIC_PROFILE
    if kind == MapKind.VARIANT:
        return VARIANT_PROFILE
    return DEFAULT_PROFILE


def _resource_nodes(static_resources: tuple[dict[str, Any], ...]) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    pass_nodes: set[str] = set()
    permit_nodes: set[str] = set()
    intel_nodes: set[str] = set()
    for res in static_resources:
        nid = res.get("nodeId", "")
        rtype = res.get("resourceType", "")
        if not nid:
            continue
        if rtype == "PASS_TOKEN":
            pass_nodes.add(nid)
        elif rtype == "OFFICIAL_PERMIT":
            permit_nodes.add(nid)
        elif rtype == "INTEL":
            intel_nodes.add(nid)
    return frozenset(pass_nodes), frozenset(permit_nodes), frozenset(intel_nodes)


def build_map_gameplay(start_raw: dict[str, Any], graph: MapGraph | None) -> MapGameplayContext:
    map_data = start_raw.get("map", {}) or {}
    gameplay = map_data.get("gameplay", {}) or start_raw.get("gameplay", {}) or {}
    buckets_raw = gameplay.get("routeTaskBuckets", {}) or {}
    buckets = {k: tuple(v or []) for k, v in buckets_raw.items()}

    water_bucket = set(buckets.get("WATER", ()))
    water_nodes = set(water_bucket)
    if graph:
        water_nodes |= _water_nodes_from_edges(graph)
    if not water_nodes:
        water_nodes = set(_DEFAULT_WATER_NODES)

    roles = gameplay.get("roles", {}) or {}
    start_node_id = roles.get("startNodeId", "")
    official_mid = _derive_official_mid_nodes(
        list(buckets.get("ROAD", ())),
        water_bucket,
        set(buckets.get("MOUNTAIN", ())),
        start_node_id,
    )

    obstacle_ids = frozenset(gameplay.get("obstacleCandidateNodeIds", []) or [])
    static_resources = tuple(start_raw.get("resources", []) or [])
    if not static_resources:
        static_resources = tuple(
            gameplay.get("resources", [])
            or map_data.get("resources", [])
            or []
        )

    map_kind = _detect_map_kind(gameplay, graph)
    profile = _profile_for_kind(map_kind)
    pass_nodes, permit_nodes, intel_nodes = _resource_nodes(static_resources)
    map_id = str(
        map_data.get("mapId")
        or start_raw.get("mapId")
        or gameplay.get("mapId")
        or ""
    )

    return MapGameplayContext(
        water_route_nodes=frozenset(water_nodes),
        official_mid_route_nodes=official_mid,
        obstacle_candidate_node_ids=obstacle_ids,
        route_task_buckets=buckets,
        static_resources=static_resources,
        map_id=map_id,
        map_kind=map_kind,
        profile=profile,
        pass_token_nodes=pass_nodes,
        official_permit_nodes=permit_nodes,
        intel_nodes=intel_nodes,
    )
