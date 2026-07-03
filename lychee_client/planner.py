"""Global planning helpers for route, task, resource, and opponent valuation.

The planner is intentionally lightweight: it does not build protocol actions.
It scores the current world so strategy.py can keep enforcing action legality.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from math import isinf

from lychee_client.map_graph import MapGraph, PROCESS_COST_FRAMES
from lychee_client.state import (
    TASK_PRIORITY,
    get_current_node_id,
    get_freshness,
    get_good_fruit,
    get_player_resources,
    get_task_score,
    get_task_template_id,
    get_team_id,
    has_resource,
    is_delivered,
    is_enemy_guard,
    is_verified,
    node_has_obstacle,
)

logger = logging.getLogger("lychee_client.planner")

MAX_ROUND = 600
SAFE_DELIVERY_BUFFER = 90
MIN_TASK_NET_VALUE = 8.0
MIN_RESOURCE_NET_VALUE = 6.0

RESOURCE_BASE_VALUE = {
    "FAST_HORSE": 28.0,
    "SHORT_HORSE": 18.0,
    "BOAT_RIGHT": 20.0,
    "OFFICIAL_PERMIT": 13.0,
    "PASS_TOKEN": 11.0,
    "ICE_BOX": 9.0,
    "INTEL": 7.0,
}


@dataclass(frozen=True)
class RouteEstimate:
    path: list[str]
    cost: float
    water_edges: int
    total_edges: int

    @property
    def water_ratio(self) -> float:
        if self.total_edges <= 0:
            return 0.0
        return self.water_edges / self.total_edges


@dataclass(frozen=True)
class MapProfile:
    water_edge_ratio: float
    mountain_edge_ratio: float
    branch_edge_ratio: float
    best_route_water_ratio: float
    best_route_cost: float
    resource_counts: dict[str, int]
    choke_nodes: set[str]

    @property
    def favors_water(self) -> bool:
        return self.best_route_water_ratio >= 0.30 or self.water_edge_ratio >= 0.25


@dataclass(frozen=True)
class GlobalPlan:
    round_num: int
    direct_eta: float
    opponent_eta: float
    score_gap: float
    task_gap: int
    water_ratio: float
    should_force_delivery: bool
    task_weight: float
    resource_weight: float
    combat_weight: float
    reason: str


def build_global_plan(
    round_num: int,
    player: dict,
    graph: MapGraph,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str] | None,
    blocked_nodes: set[str] | None,
    obstacle_nodes: set[str] | None,
    all_players: list[dict] | None,
    player_id: int,
    phase: str,
    map_profile: MapProfile | None = None,
) -> GlobalPlan:
    """Build a compact global plan used by the rule-based executor."""
    current = get_current_node_id(player) or ""
    processed = processed_node_ids or set()
    blocked = set(blocked_nodes or set())
    blocked.update(obstacle_nodes or set())
    direct = estimate_delivery_route(
        graph, current, player, gate_node_id, terminal_node_ids,
        weather, process_nodes, processed, blocked,
    )
    direct_eta = direct.cost

    opponent = _find_opponent(all_players or [], player_id)
    opponent_eta = float("inf")
    if opponent:
        opponent_eta = estimate_delivery_route(
            graph, get_current_node_id(opponent) or "", opponent,
            gate_node_id, terminal_node_ids, weather, process_nodes,
            set(), set(),
        ).cost

    my_score = float(player.get("totalScore", 0) or 0)
    opp_score = float(opponent.get("totalScore", 0) or 0) if opponent else my_score
    score_gap = my_score - opp_score
    task_gap = get_task_score(player) - (get_task_score(opponent) if opponent else get_task_score(player))
    rounds_left = MAX_ROUND - round_num
    safe_buffer = SAFE_DELIVERY_BUFFER
    if map_profile and map_profile.favors_water:
        safe_buffer -= 12
    if map_profile and map_profile.best_route_cost > 430:
        safe_buffer += 18
    delivery_risk = direct_eta >= rounds_left - safe_buffer
    opponent_time_lead = opponent_eta + 20 < direct_eta
    opponent_score_lead = score_gap < -20 or task_gap < -30
    enough_task_score = get_task_score(player) >= 60

    should_force = (
        phase == "RUSH"
        or delivery_risk
        or (enough_task_score and (round_num >= 95 or opponent_time_lead))
        or round_num >= 175
    )
    if opponent_score_lead and not delivery_risk and phase != "RUSH":
        should_force = False

    task_weight = 1.0
    resource_weight = 1.0
    combat_weight = 0.6
    reason_parts = []
    if should_force:
        task_weight = 0.35
        resource_weight = 0.45
        combat_weight = 0.35
        reason_parts.append("delivery_risk" if delivery_risk else "force_delivery")
    if opponent_score_lead:
        task_weight += 0.55
        combat_weight += 0.55
        reason_parts.append("opponent_score_lead")
    if opponent_time_lead:
        resource_weight += 0.35
        combat_weight += 0.45
        reason_parts.append("opponent_time_lead")
    if direct.water_ratio >= 0.34 or (map_profile and map_profile.favors_water):
        resource_weight += 0.25
        reason_parts.append("water_route")
    if map_profile and map_profile.choke_nodes:
        combat_weight += min(0.25, len(map_profile.choke_nodes) * 0.04)
    if get_freshness(player) < 45:
        resource_weight += 0.25
        reason_parts.append("freshness_low")

    if not reason_parts:
        reason_parts.append("balanced")

    plan = GlobalPlan(
        round_num=round_num,
        direct_eta=direct_eta,
        opponent_eta=opponent_eta,
        score_gap=score_gap,
        task_gap=task_gap,
        water_ratio=direct.water_ratio,
        should_force_delivery=should_force,
        task_weight=task_weight,
        resource_weight=resource_weight,
        combat_weight=combat_weight,
        reason="+".join(reason_parts),
    )
    logger.info(
        "PLAN global eta=%.1f oppEta=%s scoreGap=%.1f taskGap=%d water=%.2f force=%s reason=%s",
        plan.direct_eta,
        "inf" if isinf(plan.opponent_eta) else f"{plan.opponent_eta:.1f}",
        plan.score_gap,
        plan.task_gap,
        plan.water_ratio,
        plan.should_force_delivery,
        plan.reason,
    )
    return plan


def build_map_profile(
    graph: MapGraph,
    start_node_id: str,
    gate_node_id: str,
    terminal_node_ids: list[str],
    process_nodes: dict[str, dict] | None,
    nodes: list[dict] | None = None,
    resources: list[dict] | None = None,
) -> MapProfile:
    """Summarize static start-map traits used to tune global weights."""
    route_counts = {"WATER": 0, "MOUNTAIN": 0, "BRANCH": 0, "ROAD": 0}
    seen_edges: set[tuple[str, str]] = set()
    for (a, b), edge in graph.edge_info.items():
        key = tuple(sorted((a, b)))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        route_type = edge.get("routeType", "ROAD")
        route_counts[route_type] = route_counts.get(route_type, 0) + 1
    total_edges = max(1, sum(route_counts.values()))

    probe_player = {
        "currentNodeId": start_node_id,
        "verified": False,
        "goodFruit": 100,
        "resources": {},
    }
    route = estimate_delivery_route(
        graph, start_node_id, probe_player, gate_node_id, terminal_node_ids,
        None, process_nodes, set(), set(),
    )
    resource_counts: dict[str, int] = {}
    for node in nodes or graph.node_info.values():
        stock = node.get("resourceStock", {}) or {}
        for rtype, count in stock.items():
            resource_counts[rtype] = resource_counts.get(rtype, 0) + int(count or 0)
    for resource in resources or []:
        rtype = resource.get("resourceType") or resource.get("type", "")
        count = int(resource.get("count", 1) or 1)
        if rtype:
            resource_counts[rtype] = resource_counts.get(rtype, 0) + count

    choke_nodes = _find_static_choke_nodes(graph, start_node_id, gate_node_id, terminal_node_ids)
    profile = MapProfile(
        water_edge_ratio=route_counts.get("WATER", 0) / total_edges,
        mountain_edge_ratio=route_counts.get("MOUNTAIN", 0) / total_edges,
        branch_edge_ratio=route_counts.get("BRANCH", 0) / total_edges,
        best_route_water_ratio=route.water_ratio,
        best_route_cost=route.cost,
        resource_counts=resource_counts,
        choke_nodes=choke_nodes,
    )
    logger.info(
        "MAP profile waterEdges=%.2f bestWater=%.2f bestCost=%.1f resources=%s chokes=%s",
        profile.water_edge_ratio,
        profile.best_route_water_ratio,
        profile.best_route_cost,
        profile.resource_counts,
        sorted(profile.choke_nodes),
    )
    return profile


def estimate_delivery_route(
    graph: MapGraph,
    current_node_id: str,
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
    blocked_nodes: set[str] | None,
) -> RouteEstimate:
    """Estimate weighted cost from current node to verified delivery."""
    if not current_node_id:
        return RouteEstimate([], float("inf"), 0, 0)
    goals: list[str] = []
    if not is_verified(player) and gate_node_id:
        goals.append(gate_node_id)
    goals.extend(terminal_node_ids or [])
    if not goals:
        return RouteEstimate([current_node_id], 0.0, 0, 0)

    remaining_process = {
        nid: info for nid, info in (process_nodes or {}).items()
        if nid not in processed_node_ids
    }
    full_path = [current_node_id]
    total_cost = 0.0
    water_edges = 0
    total_edges = 0
    cursor = current_node_id
    blocked = set(blocked_nodes or set())

    for idx, goal in enumerate(goals):
        path = graph.weighted_shortest_path(cursor, goal, weather, blocked, remaining_process)
        if not path:
            return RouteEstimate(full_path, float("inf"), water_edges, total_edges)
        for a, b in zip(path, path[1:]):
            total_cost += graph.edge_cost(a, b, weather, blocked, remaining_process)
            total_edges += 1
            if graph.get_edge_route_type(a, b) == "WATER":
                water_edges += 1
        full_path.extend(path[1:])
        cursor = goal
        if idx == 0 and goal == gate_node_id and not is_verified(player):
            verify_cost = PROCESS_COST_FRAMES.get("VERIFY", 6)
            if get_good_fruit(player) >= 1:
                verify_cost = max(3, verify_cost - 3)
            total_cost += verify_cost

    return RouteEstimate(full_path, total_cost, water_edges, total_edges)


def task_net_value(
    task: dict,
    detour_cost: float,
    round_num: int,
    plan: GlobalPlan,
    is_current_node: bool,
) -> float:
    """Score whether a task is worth taking under the global plan."""
    template_id = get_task_template_id(task)
    task_score, process_rounds, base_spr = _task_profile(template_id)
    expire_round = int(task.get("expireRound", 0) or 0)
    if expire_round and round_num + detour_cost + process_rounds >= expire_round:
        return -999.0
    urgency_bonus = 14.0 if plan.task_gap < -30 else 0.0
    current_bonus = 6.0 if is_current_node else 0.0
    force_penalty = 18.0 if plan.should_force_delivery and not is_current_node else 0.0
    time_penalty = detour_cost * (1.15 if plan.should_force_delivery else 0.8)
    return (
        task_score * plan.task_weight
        + base_spr
        + urgency_bonus
        + current_bonus
        - process_rounds
        - time_penalty
        - force_penalty
    )


def resource_net_value(
    resource_type: str,
    player: dict,
    plan: GlobalPlan,
    route: RouteEstimate | None = None,
    map_profile: MapProfile | None = None,
) -> float:
    """Score whether a resource should be claimed now."""
    if resource_type in ("FAST_HORSE", "SHORT_HORSE") and has_resource(player, resource_type):
        return -999.0
    if resource_type == "ICE_BOX" and get_freshness(player) > 75 and plan.direct_eta < 180:
        return -8.0
    favors_water = bool(map_profile and map_profile.favors_water)
    if resource_type == "BOAT_RIGHT" and route and route.water_ratio < 0.25 and not favors_water:
        return -4.0
    if resource_type in ("OFFICIAL_PERMIT", "PASS_TOKEN"):
        permits = get_player_resources(player).get("OFFICIAL_PERMIT", 0) + get_player_resources(player).get("PASS_TOKEN", 0)
        if permits >= 2:
            return -5.0
    value = RESOURCE_BASE_VALUE.get(resource_type, 3.0)
    if resource_type == "BOAT_RIGHT" and (plan.water_ratio >= 0.34 or favors_water):
        value += 11.0
    if resource_type in ("FAST_HORSE", "SHORT_HORSE") and map_profile and map_profile.mountain_edge_ratio >= 0.25:
        value += 4.0
    if resource_type == "ICE_BOX" and map_profile and map_profile.mountain_edge_ratio >= 0.25:
        value += 4.0
    if resource_type in ("FAST_HORSE", "SHORT_HORSE") and plan.direct_eta > 120:
        value += 9.0
    if resource_type == "ICE_BOX" and get_freshness(player) < 55:
        value += 15.0
    if plan.should_force_delivery and resource_type not in ("FAST_HORSE", "SHORT_HORSE", "BOAT_RIGHT"):
        value -= 12.0
    return value * plan.resource_weight - 4.0


def should_set_guard_now(
    player: dict,
    opponent: dict | None,
    current_node_id: str,
    graph: MapGraph,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    blocked_nodes: set[str] | None,
    inquire_nodes: list[dict],
    plan: GlobalPlan,
) -> bool:
    """Decide whether setting a guard is globally worthwhile."""
    if not opponent or not current_node_id or get_good_fruit(player) < 1:
        return False
    if plan.combat_weight < 0.9:
        return False
    if plan.should_force_delivery and plan.direct_eta < plan.opponent_eta + 25:
        return False
    my_team_id = get_team_id(player)
    for node in inquire_nodes:
        if node.get("nodeId") == current_node_id:
            if is_enemy_guard(node.get("guard"), my_team_id, player.get("playerId")):
                return False
            if node.get("guard") and node.get("guard", {}).get("active", False):
                return False
            break
    opp_route = estimate_delivery_route(
        graph, get_current_node_id(opponent) or "", opponent,
        gate_node_id, terminal_node_ids, weather, None, set(), blocked_nodes,
    )
    if current_node_id not in opp_route.path:
        return False
    my_route = estimate_delivery_route(
        graph, current_node_id, player, gate_node_id, terminal_node_ids,
        weather, None, set(), blocked_nodes,
    )
    return current_node_id not in my_route.path[1:3]


def _task_profile(template_id: str) -> tuple[int, int, float]:
    for prefix, profile in TASK_PRIORITY.items():
        if template_id.startswith(prefix):
            return profile
    return (15, 5, 3.0)


def _find_opponent(all_players: list[dict], my_player_id: int) -> dict | None:
    for p in all_players:
        if p.get("playerId") != my_player_id and not is_delivered(p):
            return p
    return None


def _find_static_choke_nodes(
    graph: MapGraph,
    start_node_id: str,
    gate_node_id: str,
    terminal_node_ids: list[str],
) -> set[str]:
    if not start_node_id or not gate_node_id:
        return set()
    goals = [gate_node_id] + list(terminal_node_ids or [])
    paths: list[list[str]] = []
    cursor = start_node_id
    for goal in goals:
        path = graph.weighted_shortest_path(cursor, goal, None, None, None)
        if path:
            paths.append(path)
            cursor = goal
    if not paths:
        return set()
    route_nodes = [node for path in paths for node in path[1:-1]]
    chokepoints = set()
    for node_id in route_nodes:
        degree = len(graph.get_neighbors(node_id))
        if degree <= 2:
            chokepoints.add(node_id)
    return chokepoints
