"""Global planning helpers for route, task, resource, and opponent valuation.

The planner is intentionally lightweight: it does not build protocol actions.
It scores the current world so strategy.py can keep enforcing action legality.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from itertools import permutations
from math import isinf, isfinite

from lychee_client.map_graph import MapGraph, PROCESS_COST_FRAMES
from lychee_client.state import (
    MAX_ROUND,
    TASK_PRIORITY,
    TASK_SCORE_TARGET,
    get_current_node_id,
    get_freshness,
    get_good_fruit,
    get_player_resources,
    get_task_score,
    get_task_template_id,
    has_resource,
    get_team_id,
    is_delivered,
    is_enemy_guard,
    is_verified,
    node_has_obstacle,
)

logger = logging.getLogger("lychee_client.planner")

SAFE_DELIVERY_BUFFER = 90
MIN_TASK_NET_VALUE = 8.0
MIN_RESOURCE_NET_VALUE = 6.0
# Average extra frames to clear / forced-pass an obstacle or guard sitting on the
# route. Obstacles are NOT impassable: treating a choke-point block as unreachable
# would inflate the delivery ETA to infinity and wrongly trigger force-delivery.
OBSTACLE_TRAVERSAL_PENALTY = 15.0
# Rolling tactical window: evaluate multi-task routes within this frame budget.
DEFAULT_PLANNING_HORIZON = 18.0
MAX_SEQUENCE_TASKS = 3
MAX_SEQUENCE_PERMUTE = 4

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
class TaskSequencePlan:
    """Short-horizon multi-task route plan (15-20 frame rolling window)."""
    task_ids: tuple[str, ...]
    task_nodes: tuple[str, ...]
    total_score: int
    total_cost: float
    extra_delivery_cost: float
    net_value: float
    next_task_id: str
    next_task_node: str
    horizon: float
    reason: str


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
    task_sequence: TaskSequencePlan | None = None


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
    tasks: list[dict] | None = None,
    failed_task_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    max_round: int = MAX_ROUND,
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
    direct_eta = _resolve_delivery_eta(
        direct.cost, graph, current, player, gate_node_id, terminal_node_ids,
        process_nodes, processed, map_profile,
    )

    opponent = _find_opponent(all_players or [], player_id)
    opponent_eta = float("inf")
    if opponent:
        opp_route = estimate_delivery_route(
            graph, get_current_node_id(opponent) or "", opponent,
            gate_node_id, terminal_node_ids, weather, process_nodes,
            set(), set(),
        )
        opponent_eta = _resolve_delivery_eta(
            opp_route.cost, graph, get_current_node_id(opponent) or "", opponent,
            gate_node_id, terminal_node_ids, process_nodes, set(), map_profile,
        )

    my_score = float(player.get("totalScore", 0) or 0)
    opp_score = float(opponent.get("totalScore", 0) or 0) if opponent else my_score
    score_gap = my_score - opp_score
    task_gap = get_task_score(player) - (get_task_score(opponent) if opponent else get_task_score(player))
    rounds_left = max_round - round_num
    safe_buffer = SAFE_DELIVERY_BUFFER
    if map_profile and map_profile.favors_water:
        safe_buffer -= 12
    if map_profile and map_profile.best_route_cost > 430:
        safe_buffer += 18

    route_budget = map_profile.best_route_cost if map_profile else (
        direct_eta if isfinite(direct_eta) else 300.0
    )
    task_window_deadline = max_round - route_budget - safe_buffer * 0.5

    delivery_risk = isfinite(direct_eta) and direct_eta >= rounds_left - safe_buffer
    opponent_time_lead = (
        isfinite(direct_eta)
        and isfinite(opponent_eta)
        and opponent_eta + 20 < direct_eta
    )
    opponent_score_lead = score_gap < -20 or task_gap < -30
    enough_task_score = get_task_score(player) >= 60

    should_force = (
        phase == "RUSH"
        or delivery_risk
        or get_task_score(player) >= 90
        or (enough_task_score and opponent_time_lead)
        or round_num >= task_window_deadline
    )
    # Still inside the adaptive task window with safe delivery slack — keep scoring.
    if (
        should_force
        and not delivery_risk
        and phase != "RUSH"
        and get_task_score(player) < TASK_SCORE_TARGET
        and round_num < task_window_deadline
        and isfinite(direct_eta)
        and direct_eta < rounds_left - safe_buffer
    ):
        should_force = False
    elif (
        opponent_score_lead
        and get_task_score(player) < 90
        and isfinite(direct_eta)
        and direct_eta < rounds_left - safe_buffer - 80
        and phase != "RUSH"
    ):
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

    task_sequence: TaskSequencePlan | None = None
    if (
        not should_force
        and phase != "RUSH"
        and get_task_score(player) < TASK_SCORE_TARGET
        and tasks
    ):
        task_sequence = build_task_sequence_plan(
            round_num=round_num,
            player=player,
            graph=graph,
            gate_node_id=gate_node_id,
            terminal_node_ids=terminal_node_ids,
            weather=weather,
            blocked_nodes=blocked,
            process_nodes=process_nodes,
            tasks=tasks,
            player_id=player_id,
            failed_task_ids=failed_task_ids or set(),
            visited_node_ids=visited_node_ids or set(),
            obstacle_nodes=obstacle_nodes or set(),
            map_profile=map_profile,
            task_weight=task_weight,
            should_force_delivery=should_force,
        )

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
        task_sequence=task_sequence,
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
    if task_sequence:
        logger.info(
            "PLAN task_sequence nodes=%s score=%d cost=%.1f extra=%.1f net=%.1f "
            "horizon=%.0f next=%s reason=%s",
            list(task_sequence.task_nodes),
            task_sequence.total_score,
            task_sequence.total_cost,
            task_sequence.extra_delivery_cost,
            task_sequence.net_value,
            task_sequence.horizon,
            task_sequence.next_task_node,
            task_sequence.reason,
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
        # Prefer a route that avoids blocked nodes, but never treat blocks as
        # impassable: obstacles/guards can be cleared or forced-passed, so a
        # choke-point block must not inflate the ETA to infinity.
        path = _find_route_segment(graph, cursor, goal, weather, blocked, remaining_process)
        if not path:
            return RouteEstimate(full_path, float("inf"), water_edges, total_edges)
        segment_cost = _path_frames_cost(
            graph, path, weather, blocked, remaining_process,
        )
        if isinf(segment_cost):
            return RouteEstimate(full_path, float("inf"), water_edges, total_edges)
        total_cost += segment_cost
        for a, b in zip(path, path[1:]):
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


def _find_route_segment(
    graph: MapGraph,
    start: str,
    goal: str,
    weather: dict | None,
    blocked: set[str],
    process_nodes: dict[str, dict] | None,
) -> list[str]:
    """Find a route segment; fall back to unweighted BFS if weighted search fails."""
    path = graph.weighted_shortest_path(start, goal, weather, blocked, process_nodes)
    if not path:
        path = graph.weighted_shortest_path(start, goal, weather, None, process_nodes)
    if not path:
        path = graph.shortest_path(start, goal, weather, None)
    return path


def _path_frames_cost(
    graph: MapGraph,
    path: list[str],
    weather: dict | None,
    blocked: set[str],
    process_nodes: dict[str, dict] | None,
) -> float:
    """Sum frame cost along a path, with weather timing and obstacle penalties."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    elapsed = 0.0
    for a, b in zip(path[:-1], path[1:]):
        arrival = (graph.current_round + elapsed) if graph.current_round is not None else None
        edge = graph.edge_cost(a, b, weather, None, process_nodes, arrival_round=arrival)
        if isinf(edge):
            edge = graph.edge_cost(a, b, None, None, process_nodes, arrival_round=arrival)
        if isinf(edge):
            return float("inf")
        total += edge
        if b in blocked:
            total += OBSTACLE_TRAVERSAL_PENALTY
        elapsed += edge
    return total


def _resolve_delivery_eta(
    raw_eta: float,
    graph: MapGraph,
    current: str,
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
    map_profile: MapProfile | None,
) -> float:
    """Never let a failed path estimate collapse into inf-based force delivery."""
    if isfinite(raw_eta):
        return raw_eta
    if not current:
        return map_profile.best_route_cost if map_profile else float("inf")
    bare = estimate_delivery_route(
        graph, current, player, gate_node_id, terminal_node_ids,
        None, process_nodes, processed_node_ids, set(),
    )
    if isfinite(bare.cost):
        return bare.cost
    if map_profile:
        return map_profile.best_route_cost
    return float("inf")


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
    if resource_type == "ICE_BOX" and get_freshness(player) > 65:
        if not (map_profile and map_profile.mountain_edge_ratio >= 0.25 and plan.direct_eta > 180):
            return -8.0
    if resource_type == "ICE_BOX" and get_freshness(player) > 75 and plan.direct_eta < 180:
        return -8.0
    favors_water = bool(map_profile and map_profile.favors_water)
    if resource_type == "BOAT_RIGHT" and route and route.water_ratio < 0.25 and not favors_water:
        return -4.0
    if resource_type in ("OFFICIAL_PERMIT", "PASS_TOKEN"):
        permits = get_player_resources(player).get("OFFICIAL_PERMIT", 0) + get_player_resources(player).get("PASS_TOKEN", 0)
        if permits >= 2:
            return -5.0
        if plan.should_force_delivery or plan.combat_weight < 1.25 or plan.direct_eta < 200:
            return -6.0
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


def build_task_sequence_plan(
    round_num: int,
    player: dict,
    graph: MapGraph,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    blocked_nodes: set[str] | None,
    process_nodes: dict[str, dict] | None,
    tasks: list[dict],
    player_id: int,
    failed_task_ids: set[str],
    visited_node_ids: set[str],
    obstacle_nodes: set[str],
    map_profile: MapProfile | None,
    task_weight: float,
    should_force_delivery: bool,
) -> TaskSequencePlan | None:
    """Build a rolling 15-20 frame multi-task route plan.

    Each frame re-evaluates the best short sequence (e.g. three 30-point tasks)
    and exposes only the first task as the immediate target.
    """
    current = get_current_node_id(player) or ""
    if not current:
        return None

    goal = _delivery_goal_node(player, gate_node_id, terminal_node_ids, graph, current, weather, blocked_nodes, process_nodes)
    if not goal:
        return None

    horizon = _planning_horizon(map_profile, should_force_delivery)
    candidates = _filter_sequence_task_candidates(
        tasks, player_id, round_num, failed_task_ids, visited_node_ids,
        obstacle_nodes, player,
    )
    if not candidates:
        return None

    score_gap = max(0, TASK_SCORE_TARGET - get_task_score(player))
    max_len = min(
        MAX_SEQUENCE_TASKS,
        len(candidates),
        max(1, (score_gap + 29) // 30),
    )

    best: TaskSequencePlan | None = None
    for length in range(1, max_len + 1):
        greedy_seq = _greedy_task_sequence(
            candidates, length, current, goal, graph, weather,
            blocked_nodes, process_nodes, round_num, horizon,
        )
        best = _pick_better_sequence(
            best,
            _wrap_task_sequence(
                greedy_seq, current, goal, graph, weather, blocked_nodes,
                process_nodes, round_num, task_weight, should_force_delivery,
                horizon, "greedy",
            ),
        )

    top = sorted(
        candidates,
        key=lambda t: (
            -_task_profile(get_task_template_id(t))[0],
            _weighted_path_frames(graph, current, t.get("nodeId", ""), weather, blocked_nodes, process_nodes),
        ),
    )[:MAX_SEQUENCE_PERMUTE]
    if len(top) >= 2:
        for length in range(2, min(max_len + 1, len(top) + 1)):
            for perm in permutations(top, length):
                best = _pick_better_sequence(
                    best,
                    _wrap_task_sequence(
                        list(perm), current, goal, graph, weather, blocked_nodes,
                        process_nodes, round_num, task_weight, should_force_delivery,
                        horizon, "permute",
                    ),
                )

    if best is None or best.net_value < MIN_TASK_NET_VALUE:
        return None
    return best


def _pick_better_sequence(
    current: TaskSequencePlan | None,
    candidate: TaskSequencePlan | None,
) -> TaskSequencePlan | None:
    if candidate is None:
        return current
    if current is None or candidate.net_value > current.net_value:
        return candidate
    return current


def _wrap_task_sequence(
    sequence: list[dict],
    current: str,
    goal: str,
    graph: MapGraph,
    weather: dict | None,
    blocked_nodes: set[str] | None,
    process_nodes: dict[str, dict] | None,
    round_num: int,
    task_weight: float,
    should_force_delivery: bool,
    horizon: float,
    method: str,
) -> TaskSequencePlan | None:
    if not sequence:
        return None
    metrics = _evaluate_task_sequence_metrics(
        sequence, current, goal, graph, weather, blocked_nodes,
        process_nodes, round_num, horizon,
    )
    if metrics is None:
        return None
    total_cost, total_score, extra_delivery_cost = metrics
    time_penalty = 0.85 if should_force_delivery else 0.65
    net_value = (
        total_score * task_weight
        - total_cost * time_penalty
        - extra_delivery_cost * 0.75
    )
    first = sequence[0]
    return TaskSequencePlan(
        task_ids=tuple(t.get("taskId", "") for t in sequence),
        task_nodes=tuple(t.get("nodeId", "") for t in sequence),
        total_score=total_score,
        total_cost=total_cost,
        extra_delivery_cost=extra_delivery_cost,
        net_value=net_value,
        next_task_id=first.get("taskId", ""),
        next_task_node=first.get("nodeId", ""),
        horizon=horizon,
        reason=f"{method}_len{len(sequence)}",
    )


def _filter_sequence_task_candidates(
    tasks: list[dict],
    player_id: int,
    round_num: int,
    failed_task_ids: set[str],
    visited_node_ids: set[str],
    obstacle_nodes: set[str],
    player: dict,
) -> list[dict]:
    candidates: list[dict] = []
    for task in tasks:
        if not task.get("active", False) or task.get("completed", False) or task.get("failed", False):
            continue
        owner = task.get("ownerPlayerId", 0)
        if owner not in (0, player_id):
            continue
        protection = task.get("protectionPlayerId", 0)
        if protection not in (0, player_id):
            continue
        task_id = task.get("taskId", "")
        if task_id in failed_task_ids:
            continue
        task_node = task.get("nodeId", "")
        if not task_node:
            continue
        if task_node in visited_node_ids:
            tid = get_task_template_id(task)
            if not (tid.startswith("T04") and task_node in obstacle_nodes):
                continue
        tid = get_task_template_id(task)
        if tid.startswith("T06") and not has_resource(player, "FAST_HORSE") and not has_resource(player, "SHORT_HORSE"):
            continue
        expire_round = int(task.get("expireRound", 0) or 0)
        if expire_round > 0 and round_num >= expire_round:
            continue
        candidates.append(task)
    return candidates


def _greedy_task_sequence(
    candidates: list[dict],
    length: int,
    current: str,
    goal: str,
    graph: MapGraph,
    weather: dict | None,
    blocked_nodes: set[str] | None,
    process_nodes: dict[str, dict] | None,
    round_num: int,
    horizon: float,
) -> list[dict]:
    pool = list(candidates)
    selected: list[dict] = []
    cursor = current
    elapsed = 0.0
    for _ in range(length):
        best_task: dict | None = None
        best_ratio = float("-inf")
        for task in pool:
            node = task.get("nodeId", "")
            travel = _weighted_path_frames(graph, cursor, node, weather, blocked_nodes, process_nodes)
            if isinf(travel):
                continue
            _, process_rounds, _ = _task_profile(get_task_template_id(task))
            expire_round = int(task.get("expireRound", 0) or 0)
            projected = round_num + elapsed + travel + process_rounds
            if expire_round and projected >= expire_round:
                continue
            if elapsed + travel + process_rounds > horizon:
                continue
            score, _, _ = _task_profile(get_task_template_id(task))
            ratio = score / max(1.0, travel + process_rounds)
            if ratio > best_ratio:
                best_ratio = ratio
                best_task = task
        if best_task is None:
            break
        node = best_task.get("nodeId", "")
        travel = _weighted_path_frames(graph, cursor, node, weather, blocked_nodes, process_nodes)
        _, process_rounds, _ = _task_profile(get_task_template_id(best_task))
        elapsed += travel + process_rounds
        selected.append(best_task)
        pool.remove(best_task)
        cursor = node
    return selected


def _evaluate_task_sequence_metrics(
    sequence: list[dict],
    current: str,
    goal: str,
    graph: MapGraph,
    weather: dict | None,
    blocked_nodes: set[str] | None,
    process_nodes: dict[str, dict] | None,
    round_num: int,
    horizon: float,
) -> tuple[float, int, float] | None:
    cursor = current
    total_cost = 0.0
    total_score = 0
    blocked = blocked_nodes or set()
    for task in sequence:
        node = task.get("nodeId", "")
        travel = _weighted_path_frames(graph, cursor, node, weather, blocked, process_nodes)
        if isinf(travel):
            return None
        score, process_rounds, _ = _task_profile(get_task_template_id(task))
        expire_round = int(task.get("expireRound", 0) or 0)
        projected = round_num + total_cost + travel + process_rounds
        if expire_round and projected >= expire_round:
            return None
        total_cost += travel + process_rounds
        total_score += score
        cursor = node
    if total_cost > horizon:
        return None
    direct = _weighted_path_frames(graph, current, goal, weather, blocked, process_nodes)
    via_tasks = total_cost + _weighted_path_frames(graph, cursor, goal, weather, blocked, process_nodes)
    if isinf(direct) or isinf(via_tasks):
        return None
    extra_delivery_cost = max(0.0, via_tasks - direct)
    return total_cost, total_score, extra_delivery_cost


def _planning_horizon(map_profile: MapProfile | None, should_force_delivery: bool) -> float:
    """Adaptive rolling window length (15-20 frames by default)."""
    horizon = DEFAULT_PLANNING_HORIZON
    if map_profile:
        horizon += max(-2.0, min(4.0, (380.0 - map_profile.best_route_cost) / 80.0))
    if should_force_delivery:
        horizon *= 0.55
    return max(12.0, min(22.0, horizon))


def _weighted_path_frames(
    graph: MapGraph,
    start: str,
    end: str,
    weather: dict | None,
    blocked_nodes: set[str] | None,
    process_nodes: dict[str, dict] | None,
) -> float:
    if not start or not end:
        return float("inf")
    if start == end:
        return 0.0
    blocked = blocked_nodes or set()
    path = _find_route_segment(graph, start, end, weather, blocked, process_nodes)
    if not path:
        return float("inf")
    return _path_frames_cost(graph, path, weather, blocked, process_nodes)


def _delivery_goal_node(
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    graph: MapGraph,
    current: str,
    weather: dict | None,
    blocked_nodes: set[str] | None,
    process_nodes: dict[str, dict] | None,
) -> str:
    if not is_verified(player) and gate_node_id:
        return gate_node_id
    for terminal in terminal_node_ids or []:
        return terminal
    if gate_node_id:
        return gate_node_id
    return current


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
