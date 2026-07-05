"""Strategy: full decision engine implementing 策略设计文档 L2–L8.

Priority order per frame (策略文档 §14 伪代码):
  P0  Stable online, every-frame heartbeat, zero illegal actions
  P1  Must deliver (goodFruit>0, freshness>0, verified, at S15)
  P2  Task score ≥ 90
  P3  Deliver early (time score)
  P4  Preserve good fruit & freshness
  P5  Moderate combat (guard/break/squad) without sacrificing P1–P4
"""

from __future__ import annotations

import logging
import math
from typing import Any

from lychee_client.map_graph import MapGraph, ROUTE_FRESHNESS_LOSS, PROCESS_COST_FRAMES
from lychee_client.map_gameplay import MapGameplayContext, MapStrategyProfile, default_map_gameplay
from lychee_client.state import (
    can_move, can_act, get_current_node_id, needs_processing,
    is_delivered, is_retired, is_verified, is_at_node, is_in_passive_state,
    is_in_limited_state,
    find_available_resources, find_task_at_node, get_enemy_busy_task_ids,
    get_good_fruit, get_bad_fruit, get_freshness,
    get_player_resources, has_resource, get_squad_count,
    get_action_points, get_task_score, get_blocked_nodes,
    classify_opponent_mode, get_team_id, get_task_template_id,
    is_task_available, get_task_point_value,
    is_verify_process, is_enemy_guard, is_own_guard, guard_is_active, node_has_obstacle,
    TASK_SCORE_TARGET, TASK_SCORE_STRETCH, MAX_TASK_DETOUR_COST, MAX_TASK_DETOUR_CEILING,
    ROUTE_TASK_BONUS_PER_SCORE, ROUTE_TASK_COUNT_BONUS, ROUTE_HIGH_VALUE_TASK_BONUS,
    ROUTE_VISITED_BACKTRACK_PENALTY,
    ROUTE_BUCKET_BONUS_PER_SCORE, NEAR_GATE_RESOURCE_HOPS,
    HORSE_USE_MIN_HOP_COST, HORSE_USE_MIN_HOP_COST_EARLY,
    ICE_BOX_FRESHNESS_THRESHOLD, RUSH_PROTECT_FRESHNESS,
    RESOURCE_CLAIM_PRIORITY, TASK_PRIORITY, MAX_ROUND,
    SQUAD_CLEAR_COST, SQUAD_RESERVE_FOR_LATE, SQUAD_CLEAR_MIN_SQUAD,
    SQUAD_CLEAR_NEXT_HOP_MIN_SQUAD,
    GATE_CORRIDOR_NODES, GATE_CORRIDOR_ORDER, GATE_CORRIDOR_SQUAD_MIN,
    MAX_ACTIVE_GUARDS, GUARD_GOOD_FRUIT_RESERVE, GUARD_MIN_LEAD_FIRST,
    GUARD_STUCK_AVOID_ROUNDS, GUARD_SILENT_WAIT_LIMIT,
    GUARD_RESERVE_FOR_GATE, FINAL_CORRIDOR_GATE_HOPS,
    FINAL_CORRIDOR_GUARD_MIN_LEAD, FINAL_CORRIDOR_GUARD_TASK_MIN,
    ICE_BOX_NEAR_GATE_HOPS, GATE_ENTRY_DEADLINE_ROUND, GATE_ARRIVAL_TARGET_ROUND,
    EARLY_GAME_MAX_ROUND,
    SQUAD_CLEAR_MIN_ROUND, SQUAD_CLEAR_MIDMAP_MIN_ROUND, LATE_GAME_NO_MID_TASK_ROUND,
    SCOUT_MARKER_VALID_FRAMES, SQUAD_SCOUT_MIN_DELAY, SQUAD_SCOUT_MAX_DELAY,
    FORCE_DELIVERY_ETA_BUFFER, FORCE_DELIVERY_MIN_REMAINING,
    FORCE_DELIVERY_ETA_REMAINING_MAX, FORCE_DELIVERY_LATE_REMAINING,
    RUSH_TASK_DETOUR_BONUS, DELIVERY_CRITICAL_SLACK_MULT,
    WATER_ROUTE_TASK_MIN,
    TASK_DETOUR_SLACK_RESERVE, WATER_ROUTE_NAV_PENALTY, INTEL_MAX_DISTANCE,
)
from lychee_client.decision import (
    make_action, make_move_action, make_wait_action,
    make_process_action, make_dock_action, make_verify_gate_action,
    make_empty_action, make_window_card_action,
    make_claim_resource_action, make_claim_task_action,
    make_deliver_action, make_break_guard_action,
    make_forced_pass_action, make_clear_action, make_set_guard_action,
    make_use_resource_action,
    make_squad_scout_action, make_squad_clear_action,
    make_squad_reinforce_action, make_squad_weaken_action,
    make_rush_protect_action, make_rush_speed_action,
)

logger = logging.getLogger("lychee_client.strategy")


def _map_ctx(map_gameplay: MapGameplayContext | None) -> MapGameplayContext:
    return map_gameplay if map_gameplay is not None else default_map_gameplay()


def _profile(map_gameplay: MapGameplayContext | None) -> MapStrategyProfile:
    return _map_ctx(map_gameplay).profile


def _ice_threshold(map_gameplay: MapGameplayContext | None) -> float:
    prof = _profile(map_gameplay)
    threshold = prof.ice_box_freshness_threshold
    if threshold > 0:
        return threshold
    return ICE_BOX_FRESHNESS_THRESHOLD


def _ice_rush_threshold(map_gameplay: MapGameplayContext | None) -> float:
    prof = _profile(map_gameplay)
    threshold = getattr(prof, "ice_box_rush_use_threshold", 0.0)
    if threshold > 0:
        return threshold
    return 95.0


def _water_task_min(map_gameplay: MapGameplayContext | None) -> int:
    prof = _profile(map_gameplay)
    if prof.water_route_task_min > 0:
        return prof.water_route_task_min
    return WATER_ROUTE_TASK_MIN


def _force_delivery_buffer(map_gameplay: MapGameplayContext | None) -> float:
    prof = _profile(map_gameplay)
    if prof.force_delivery_slack_buffer > 0:
        return prof.force_delivery_slack_buffer
    return FORCE_DELIVERY_ETA_BUFFER


def _gate_verify_frames(process_nodes: dict[str, dict] | None, gate_node_id: str) -> int:
    if process_nodes and gate_node_id:
        info = process_nodes.get(gate_node_id, {})
        pt = info.get("processType", "")
        if is_verify_process(pt):
            return int(info.get("processRound") or PROCESS_COST_FRAMES.get("VERIFY", 6) or 6)
    return int(PROCESS_COST_FRAMES.get("VERIFY", 6) or 6)


def _must_wait_for_gate_verify(
    player: dict,
    gate_node_id: str,
    current_node_id: str | None,
    last_move_failed: bool = False,
    last_move_error: str = "",
) -> bool:
    """True when still at the gate and server/client agree verification is pending."""
    if not gate_node_id or not current_node_id or current_node_id != gate_node_id:
        return False
    if last_move_failed and last_move_error == "VERIFY_REQUIRED":
        return True
    return not is_verified(player)


def _is_delivery_critical(
    round_num: int,
    player: dict,
    delivery_slack: float,
    force_buffer: float,
    gate_node_id: str,
    current_node_id: str | None,
) -> bool:
    """Delivery-first mode: skip tasks/combat/rush_speed until verified."""
    if is_verified(player) or is_delivered(player):
        return False
    if round_num < EARLY_GAME_MAX_ROUND:
        return False
    if round_num >= GATE_ENTRY_DEADLINE_ROUND:
        return True
    if (
        gate_node_id
        and current_node_id
        and round_num >= GATE_ARRIVAL_TARGET_ROUND
        and not is_at_node(player, gate_node_id)
    ):
        return True
    return delivery_slack <= force_buffer * DELIVERY_CRITICAL_SLACK_MULT


def _make_verify_gate_with_tactic(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    current_node_id: str,
    gate_node_id: str,
    inquire_nodes: list[dict],
    my_team_id: str,
    process_nodes: dict[str, dict] | None,
    verify_gate_plain_only: bool = False,
) -> dict:
    """Build VERIFY_GATE; attach BREAK_ORDER for contested or time-sensitive verification."""
    action = make_verify_gate_action(current_node_id)
    if verify_gate_plain_only:
        return make_action(match_id, round_num, player_id, [action])

    rush_used = int(player.get("rushTacticUsedCount", 0) or 0)
    remaining = max(0, MAX_ROUND - round_num)
    verify_frames = _gate_verify_frames(process_nodes, gate_node_id)

    use_break = False
    for node in inquire_nodes:
        if node.get("nodeId") == gate_node_id:
            guard = node.get("guard")
            if is_enemy_guard(guard, my_team_id, player_id):
                use_break = True
            break
    # BREAK_ORDER also legally shortens gate verification by 3 frames; use it when
    # delivery timing is tight even if the gate itself is not guarded.
    use_break = use_break or remaining <= verify_frames + 12

    if (
        use_break
        and rush_used == 0
        and remaining >= verify_frames + 2
        and (get_bad_fruit(player) >= 2 or get_good_fruit(player) >= 2)
    ):
        action["rushTactic"] = "BREAK_ORDER"
    return make_action(match_id, round_num, player_id, [action])


def _break_guard_investment(player: dict, reserve_good_fruit: int = 1) -> tuple[int, int]:
    """Choose BREAK_GUARD fruit while preserving at least one good fruit for delivery."""
    bad = min(get_bad_fruit(player), 2)
    spendable_good = max(0, get_good_fruit(player) - reserve_good_fruit)
    good = min(spendable_good, 2)
    return good, bad


def _has_move_speed_buff(player: dict) -> bool:
    for buff in player.get("buffs", []) or []:
        if not isinstance(buff, dict):
            continue
        bt = str(buff.get("buffType") or buff.get("type") or "").upper()
        if "HORSE" in bt or bt in ("FAST_HORSE", "SHORT_HORSE", "RUSH_SPEED", "MOVE_SPEED"):
            return True
    return False


def _make_process_action(
    match_id: str,
    round_num: int,
    player_id: int,
    process_type: str,
    current_node_id: str,
    phase: str,
) -> dict:
    """Map processType to the correct protocol action."""
    if process_type == "BOARD":
        return make_action(match_id, round_num, player_id, [make_dock_action(current_node_id)])
    if is_verify_process(process_type):
        if phase == "RUSH":
            return make_action(match_id, round_num, player_id, [make_verify_gate_action(current_node_id)])
        return make_empty_action(match_id, round_num, player_id)
    return make_action(match_id, round_num, player_id, [make_process_action(current_node_id)])


def _append_squad_action(
    action_msg: dict,
    squad_action: dict | None,
) -> dict:
    if squad_action is None:
        return action_msg
    actions = action_msg.get("msg_data", {}).get("actions", [])
    if len(actions) >= 2:
        return action_msg
    if len(actions) == 1:
        actions = actions + [squad_action]
    else:
        actions = [squad_action]
    action_msg["msg_data"]["actions"] = actions
    return action_msg


def _find_gate_corridor_obstacle(
    inquire_nodes: list[dict],
    squad_clear_pending: set[str],
    graph: MapGraph | None = None,
    current_node_id: str = "",
    gate_node_id: str = "",
    weather: dict | None = None,
    map_gameplay: MapGameplayContext | None = None,
) -> str | None:
    """Return first obstacle on S10-S14 corridor, then on planned path candidates."""
    obstacle_at: set[str] = set()
    for node in inquire_nodes:
        nid = node.get("nodeId", "")
        if nid and node_has_obstacle(node):
            obstacle_at.add(nid)
    for nid in GATE_CORRIDOR_ORDER:
        if nid in GATE_CORRIDOR_NODES and nid in obstacle_at and nid not in squad_clear_pending:
            return nid

    ctx = _map_ctx(map_gameplay)
    if not graph or not current_node_id or not gate_node_id or not ctx.obstacle_candidate_node_ids:
        return None
    path = graph.shortest_path(current_node_id, gate_node_id, weather, obstacle_at)
    for nid in path:
        if (
            nid in ctx.obstacle_candidate_node_ids
            and nid in obstacle_at
            and nid not in squad_clear_pending
        ):
            return nid
    return None


def _maybe_append_gate_corridor_squad_clear(
    action_msg: dict,
    player: dict,
    phase: str,
    inquire_nodes: list[dict],
    squad_clear_pending: set[str],
    round_num: int,
    graph: MapGraph | None = None,
    current_node_id: str = "",
    gate_node_id: str = "",
    weather: dict | None = None,
    map_gameplay: MapGameplayContext | None = None,
    force_delivery: bool = False,
    obstacle_nodes: set[str] | None = None,
    tasks: list[dict] | None = None,
    failed_task_ids: set[str] | None = None,
    opp_player: dict | None = None,
) -> dict:
    """Append SQUAD_CLEAR for own-route obstacles (incl. force delivery) without blocking main action."""
    if phase == "RUSH":
        return action_msg
    if is_retired(player) or is_delivered(player):
        return action_msg
    if is_in_passive_state(player) or player.get("state") == "CONTESTING":
        return action_msg
    if get_squad_count(player) < GATE_CORRIDOR_SQUAD_MIN:
        return action_msg
    actions = action_msg.get("msg_data", {}).get("actions", [])
    if len(actions) >= 2:
        return action_msg
    if any(a.get("action", "").startswith("SQUAD_") for a in actions):
        return action_msg
    if not graph or not current_node_id or not gate_node_id:
        return action_msg
    if obstacle_nodes is None:
        obstacle_nodes = set()
    if tasks is None:
        tasks = []
    if failed_task_ids is None:
        failed_task_ids = set()
    goal = gate_node_id
    clear_target = _find_squad_clear_target(
        graph, current_node_id, goal, inquire_nodes, obstacle_nodes,
        weather, opp_player, tasks, failed_task_ids,
        get_task_score(player), get_squad_count(player), squad_clear_pending,
        map_gameplay, round_num=round_num, force_delivery=force_delivery,
        gate_node_id=gate_node_id,
    )
    if not clear_target:
        return action_msg
    logger.info(
        "Round %d: SQUAD_CLEAR append at %s (force_delivery=%s)",
        round_num, clear_target, force_delivery,
    )
    return _append_squad_action(action_msg, make_squad_clear_action(clear_target))


def decide_action(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node: dict | None = None,
    process_nodes: dict[str, dict] | None = None,
    contests: list[dict] | None = None,
    events: list[dict] | None = None,
    active_contest_id: str = "",
    last_move_failed: bool = False,
    last_move_error: str = "",
    gate_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    tasks: list[dict] | None = None,
    phase: str = "",
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    weather: dict | None = None,
    all_players: list[dict] | None = None,
    inquire_nodes: list[dict] | None = None,
    failed_task_ids: set[str] | None = None,
    rush_speed_failed: bool = False,
    guard_blocked_targets: set[str] | None = None,
    avoid_route_nodes: set[str] | None = None,
    pending_task_hold_task_id: str = "",
    pending_task_hold_node_id: str = "",
    pending_task_hold_until_round: int = 0,
    forced_pass_failed_targets: set[str] | None = None,
    squad_clear_pending: set[str] | None = None,
    guard_stuck_rounds: int = 0,
    guard_stuck_target: str = "",
    own_guard_sites: set[str] | None = None,
    map_gameplay: MapGameplayContext | None = None,
    task_claimed_this_stop: bool = False,
    verify_gate_plain_only: bool = False,
) -> dict:
    """Decide the action for the current round.

    Implements the single-frame decision pseudocode from 策略文档 §14.
    Returns a complete action message dict.
    """
    # Defaults
    if terminal_node_ids is None:
        terminal_node_ids = []
    if tasks is None:
        tasks = []
    if processed_node_ids is None:
        processed_node_ids = set()
    if visited_node_ids is None:
        visited_node_ids = set()
    if weather is None:
        weather = {}
    if all_players is None:
        all_players = []
    if inquire_nodes is None:
        inquire_nodes = []
    if failed_task_ids is None:
        failed_task_ids = set()
    if guard_blocked_targets is None:
        guard_blocked_targets = set()
    if avoid_route_nodes is None:
        avoid_route_nodes = set()
    if forced_pass_failed_targets is None:
        forced_pass_failed_targets = set()
    if squad_clear_pending is None:
        squad_clear_pending = set()
    if own_guard_sites is None:
        own_guard_sites = set()

    try:
        action_msg = _decide_action_impl(
            match_id, round_num, player_id, player, graph,
            current_node, process_nodes, contests, events,
            active_contest_id, last_move_failed, last_move_error,
            gate_node_id, terminal_node_ids, tasks, phase,
            processed_node_ids, visited_node_ids, weather, all_players, inquire_nodes,
            failed_task_ids, rush_speed_failed, guard_blocked_targets, avoid_route_nodes,
            pending_task_hold_task_id, pending_task_hold_node_id, pending_task_hold_until_round,
            forced_pass_failed_targets, squad_clear_pending,
            guard_stuck_rounds, guard_stuck_target, own_guard_sites,
            map_gameplay, task_claimed_this_stop,
            verify_gate_plain_only,
        )
        current_node_id = get_current_node_id(player) or ""
        force_buffer = _force_delivery_buffer(map_gameplay)
        delivery_slack = _delivery_slack_frames(
            round_num, player, graph, current_node_id,
            gate_node_id, terminal_node_ids, weather,
            process_nodes, processed_node_ids,
            map_gameplay=map_gameplay,
        )
        append_force_delivery = _should_force_delivery(
            round_num, phase, player, graph, current_node_id,
            gate_node_id, terminal_node_ids, weather,
            process_nodes, processed_node_ids,
            map_gameplay=map_gameplay,
        )
        if _is_delivery_critical(
            round_num, player, delivery_slack, force_buffer,
            gate_node_id, current_node_id,
        ):
            append_force_delivery = True
        opp_player = None
        for p in all_players:
            if p.get("playerId") != player_id:
                opp_player = p
                break
        obstacle_nodes: set[str] = set()
        my_team_id = get_team_id(player)
        for node in inquire_nodes:
            nid = node.get("nodeId", "")
            if not nid or not node_has_obstacle(node):
                continue
            guard = node.get("guard") if isinstance(node.get("guard"), dict) else {}
            if is_enemy_guard(guard, my_team_id, player_id):
                continue
            obstacle_nodes.add(nid)
        return _maybe_append_gate_corridor_squad_clear(
            action_msg, player, phase, inquire_nodes, squad_clear_pending, round_num,
            graph, current_node_id, gate_node_id, weather, map_gameplay,
            force_delivery=append_force_delivery,
            obstacle_nodes=obstacle_nodes,
            tasks=tasks,
            failed_task_ids=failed_task_ids,
            opp_player=opp_player,
        )
    except Exception as e:
        logger.error("Round %d: Strategy error: %s", round_num, e, exc_info=True)
        return make_empty_action(match_id, round_num, player_id)


def _decide_action_impl(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node: dict | None = None,
    process_nodes: dict[str, dict] | None = None,
    contests: list[dict] | None = None,
    events: list[dict] | None = None,
    active_contest_id: str = "",
    last_move_failed: bool = False,
    last_move_error: str = "",
    gate_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    tasks: list[dict] | None = None,
    phase: str = "",
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    weather: dict | None = None,
    all_players: list[dict] | None = None,
    inquire_nodes: list[dict] | None = None,
    failed_task_ids: set[str] | None = None,
    rush_speed_failed: bool = False,
    guard_blocked_targets: set[str] | None = None,
    avoid_route_nodes: set[str] | None = None,
    pending_task_hold_task_id: str = "",
    pending_task_hold_node_id: str = "",
    pending_task_hold_until_round: int = 0,
    forced_pass_failed_targets: set[str] | None = None,
    squad_clear_pending: set[str] | None = None,
    guard_stuck_rounds: int = 0,
    guard_stuck_target: str = "",
    own_guard_sites: set[str] | None = None,
    map_gameplay: MapGameplayContext | None = None,
    task_claimed_this_stop: bool = False,
    verify_gate_plain_only: bool = False,
) -> dict:
    if guard_blocked_targets is None:
        guard_blocked_targets = set()
    if avoid_route_nodes is None:
        avoid_route_nodes = set()
    if forced_pass_failed_targets is None:
        forced_pass_failed_targets = set()
    if squad_clear_pending is None:
        squad_clear_pending = set()
    if own_guard_sites is None:
        own_guard_sites = set()

    # --- P0: Stability ---
    if is_retired(player) or is_delivered(player):
        return make_empty_action(match_id, round_num, player_id)

    state = player.get("state", "")
    current_node_id = get_current_node_id(player)
    my_team_id = get_team_id(player)

    # Passive states: PROCESSING, VERIFYING, FORCED_PASSING, RESTING → heartbeat
    if is_in_passive_state(player):
        return make_empty_action(match_id, round_num, player_id)

    blocked = get_blocked_nodes(inquire_nodes, my_team_id, player_id)
    route_blocked = set(blocked)
    route_blocked.update(guard_blocked_targets)
    route_blocked.update(avoid_route_nodes)
    opp_player = _find_opponent(all_players, player_id)
    enemy_busy_task_ids = get_enemy_busy_task_ids(all_players, player_id)
    mode = classify_opponent_mode(player, opp_player, phase)

    obstacle_nodes: set[str] = set()
    for node in inquire_nodes:
        nid = node.get("nodeId", "")
        if not nid or not node_has_obstacle(node):
            continue
        guard = node.get("guard") if isinstance(node.get("guard"), dict) else {}
        if is_enemy_guard(guard, my_team_id, player_id):
            continue
        obstacle_nodes.add(nid)
    delivery_slack = _delivery_slack_frames(
        round_num, player, graph, current_node_id,
        gate_node_id, terminal_node_ids, weather,
        process_nodes, processed_node_ids,
        route_blocked=route_blocked,
        map_gameplay=map_gameplay,
    )
    max_task_detour = _max_task_detour_budget(delivery_slack, phase, round_num)
    force_buffer = _force_delivery_buffer(map_gameplay)
    force_delivery = _should_force_delivery(
        round_num, phase, player, graph, current_node_id,
        gate_node_id, terminal_node_ids, weather,
        process_nodes, processed_node_ids,
        route_blocked=route_blocked,
        map_gameplay=map_gameplay,
    )
    delivery_critical = _is_delivery_critical(
        round_num, player, delivery_slack, force_buffer,
        gate_node_id, current_node_id,
    )
    if delivery_critical:
        force_delivery = True
    guard_wait_kwargs = {
        "force_delivery": force_delivery,
        "graph": graph,
        "gate_node_id": gate_node_id,
        "terminal_node_ids": terminal_node_ids,
        "weather": weather,
        "route_blocked": route_blocked,
        "avoid_route_nodes": avoid_route_nodes,
        "guard_blocked_targets": guard_blocked_targets,
        "guard_stuck_rounds": guard_stuck_rounds,
        "guard_stuck_target": guard_stuck_target,
        "process_nodes": process_nodes,
        "processed_node_ids": processed_node_ids,
    }

    if state != "CONTESTING" and _find_contest_id(player_id, contests, None, ""):
        on_water_route = _is_on_water_route(graph, current_node_id, gate_node_id, terminal_node_ids)
        return _handle_contesting(
            match_id, round_num, player_id, player,
            contests, events, active_contest_id, player,
            all_players, phase, on_water_route,
            graph=graph, gate_node_id=gate_node_id,
            terminal_node_ids=terminal_node_ids, obstacle_nodes=obstacle_nodes,
        )

    if state != "CONTESTING" and not is_in_passive_state(player):
        if not _should_defer_ice_box(player, last_move_error, route_blocked, guard_blocked_targets, avoid_route_nodes):
            ice_use = _try_use_ice_box(
                match_id, round_num, player_id, player, map_gameplay,
                phase=phase, force_delivery=force_delivery,
                graph=graph, current_node_id=current_node_id,
                weather=weather, gate_node_id=gate_node_id,
            )
            if ice_use is not None:
                return ice_use

    if state == "CONTESTING":
        on_water_route = _is_on_water_route(graph, current_node_id, gate_node_id, terminal_node_ids)
        return _handle_contesting(
            match_id, round_num, player_id, player,
            contests, events, active_contest_id, player,
            all_players, phase, on_water_route,
            graph=graph, gate_node_id=gate_node_id,
            terminal_node_ids=terminal_node_ids, obstacle_nodes=obstacle_nodes,
        )

    if is_in_limited_state(player):
        guard_target = _resolve_guard_block_target(player, route_blocked, guard_blocked_targets)

        if state == "WAITING":
            next_node = player.get("nextNodeId", "")
            if _must_wait_for_gate_verify(
                player, gate_node_id, current_node_id, last_move_failed, last_move_error,
            ):
                if last_move_failed and last_move_error == "OBJECT_BUSY":
                    logger.info(
                        "Round %d: gate verify busy at %s, WAIT before retry",
                        round_num, current_node_id,
                    )
                    return make_action(match_id, round_num, player_id, [make_wait_action()])
                # VERIFY_GATE is IDLE-only (任务书 §8.2); WAITING must hold until IDLE.
                if phase == "RUSH":
                    logger.info(
                        "Round %d: WAITING at unverified gate %s in RUSH, "
                        "WAIT for IDLE verify (last=%s)",
                        round_num, current_node_id, last_move_error or "pending",
                    )
                else:
                    logger.info(
                        "Round %d: WAITING at unverified gate %s until RUSH",
                        round_num, current_node_id,
                    )
                return make_action(match_id, round_num, player_id, [make_wait_action()])

            pending_process_type = _get_pending_station_process_type(
                current_node_id, next_node, process_nodes, processed_node_ids,
            )
            if pending_process_type:
                if _has_current_process_for_node(player, current_node_id):
                    logger.info("Round %d: station process running at %s, sending empty action", round_num, current_node_id)
                    return make_empty_action(match_id, round_num, player_id)
                if last_move_failed and last_move_error == "OBJECT_BUSY":
                    logger.info(
                        "Round %d: station process busy at %s, WAIT before retry",
                        round_num, current_node_id,
                    )
                    return make_action(match_id, round_num, player_id, [make_wait_action()])
                logger.info("Round %d: station process not started at %s, retrying %s", round_num, current_node_id, pending_process_type)
                return _make_process_action(
                    match_id, round_num, player_id,
                    pending_process_type, current_node_id, phase,
                )

            if last_move_failed and last_move_error == "PROCESS_REQUIRED":
                process_type = process_nodes.get(current_node_id, {}).get("processType") if process_nodes and current_node_id else ""
                if process_type:
                    logger.info("Round %d: PROCESS_REQUIRED in WAITING at %s, retrying %s", round_num, current_node_id, process_type)
                    return _make_process_action(
                        match_id, round_num, player_id,
                        process_type, current_node_id, phase,
                    )
                logger.info("Round %d: PROCESS_REQUIRED in WAITING at %s, sending WAIT", round_num, current_node_id)
                return make_action(match_id, round_num, player_id, [make_wait_action()])

            if not force_delivery and current_node_id and not next_node:
                if (
                    pending_task_hold_node_id == current_node_id
                    and round_num <= pending_task_hold_until_round
                    and not _task_hold_blocked_by_enemy(
                        pending_task_hold_task_id, enemy_busy_task_ids,
                    )
                ):
                    logger.info(
                        "Round %d: waiting for busy task at %s until %d",
                        round_num, current_node_id, pending_task_hold_until_round,
                    )
                    return make_action(match_id, round_num, player_id, [make_wait_action()])
                if pending_task_hold_node_id == current_node_id and pending_task_hold_task_id:
                    task_retry = _retry_task_at_current_node(
                        match_id, round_num, player_id, player, graph,
                        current_node_id, tasks, failed_task_ids,
                        preferred_task_id=pending_task_hold_task_id,
                        enemy_busy_task_ids=enemy_busy_task_ids,
                        map_gameplay=map_gameplay,
                    )
                    if task_retry is not None:
                        return task_retry
                task_retry = _retry_task_at_current_node(
                    match_id, round_num, player_id, player, graph,
                    current_node_id, tasks, failed_task_ids,
                    enemy_busy_task_ids=enemy_busy_task_ids,
                    map_gameplay=map_gameplay,
                )
                if task_retry is not None:
                    return task_retry

            if force_delivery and current_node_id and not next_node:
                move_action = _plan_limited_state_force_delivery_move(
                    match_id, round_num, player_id, player, graph,
                    current_node_id, gate_node_id, terminal_node_ids,
                    weather, process_nodes, processed_node_ids,
                    route_blocked, avoid_route_nodes,
                )
                if move_action is not None:
                    return move_action

            if guard_target:
                return _handle_limited_state_guard_block(
                    match_id, round_num, player_id, player, state,
                    guard_target, last_move_failed, last_move_error,
                    inquire_nodes, my_team_id, guard_wait_kwargs,
                )

            if last_move_failed and last_move_error in ("OBJECT_BUSY", "MOVING_ACTION_FORBIDDEN"):
                logger.info("Round %d: %s in WAITING, sending WAIT", round_num, last_move_error)
                return make_action(match_id, round_num, player_id, [make_wait_action()])

            if next_node:
                if next_node in guard_blocked_targets:
                    return _handle_limited_state_guard_block(
                        match_id, round_num, player_id, player, state,
                        next_node, last_move_failed, last_move_error,
                        inquire_nodes, my_team_id, guard_wait_kwargs,
                    )
                if next_node in avoid_route_nodes:
                    graph_obj = graph
                    detour = _find_delivery_detour_step(
                        graph_obj, current_node_id or "", player, gate_node_id,
                        terminal_node_ids, weather, process_nodes, processed_node_ids,
                        route_blocked, avoid_nodes=avoid_route_nodes,
                    )
                    if detour:
                        logger.info(
                            "Round %d: WAITING reroute via %s (avoid hop %s)",
                            round_num, detour, next_node,
                        )
                        return make_action(match_id, round_num, player_id, [make_move_action(detour)])
                    return make_action(match_id, round_num, player_id, [make_wait_action()])
                return make_action(match_id, round_num, player_id, [make_move_action(next_node)])
            if current_node_id:
                move_target = _find_move_target(
                    graph, current_node_id, player, gate_node_id, terminal_node_ids,
                    weather, route_blocked, obstacle_nodes=obstacle_nodes,
                    process_nodes=process_nodes,
                    processed_node_ids=processed_node_ids, visited_node_ids=visited_node_ids,
                    tasks=tasks, player_id=player_id, failed_task_ids=failed_task_ids,
                    enemy_busy_task_ids=enemy_busy_task_ids, phase=phase,
                    delivery_slack=delivery_slack, max_task_detour=max_task_detour,
                    map_gameplay=map_gameplay, round_num=round_num,
                )
                if move_target and move_target not in route_blocked:
                    return make_action(match_id, round_num, player_id, [make_move_action(move_target)])
                if move_target:
                    return _wait_and_weaken_guard(
                        match_id, round_num, player_id, player,
                        inquire_nodes, move_target, my_team_id,
                        **guard_wait_kwargs,
                    )

        if state == "MOVING":
            if _must_wait_for_gate_verify(
                player, gate_node_id, current_node_id, last_move_failed, last_move_error,
            ):
                logger.info(
                    "Round %d: MOVING at unverified gate %s, WAIT for IDLE verify",
                    round_num, current_node_id,
                )
                return make_action(match_id, round_num, player_id, [make_wait_action()])
            if guard_target or (
                last_move_failed
                and last_move_error in ("MOVE_BLOCKED_BY_GUARD", "MOVING_ACTION_FORBIDDEN")
            ):
                target = guard_target or player.get("nextNodeId", "")
                if target:
                    return _handle_limited_state_guard_block(
                        match_id, round_num, player_id, player, state,
                        target, last_move_failed, last_move_error,
                        inquire_nodes, my_team_id, guard_wait_kwargs,
                    )
            moving_action = _handle_moving(match_id, round_num, player_id, player, graph, weather, phase)
            if moving_action.get("msg_data", {}).get("actions"):
                return moving_action

        return make_empty_action(match_id, round_num, player_id)

    # Must be IDLE to act
    if not can_act(player):
        return make_empty_action(match_id, round_num, player_id)

    if current_node_id is None:
        return make_empty_action(match_id, round_num, player_id)

    ice_claim = None
    if not delivery_critical:
        ice_claim = _try_claim_ice_box(
            match_id, round_num, player_id, player, current_node,
        )
    if ice_claim is not None:
        return ice_claim

    if (
        not force_delivery
        and pending_task_hold_node_id == current_node_id
        and round_num <= pending_task_hold_until_round
        and not _task_hold_blocked_by_enemy(
            pending_task_hold_task_id, enemy_busy_task_ids,
        )
    ):
        logger.info(
            "Round %d: holding at %s for busy task until %d",
            round_num, current_node_id, pending_task_hold_until_round,
        )
        return make_action(match_id, round_num, player_id, [make_wait_action()])
    if (
        not force_delivery
        and pending_task_hold_node_id == current_node_id
        and pending_task_hold_task_id
    ):
        task_retry = _retry_task_at_current_node(
            match_id, round_num, player_id, player, graph,
            current_node_id, tasks, failed_task_ids,
            preferred_task_id=pending_task_hold_task_id,
            enemy_busy_task_ids=enemy_busy_task_ids,
            map_gameplay=map_gameplay,
        )
        if task_retry is not None:
            return task_retry

    leave_guard_action = _try_leave_own_guard_node(
        match_id, round_num, player_id, player, graph,
        current_node_id, gate_node_id, terminal_node_ids,
        weather, route_blocked, avoid_route_nodes,
        process_nodes, processed_node_ids, own_guard_sites,
    )
    if leave_guard_action is not None:
        return leave_guard_action

    # Don't use blocked_nodes as hard filter in BFS — it causes TARGET_NOT_REACHABLE
    # Instead, use weighted routing to prefer unblocked paths
    blocked_soft = route_blocked  # used for weighted routing and combat

    # --- P1: Delivery flow (策略文档 §4.2 FSM) ---

    # At S15: DELIVER if verified and can deliver
    if current_node_id in terminal_node_ids:
        if is_verified(player) and get_good_fruit(player) > 0 and get_freshness(player) > 0:
            return make_action(match_id, round_num, player_id, [make_deliver_action()])
        # At S15 but not verified → go back to S14 (策略文档 §4.2: 无视设卡与障碍)
        if gate_node_id and not is_verified(player):
            # Direct move to S14 — guards/obstacles don't block this path
            step = graph.next_step_toward(current_node_id, gate_node_id, weather, None)
            if step:
                return make_action(match_id, round_num, player_id, [make_move_action(step)])
        # At S15, verified but no good fruit/freshness → WAIT (can't deliver)
        return make_empty_action(match_id, round_num, player_id)

    # At S14 (gate): VERIFY_GATE in RUSH phase (IDLE only)
    if gate_node_id and is_at_node(player, gate_node_id) and _must_wait_for_gate_verify(
        player, gate_node_id, current_node_id, last_move_failed, last_move_error,
    ):
        if phase == "RUSH":
            return _make_verify_gate_with_tactic(
                match_id, round_num, player_id, player, current_node_id,
                gate_node_id, inquire_nodes, my_team_id, process_nodes,
                verify_gate_plain_only=verify_gate_plain_only,
            )
        logger.info(
            "Round %d: at unverified gate %s before RUSH, waiting",
            round_num, current_node_id,
        )
        return make_action(match_id, round_num, player_id, [make_wait_action()])

    # --- Fixed processing (策略文档 §4.1: 再次到达同一站需重新处理) ---
    # Process at current node ONLY if not already processed this visit.
    # processed_node_ids tracks nodes where we completed processing this session.
    # If already processed, skip to MOVE (even if node has processType).
    already_processed_here = current_node_id in processed_node_ids
    process_type = None if already_processed_here else _get_process_type(current_node, process_nodes, current_node_id)
    if process_type and is_verify_process(process_type) and is_verified(player):
        process_type = None

    # 已验核且在宫门：优先前往终点交付
    if (
        gate_node_id
        and is_at_node(player, gate_node_id)
        and is_verified(player)
        and terminal_node_ids
    ):
        for tid in terminal_node_ids:
            if tid == current_node_id:
                continue
            step = graph.next_step_toward(current_node_id, tid, weather, None, use_weighted=True)
            if step:
                logger.info("Round %d: verified at gate, move to %s for delivery", round_num, step)
                return make_action(match_id, round_num, player_id, [make_move_action(step)])

    if process_type:
        if last_move_failed and last_move_error == "OBJECT_BUSY":
            logger.info(
                "Round %d: station process busy at %s, WAIT before retry",
                round_num, current_node_id,
            )
            return make_action(match_id, round_num, player_id, [make_wait_action()])
        if last_move_failed and "WINDOW" in last_move_error.upper():
            move_target = _find_move_target(
                graph, current_node_id, player, gate_node_id, terminal_node_ids,
                weather, route_blocked, obstacle_nodes=obstacle_nodes, process_nodes=process_nodes,
                processed_node_ids=processed_node_ids, visited_node_ids=visited_node_ids,
                tasks=tasks, player_id=player_id, failed_task_ids=failed_task_ids,
                enemy_busy_task_ids=enemy_busy_task_ids, phase=phase,
                delivery_slack=delivery_slack, max_task_detour=max_task_detour,
                map_gameplay=map_gameplay, round_num=round_num,
            )
            if move_target:
                return make_action(match_id, round_num, player_id, [make_move_action(move_target)])
            return make_empty_action(match_id, round_num, player_id)

        return _make_process_action(match_id, round_num, player_id, process_type, current_node_id, phase)

    if last_move_failed and last_move_error == "PROCESS_REQUIRED":
        process_type = _get_process_type(current_node, process_nodes, current_node_id)
        if process_type:
            logger.info("Round %d: PROCESS_REQUIRED at %s, sending %s", round_num, current_node_id, process_type)
            return _make_process_action(match_id, round_num, player_id, process_type, current_node_id, phase)
        return make_action(match_id, round_num, player_id, [make_process_action(current_node_id)])

    # --- Handle OBJECT_BUSY: wait one round and retry ---
    if last_move_failed and last_move_error == "OBJECT_BUSY":
        # The process target is busy (e.g., window contest just ended, still transitioning)
        # Wait one round, then retry process on next round
        logger.info("Round %d: OBJECT_BUSY, waiting", round_num)
        return make_action(match_id, round_num, player_id, [make_wait_action()])

    # --- Handle blocked movement ---
    if last_move_failed and last_move_error == "MOVE_BLOCKED_BY_GUARD":
        return _handle_blocked_by_guard(
            match_id, round_num, player_id, player, graph,
            current_node_id, gate_node_id, terminal_node_ids,
            weather, blocked_soft, inquire_nodes, process_nodes=process_nodes,
        )

    horse_use = _use_horse_immediately(
        match_id, round_num, player_id, player, current_node_id,
    )
    if horse_use is not None:
        return horse_use

    # --- Dual guard: only after task score target (unless gate fight) ---
    # (moved below task handling)

    # --- Gate corridor monitor: weaken / reroute before chasing tasks ---
    if can_act(player) and not force_delivery:
        gate_action = _try_gate_corridor_guard_strategy(
            match_id, round_num, player_id, player, graph,
            current_node_id, gate_node_id, terminal_node_ids, weather,
            inquire_nodes, my_team_id, route_blocked, avoid_route_nodes,
            process_nodes, processed_node_ids,
        )
        if gate_action is not None:
            return gate_action

    # --- P2/P3: Task strategy — 交付余量内尽可能多做顺路/小绕路任务 ---
    if not force_delivery and not delivery_critical:
        task_action = _handle_tasks(
            match_id, round_num, player_id, player, graph,
            current_node_id, tasks, player_id, phase, weather, blocked,
            goal_node_id=gate_node_id, terminal_node_ids=terminal_node_ids,
            obstacle_nodes=obstacle_nodes, process_nodes=process_nodes,
            processed_node_ids=processed_node_ids, visited_node_ids=visited_node_ids,
            failed_task_ids=failed_task_ids,
            enemy_busy_task_ids=enemy_busy_task_ids,
            max_task_detour=max_task_detour,
            allow_detour=delivery_slack > 0,
            delivery_slack=delivery_slack,
            map_gameplay=map_gameplay,
            task_claimed_this_stop=task_claimed_this_stop,
            inquire_nodes=inquire_nodes,
            my_team_id=my_team_id,
            guard_blocked_targets=guard_blocked_targets,
        )
        if task_action is not None:
            return task_action

    if not force_delivery:
        guard_action = _try_set_guard_action(
            match_id, round_num, player_id, player, graph,
            current_node_id, gate_node_id, terminal_node_ids,
            weather, obstacle_nodes, inquire_nodes, my_team_id,
            opp_player, mode, phase, process_nodes=process_nodes,
            force_delivery=force_delivery,
            map_gameplay=map_gameplay,
        )
        if guard_action is not None:
            return guard_action

    # --- P4: Resource strategy (策略文档 §6) ---
    # Skip resource claiming when close to gate (prioritize delivery)
    dist_to_gate = 0
    if gate_node_id:
        dist_to_gate = graph.path_length(current_node_id, gate_node_id, weather, None)
    if not force_delivery and dist_to_gate > 4:  # Only claim resources when not close to gate
        resource_action = _handle_resources(
            match_id, round_num, player_id, player, graph,
            current_node_id, current_node, phase, weather,
            gate_node_id=gate_node_id, process_nodes=process_nodes,
            map_gameplay=map_gameplay,
        )
        if resource_action is not None:
            return resource_action
    if force_delivery:
        resource_action = _handle_force_delivery_resource(
            match_id, round_num, player_id, player, graph,
            current_node_id, current_node, gate_node_id,
            terminal_node_ids, weather, process_nodes, processed_node_ids,
        )
        if resource_action is not None:
            return resource_action

    # --- P5: Use resources (ice box, horses) ---
    use_res_action = _handle_use_resources(
        match_id, round_num, player_id, player,
        current_node_id, graph, weather, phase,
        gate_node_id, terminal_node_ids, process_nodes, processed_node_ids,
        map_gameplay=map_gameplay,
        inquire_nodes=inquire_nodes,
        visited_node_ids=visited_node_ids,
    )
    if use_res_action is not None:
        return use_res_action

    # --- P5: Combat (策略文档 §8) — guard, break, squad ---
    if not force_delivery:
        combat_action = _handle_combat(
            match_id, round_num, player_id, player, graph,
            current_node_id, gate_node_id, terminal_node_ids,
            weather, blocked_soft, mode, phase, inquire_nodes, opp_player,
            obstacle_nodes=obstacle_nodes,
            process_nodes=process_nodes,
            visited_node_ids=visited_node_ids,
            my_team_id=my_team_id,
            tasks=tasks,
            failed_task_ids=failed_task_ids,
            squad_clear_pending=squad_clear_pending,
            map_gameplay=map_gameplay,
            processed_node_ids=processed_node_ids,
        )
        if combat_action is not None:
            return combat_action

    # --- Rush tactics (策略文档 §10) — reserve quota for gate verify ---
    rush_action = _handle_rush_tactics(
        match_id, round_num, player_id, player,
        current_node_id, phase, mode,
        graph=graph, gate_node_id=gate_node_id,
        terminal_node_ids=terminal_node_ids, weather=weather,
        obstacle_nodes=obstacle_nodes, process_nodes=process_nodes,
        processed_node_ids=processed_node_ids, visited_node_ids=visited_node_ids,
        rush_speed_failed=rush_speed_failed,
        map_gameplay=map_gameplay,
        force_delivery=force_delivery,
        delivery_critical=delivery_critical,
    )
    if rush_action is not None:
        return rush_action

    # --- NAVIGATION: Move toward goal ---
    if round_num <= EARLY_GAME_MAX_ROUND:
        early_move = _try_early_start_move(
            match_id, round_num, player_id, player, graph,
            current_node_id, gate_node_id, terminal_node_ids,
            obstacle_nodes, visited_node_ids, map_gameplay,
            tasks=tasks, weather=weather, process_nodes=process_nodes,
            failed_task_ids=failed_task_ids,
            enemy_busy_task_ids=enemy_busy_task_ids,
            max_task_detour=max_task_detour,
        )
        if early_move is not None:
            return early_move

    if force_delivery:
        move_action = _plan_force_delivery_move(
            match_id, round_num, player_id, player, graph,
            current_node_id, gate_node_id, terminal_node_ids,
            weather, process_nodes, processed_node_ids,
            route_blocked, avoid_route_nodes, guard_stuck_rounds, guard_stuck_target,
            inquire_nodes, tasks, failed_task_ids, obstacle_nodes, my_team_id,
            forced_pass_failed_targets, last_move_failed, last_move_error,
        )
        if move_action is not None:
            return move_action

    move_target = _find_move_target(
        graph, current_node_id, player, gate_node_id, terminal_node_ids,
        weather, route_blocked, obstacle_nodes=obstacle_nodes, process_nodes=process_nodes,
        processed_node_ids=processed_node_ids,
        visited_node_ids=set() if force_delivery else visited_node_ids,
        tasks=tasks, player_id=player_id, failed_task_ids=failed_task_ids,
        enemy_busy_task_ids=enemy_busy_task_ids, phase=phase,
        force_delivery=force_delivery, delivery_slack=delivery_slack,
        max_task_detour=max_task_detour,
        map_gameplay=map_gameplay, round_num=round_num,
    )

    # Next hop has enemy guard → break / forced pass / detour before MOVE
    if move_target and move_target in route_blocked:
        guard_action = _handle_blocked_by_guard(
            match_id, round_num, player_id, player, graph,
            current_node_id, gate_node_id, terminal_node_ids,
            weather, route_blocked, inquire_nodes, process_nodes=process_nodes,
        )
        if guard_action.get("msg_data", {}).get("actions"):
            return guard_action

    # Handle obstacle on next step (策略文档 §3.4: 道路障碍 → T04/CLEAR/FORCED_PASS)
    if move_target and move_target in obstacle_nodes:
        # Priority 1: CLAIM_TASK if T04 task exists at obstacle node (score + clear)
        t04_task = None
        for task in tasks:
            if (task.get("nodeId") == move_target
                    and task.get("active", False)
                    and not task.get("completed", False)
                    and not task.get("failed", False)
                    and get_task_template_id(task).startswith("T04")
                    and task.get("taskId", "") not in failed_task_ids
                    and task.get("taskId", "") not in enemy_busy_task_ids
                    and _t04_targets_obstacle(task, current_node_id, obstacle_nodes, graph)):
                t04_task = task
                break
        if t04_task:
            t04_id = t04_task.get("taskId", "")
            if t04_id not in enemy_busy_task_ids:
                logger.info("Round %d: Obstacle at %s, claiming T04 task", round_num, move_target)
                return make_action(match_id, round_num, player_id, [
                    make_claim_task_action(t04_id)
                ])
            logger.info(
                "Round %d: Skip T04 at %s, enemy processing task %s",
                round_num, move_target, t04_id,
            )

        # Priority 2: CLEAR if we have good fruit to spare (策略文档 §3.4: 1好果6帧)
        good_fruit = get_good_fruit(player)
        if good_fruit >= 2:  # Reserve at least 1 for DELIVER
            logger.info("Round %d: Obstacle at %s, using CLEAR", round_num, move_target)
            return make_action(match_id, round_num, player_id, [
                make_clear_action(move_target)
            ])

        # Priority 3: FORCED_PASS for obstacle-only (策略文档 §3.4: 8帧, no fruit cost)
        logger.info("Round %d: Obstacle at %s, using FORCED_PASS", round_num, move_target)
        return make_action(match_id, round_num, player_id, [
            make_forced_pass_action(move_target)
        ])
    if move_target:
        logger.info("Round %d: NAV move to %s (goal=%s)", round_num, move_target,
                     gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "?"))
        return make_action(match_id, round_num, player_id, [make_move_action(move_target)])

    return make_empty_action(match_id, round_num, player_id)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _find_opponent(all_players: list[dict], my_player_id: int) -> dict | None:
    """Find the opponent player dict."""
    for p in all_players:
        if p.get("playerId") != my_player_id:
            return p
    return None


def _is_on_water_route(
    graph: MapGraph, current_node_id: str,
    gate_node_id: str, terminal_node_ids: list[str],
) -> bool:
    """Check if the current path to goal goes through water edges."""
    goal = gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "")
    if not goal or not current_node_id:
        return False
    path = graph.weighted_shortest_path(current_node_id, goal)
    if not path:
        return False
    for i in range(len(path) - 1):
        if graph.get_edge_route_type(path[i], path[i + 1]) == "WATER":
            return True
    return False


def _get_weather_penalized_routes(weather: dict) -> set[str]:
    """Get route types that should be penalized/avoided based on weather forecast.

    Returns set of route types to avoid (策略文档 §3.2, §6).
    """
    avoid = set()
    if not weather:
        return avoid
    forecasts = weather.get("forecast", []) + weather.get("active", [])
    for fw in forecasts:
        wtype = fw.get("type", "")
        region = fw.get("region", "")
        if wtype == "HOT" or region == "ALL":
            avoid.add("MOUNTAIN")
        elif wtype == "HEAVY_RAIN" or region == "WATER":
            avoid.add("WATER")
        elif wtype == "MOUNTAIN_FOG" or region == "MOUNTAIN":
            avoid.add("MOUNTAIN")
    return avoid


def _get_process_type(
    current_node: dict | None,
    process_nodes: dict[str, dict] | None,
    current_node_id: str,
) -> str | None:
    """Get the process type for the current node."""
    if current_node and needs_processing(current_node):
        return current_node.get("processType", "")
    if process_nodes and current_node_id in process_nodes:
        pn = process_nodes[current_node_id]
        return pn.get("processType", "")
    return None


def _get_goal_node(
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    graph: MapGraph,
    current_node_id: str,
    weather: dict | None = None,
    blocked: set[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
) -> str | None:
    """Determine the current navigation goal based on player state."""
    if is_delivered(player):
        return None
    if not is_verified(player) and gate_node_id:
        return gate_node_id
    if is_verified(player) and terminal_node_ids:
        # Find nearest terminal via weighted path
        best = None
        best_cost = float('inf')
        for tid in terminal_node_ids:
            path = graph.weighted_shortest_path(current_node_id, tid, weather, blocked, process_nodes)
            if path:
                cost = sum(graph.edge_cost(path[i], path[i+1], weather, blocked, process_nodes)
                           for i in range(len(path)-1))
                if cost < best_cost:
                    best_cost = cost
                    best = tid
        return best
    return None


def _get_node_guard(inquire_nodes: list[dict], node_id: str) -> dict:
    for node in inquire_nodes:
        if node.get("nodeId") == node_id:
            guard = node.get("guard")
            return guard if isinstance(guard, dict) else {}
    return {}


def _get_inquire_node(inquire_nodes: list[dict], node_id: str) -> dict | None:
    for node in inquire_nodes:
        if node.get("nodeId") == node_id:
            return node
    return None


def _confirmed_obstacle(
    inquire_nodes: list[dict],
    node_id: str,
    obstacle_nodes: set[str],
) -> bool:
    node = _get_inquire_node(inquire_nodes, node_id)
    if node is not None:
        return node_has_obstacle(node)
    return node_id in obstacle_nodes


def _collect_gate_corridor_enemy_guards(
    inquire_nodes: list[dict],
    my_team_id: str,
    player_id: int,
) -> dict[str, dict]:
    """Scan public state for active enemy guards on the gate corridor (S10-S14)."""
    guarded: dict[str, dict] = {}
    for node in inquire_nodes:
        nid = node.get("nodeId", "")
        if not nid or nid not in GATE_CORRIDOR_NODES:
            continue
        guard = node.get("guard")
        if not isinstance(guard, dict):
            continue
        if is_enemy_guard(guard, my_team_id, player_id):
            guarded[nid] = guard
    return guarded


def _node_has_enemy_guard(
    inquire_nodes: list[dict],
    node_id: str,
    my_team_id: str,
    player_id: int,
) -> bool:
    guard = _get_node_guard(inquire_nodes, node_id)
    return is_enemy_guard(guard, my_team_id, player_id)


def _gate_corridor_path_blocked(
    graph: MapGraph,
    current_node_id: str,
    gate_node_id: str,
    weather: dict | None,
    guard_nodes: set[str],
    avoid_route_nodes: set[str],
    process_nodes: dict[str, dict] | None = None,
) -> bool:
    """True when shortest path to gate must pass through a guarded corridor node."""
    if not current_node_id or not gate_node_id or not guard_nodes:
        return False
    blocked = set(guard_nodes) | set(avoid_route_nodes)
    direct = graph.shortest_path(current_node_id, gate_node_id, weather, blocked)
    if direct:
        return False
    return bool(graph.shortest_path(current_node_id, gate_node_id, weather, avoid_route_nodes))


def _try_gate_corridor_guard_strategy(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node_id: str,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    inquire_nodes: list[dict],
    my_team_id: str,
    route_blocked: set[str],
    avoid_route_nodes: set[str],
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
) -> dict | None:
    """React to corridor guards: weaken proactively or reroute before task/navigation."""
    if not gate_node_id or not current_node_id:
        return None
    guarded = _collect_gate_corridor_enemy_guards(inquire_nodes, my_team_id, player_id)
    if not guarded:
        return None

    guard_nodes = set(guarded)
    path_blocked = _gate_corridor_path_blocked(
        graph, current_node_id, gate_node_id, weather,
        guard_nodes, avoid_route_nodes, process_nodes,
    )

    squad_count = get_squad_count(player)
    if squad_count >= 2 and path_blocked:
        best_nid = ""
        best_defense = -1
        for nid, guard in guarded.items():
            defense = int(guard.get("defense", 0) or 0)
            if defense > best_defense:
                best_defense = defense
                best_nid = nid
        if best_nid:
            logger.info(
                "Round %d: Gate strategy: squad weaken at %s (corridor blocked, def=%d)",
                round_num, best_nid, best_defense,
            )
            return make_action(match_id, round_num, player_id, [
                make_squad_weaken_action(best_nid),
            ])

    if path_blocked:
        detour = _find_delivery_detour_step(
            graph, current_node_id, player, gate_node_id, terminal_node_ids,
            weather, process_nodes, processed_node_ids, route_blocked,
            avoid_nodes=avoid_route_nodes | guard_nodes,
        )
        if detour and detour not in guard_nodes:
            logger.info(
                "Round %d: Gate strategy: reroute via %s (corridor guards=%s)",
                round_num, detour, sorted(guard_nodes),
            )
            return make_action(match_id, round_num, player_id, [make_move_action(detour)])

    return None


def _is_paying_guard_travel_tax(
    state: str,
    player: dict,
    last_move_error: str,
    block_node: str,
    route_blocked: set[str],
    guard_blocked_targets: set[str] | None = None,
    avoid_route_nodes: set[str] | None = None,
) -> bool:
    """WAITING/MOVING 交设卡时间税时只能 WAIT/EMPTY，不能 BREAK/CLEAR。"""
    if last_move_error == "MOVING_ACTION_FORBIDDEN":
        return True
    if state != "WAITING":
        return False
    guard_blocked = guard_blocked_targets or set()
    avoid_nodes = avoid_route_nodes or set()
    next_node = player.get("nextNodeId", "")
    if next_node:
        if next_node in guard_blocked:
            return True
        if next_node in avoid_nodes and next_node not in guard_blocked:
            return False
    if block_node and block_node in guard_blocked:
        return True
    return False


def _handle_limited_state_guard_block(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    state: str,
    block_node: str,
    last_move_failed: bool,
    last_move_error: str,
    inquire_nodes: list[dict],
    my_team_id: str,
    guard_wait_kwargs: dict,
) -> dict:
    """Limited state handler: never BREAK/CLEAR while paying guard travel tax."""
    route_blocked = guard_wait_kwargs.get("route_blocked") or set()
    guard_blocked = guard_wait_kwargs.get("guard_blocked_targets") or set()
    avoid_route_nodes = guard_wait_kwargs.get("avoid_route_nodes") or set()
    if _is_paying_guard_travel_tax(
        state, player, last_move_error, block_node, route_blocked,
        guard_blocked_targets=guard_blocked,
        avoid_route_nodes=avoid_route_nodes,
    ):
        logger.info(
            "Round %d: %s paying guard tax at %s, WAIT only (no BREAK/CLEAR)",
            round_num, state, block_node,
        )
        msg = make_action(match_id, round_num, player_id, [make_wait_action()])
        squad = _make_squad_weaken_action(
            inquire_nodes, block_node, my_team_id, player_id, player,
        )
        if squad:
            return _append_squad_action(msg, squad)
        return msg

    next_node = player.get("nextNodeId", "")
    if (
        state == "WAITING"
        and next_node
        and next_node in avoid_route_nodes
        and next_node not in guard_blocked
    ):
        graph = guard_wait_kwargs.get("graph")
        gate_node_id = guard_wait_kwargs.get("gate_node_id", "")
        terminal_node_ids = guard_wait_kwargs.get("terminal_node_ids") or []
        weather = guard_wait_kwargs.get("weather")
        process_nodes = guard_wait_kwargs.get("process_nodes")
        processed_node_ids = guard_wait_kwargs.get("processed_node_ids") or set()
        current_node_id = get_current_node_id(player) or ""
        if graph and current_node_id:
            detour = _find_delivery_detour_step(
                graph, current_node_id, player, gate_node_id, terminal_node_ids,
                weather, process_nodes, processed_node_ids, route_blocked,
                avoid_nodes=avoid_route_nodes,
            )
            if detour:
                logger.info(
                    "Round %d: WAITING detour via %s (avoid stuck hop %s)",
                    round_num, detour, next_node,
                )
                return make_action(match_id, round_num, player_id, [make_move_action(detour)])

    return _wait_and_weaken_guard(
        match_id, round_num, player_id, player,
        inquire_nodes, block_node, my_team_id,
        **guard_wait_kwargs,
    )


def _estimate_delivery_eta(
    graph: MapGraph,
    current_node_id: str,
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str] | None,
    route_blocked: set[str] | None = None,
) -> float:
    """Estimate frames to delivery-ready state along the planned route.

    Only counts process frames for nodes on the chosen path, not all map process sites.
    """
    if not current_node_id:
        return float("inf")

    blocked = route_blocked or set()
    goal = _get_goal_node(
        player, gate_node_id, terminal_node_ids, graph,
        current_node_id, weather, blocked, None,
    )
    if not goal:
        return 0.0

    path = graph.weighted_shortest_path(
        current_node_id, goal, weather, blocked, None,
    )
    if not path:
        path = graph.shortest_path(current_node_id, goal, weather, blocked)
    if not path:
        return float("inf")

    total = sum(
        graph.edge_cost(path[i], path[i + 1], weather, blocked, None)
        for i in range(len(path) - 1)
    )

    processed = processed_node_ids or set()
    proc_nodes = process_nodes or {}
    for nid in path:
        if nid in processed:
            continue
        info = proc_nodes.get(nid)
        if not info:
            continue
        pt = info.get("processType", "")
        frames = int(info.get("processRound") or PROCESS_COST_FRAMES.get(pt, 0) or 0)
        if is_verify_process(pt):
            if not is_verified(player):
                total += frames
        elif pt:
            total += frames

    return total


def _delivery_slack_frames(
    round_num: int,
    player: dict,
    graph: MapGraph | None,
    current_node_id: str | None,
    gate_node_id: str,
    terminal_node_ids: list[str] | None,
    weather: dict | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str] | None,
    max_round: int = MAX_ROUND,
    route_blocked: set[str] | None = None,
    map_gameplay: MapGameplayContext | None = None,
) -> float:
    """Frames left after estimated delivery ETA and safety buffer (for detour tasks)."""
    eta_buffer = _force_delivery_buffer(map_gameplay)
    remaining = max(0, max_round - round_num)
    if remaining <= 0:
        return 0.0
    if graph is None or not current_node_id:
        if get_task_score(player) >= TASK_SCORE_TARGET:
            return max(0.0, remaining - FORCE_DELIVERY_LATE_REMAINING - eta_buffer)
        return float(remaining)
    eta = _estimate_delivery_eta(
        graph, current_node_id, player, gate_node_id,
        terminal_node_ids or [], weather, process_nodes, processed_node_ids,
        route_blocked=route_blocked,
    )
    if eta == float("inf"):
        return max(0.0, float(remaining) - FORCE_DELIVERY_ETA_REMAINING_MAX)
    return max(0.0, remaining - eta - eta_buffer)


def _max_task_detour_budget(delivery_slack: float, phase: str = "", round_num: int = 0) -> int:
    """How many detour frames we can spend on off-path tasks while still delivering."""
    if round_num >= GATE_ENTRY_DEADLINE_ROUND:
        return 0
    if round_num >= LATE_GAME_NO_MID_TASK_ROUND:
        return 0
    if delivery_slack <= 0:
        return 0
    if delivery_slack > 120:
        budget = MAX_TASK_DETOUR_COST + int(delivery_slack * 0.18)
    elif delivery_slack > 60:
        budget = MAX_TASK_DETOUR_COST + int(delivery_slack * 0.12)
    else:
        budget = max(6, int(delivery_slack * 0.30))
    if phase == "RUSH":
        budget += RUSH_TASK_DETOUR_BONUS
    return min(MAX_TASK_DETOUR_CEILING, budget)


def _should_force_delivery(
    round_num: int,
    phase: str,
    player: dict,
    graph: MapGraph | None = None,
    current_node_id: str | None = None,
    gate_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    weather: dict | None = None,
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
    max_round: int = MAX_ROUND,
    route_blocked: set[str] | None = None,
    map_gameplay: MapGameplayContext | None = None,
) -> bool:
    """Pure delivery rush when ETA slack falls below the map-specific buffer."""
    if is_verified(player) or is_delivered(player):
        return False
    if round_num < EARLY_GAME_MAX_ROUND and phase != "RUSH":
        return False
    remaining = max(0, max_round - round_num)
    if remaining <= 0:
        return True
    if remaining > FORCE_DELIVERY_MIN_REMAINING and phase != "RUSH":
        return False
    slack = _delivery_slack_frames(
        round_num, player, graph, current_node_id, gate_node_id,
        terminal_node_ids, weather, process_nodes, processed_node_ids, max_round,
        route_blocked=route_blocked,
        map_gameplay=map_gameplay,
    )
    if (
        phase == "RUSH"
        and remaining > FORCE_DELIVERY_MIN_REMAINING
        and get_task_score(player) < TASK_SCORE_TARGET
    ):
        return slack <= _force_delivery_buffer(map_gameplay) * 0.5
    return slack <= _force_delivery_buffer(map_gameplay)


def _find_direct_delivery_step(
    graph: MapGraph,
    current_node_id: str,
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
    avoid_nodes: set[str] | None = None,
) -> str | None:
    goal_node = _get_goal_node(
        player, gate_node_id, terminal_node_ids, graph,
        current_node_id, weather, None, process_nodes,
    )
    if not goal_node:
        return None

    remaining_process_nodes = None
    if process_nodes:
        remaining_process_nodes = {
            nid: info for nid, info in process_nodes.items()
            if nid not in processed_node_ids
        }

    if avoid_nodes:
        neighbors = graph.get_neighbors(current_node_id)
        best_step = None
        best_cost = float("inf")
        for neighbor in neighbors:
            if neighbor in avoid_nodes:
                continue
            path = graph.weighted_shortest_path(
                neighbor, goal_node, weather, None, remaining_process_nodes,
            )
            if not path:
                continue
            hop_cost = graph.edge_cost(
                current_node_id, neighbor, weather, None, remaining_process_nodes,
            )
            tail_cost = sum(
                graph.edge_cost(path[i], path[i + 1], weather, None, remaining_process_nodes)
                for i in range(len(path) - 1)
            )
            total = hop_cost + tail_cost
            if total < best_cost:
                best_cost = total
                best_step = neighbor
        if best_step:
            return best_step

    step = graph.next_step_toward(
        current_node_id, goal_node, weather, None,
        use_weighted=True, process_nodes=remaining_process_nodes,
    )
    if step and avoid_nodes and step in avoid_nodes:
        return None
    if step:
        return step
    return graph.next_step_toward(current_node_id, goal_node, weather, None, use_weighted=False)


def _find_delivery_detour_step(
    graph: MapGraph,
    current_node_id: str,
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
    route_blocked: set[str],
    avoid_nodes: set[str] | None = None,
) -> str | None:
    """Pick an unblocked neighbor that still reaches the delivery goal."""
    if avoid_nodes is None:
        avoid_nodes = set()
    goal_node = _get_goal_node(
        player, gate_node_id, terminal_node_ids, graph,
        current_node_id, weather, route_blocked, process_nodes,
    )
    if not goal_node:
        return None

    remaining_process_nodes = None
    if process_nodes:
        remaining_process_nodes = {
            nid: info for nid, info in process_nodes.items()
            if nid not in processed_node_ids
        }

    routing_blocked = set(route_blocked) | set(avoid_nodes)
    best_step = None
    best_cost = float("inf")
    for neighbor in graph.get_neighbors(current_node_id):
        if neighbor in avoid_nodes:
            continue
        path = graph.weighted_shortest_path(
            neighbor, goal_node, weather, routing_blocked, remaining_process_nodes,
        )
        if not path:
            continue
        hop_cost = graph.edge_cost(
            current_node_id, neighbor, weather, routing_blocked, remaining_process_nodes,
        )
        tail_cost = sum(
            graph.edge_cost(path[i], path[i + 1], weather, routing_blocked, remaining_process_nodes)
            for i in range(len(path) - 1)
        )
        total = hop_cost + tail_cost
        if total < best_cost:
            best_cost = total
            best_step = neighbor
    return best_step


def _should_detour_force_delivery(
    direct_target: str,
    route_blocked: set[str],
    avoid_route_nodes: set[str],
    guard_stuck_rounds: int,
    guard_stuck_target: str,
) -> bool:
    if not direct_target:
        return False
    if direct_target in avoid_route_nodes:
        return True
    if guard_stuck_target and direct_target == guard_stuck_target:
        if guard_stuck_rounds >= GUARD_SILENT_WAIT_LIMIT:
            return True
    if direct_target in route_blocked and guard_stuck_rounds >= GUARD_SILENT_WAIT_LIMIT:
        return True
    return False


def _plan_limited_state_force_delivery_move(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node_id: str,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
    route_blocked: set[str],
    avoid_route_nodes: set[str],
) -> dict | None:
    """WAITING/MOVING 下 force delivery：仅 MOVE/WAIT/马类（任务书 §8.2）。"""
    if _must_wait_for_gate_verify(player, gate_node_id, current_node_id):
        logger.info(
            "Round %d: limited force delivery blocked at unverified gate %s, WAIT",
            round_num, current_node_id,
        )
        return make_action(match_id, round_num, player_id, [make_wait_action()])
    next_node = player.get("nextNodeId", "")
    if next_node:
        if next_node in route_blocked:
            logger.info(
                "Round %d: limited force delivery blocked hop %s, WAIT",
                round_num, next_node,
            )
            return make_action(match_id, round_num, player_id, [make_wait_action()])
        return make_action(match_id, round_num, player_id, [make_move_action(next_node)])

    target = _find_direct_delivery_step(
        graph, current_node_id, player, gate_node_id, terminal_node_ids,
        weather, process_nodes, processed_node_ids,
        avoid_nodes=avoid_route_nodes,
    )
    if target and target not in route_blocked:
        horse_action = _use_horse_before_expensive_hop(
            match_id, round_num, player_id, player, graph,
            current_node_id, target, weather, process_nodes,
            force_delivery=True,
        )
        if horse_action is not None:
            return horse_action
        logger.info(
            "Round %d: limited force delivery move to %s",
            round_num, target,
        )
        return make_action(match_id, round_num, player_id, [make_move_action(target)])

    detour = _find_delivery_detour_step(
        graph, current_node_id, player, gate_node_id, terminal_node_ids,
        weather, process_nodes, processed_node_ids, route_blocked,
        avoid_nodes=avoid_route_nodes,
    )
    if detour:
        logger.info(
            "Round %d: limited force delivery detour via %s",
            round_num, detour,
        )
        return make_action(match_id, round_num, player_id, [make_move_action(detour)])

    return make_action(match_id, round_num, player_id, [make_wait_action()])


def _plan_force_delivery_move(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node_id: str,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
    route_blocked: set[str],
    avoid_route_nodes: set[str],
    guard_stuck_rounds: int,
    guard_stuck_target: str,
    inquire_nodes: list[dict],
    tasks: list[dict],
    failed_task_ids: set[str],
    obstacle_nodes: set[str],
    my_team_id: str,
    forced_pass_failed_targets: set[str],
    last_move_failed: bool,
    last_move_error: str,
    *,
    log_prefix: str = "FORCE_DELIVERY",
) -> dict | None:
    """Resolve the next force-delivery hop, including detour and guard handling."""
    path_avoid = set(avoid_route_nodes)
    if current_node_id not in GATE_CORRIDOR_NODES:
        path_avoid |= obstacle_nodes
    direct_target = _find_direct_delivery_step(
        graph, current_node_id, player, gate_node_id, terminal_node_ids,
        weather, process_nodes, processed_node_ids,
        avoid_nodes=path_avoid,
    )
    target = direct_target
    detour = False

    if _should_detour_force_delivery(
        direct_target or "", route_blocked, avoid_route_nodes,
        guard_stuck_rounds, guard_stuck_target,
    ):
        alt = _find_delivery_detour_step(
            graph, current_node_id, player, gate_node_id, terminal_node_ids,
            weather, process_nodes, processed_node_ids, route_blocked,
            avoid_nodes=avoid_route_nodes | ({direct_target} if direct_target else set()),
        )
        if alt:
            logger.info(
                "Round %d: %s detour via %s (avoid %s, stuck=%d)",
                round_num, log_prefix, alt, direct_target or guard_stuck_target, guard_stuck_rounds,
            )
            target = alt
            detour = True

    if not target:
        alt = _find_delivery_detour_step(
            graph, current_node_id, player, gate_node_id, terminal_node_ids,
            weather, process_nodes, processed_node_ids, route_blocked,
            avoid_nodes=avoid_route_nodes,
        )
        if alt:
            logger.info("Round %d: %s fallback detour via %s", round_num, log_prefix, alt)
            target = alt
            detour = True

    if not target:
        return None

    if not detour:
        if target in route_blocked or target in obstacle_nodes:
            blocker_action = _handle_force_delivery_blocker(
                match_id, round_num, player_id, player,
                target, inquire_nodes, tasks, failed_task_ids,
                obstacle_nodes, my_team_id, route_blocked,
            )
            if blocker_action.get("msg_data", {}).get("actions"):
                return blocker_action
        hop_action = _resolve_guarded_delivery_hop(
            match_id, round_num, player_id, player,
            current_node_id, target, forced_pass_failed_targets,
            inquire_nodes, tasks, failed_task_ids, obstacle_nodes,
            my_team_id, route_blocked,
            last_move_failed=last_move_failed,
            last_move_error=last_move_error,
        )
        if hop_action is not None:
            return hop_action

    horse_action = _use_horse_before_expensive_hop(
        match_id, round_num, player_id, player, graph,
        current_node_id, target, weather, process_nodes,
        force_delivery=True,
    )
    if horse_action is not None:
        return horse_action

    logger.info("Round %d: %s move to %s (goal=%s)%s", round_num, log_prefix, target, gate_node_id, " detour" if detour else "")
    return make_action(match_id, round_num, player_id, [make_move_action(target)])


def _handle_force_delivery_blocker(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    target_node_id: str,
    inquire_nodes: list[dict],
    tasks: list[dict],
    failed_task_ids: set[str],
    obstacle_nodes: set[str],
    my_team_id: str,
    route_blocked: set[str] | None = None,
) -> dict:
    if route_blocked is None:
        route_blocked = set()

    guard = _get_node_guard(inquire_nodes, target_node_id)
    if is_enemy_guard(guard, my_team_id, player_id):
        good, bad = _break_guard_investment(player)
        if good + bad > 0:
            action = make_break_guard_action(target_node_id, good_fruit=good, bad_fruit=bad)
            logger.info("Round %d: FORCE_DELIVERY break guard at %s", round_num, target_node_id)
            return make_action(match_id, round_num, player_id, [action])
        logger.info("Round %d: FORCE_DELIVERY forced pass guard at %s", round_num, target_node_id)
        return make_action(match_id, round_num, player_id, [make_forced_pass_action(target_node_id)])

    if target_node_id in obstacle_nodes:
        if not _confirmed_obstacle(inquire_nodes, target_node_id, obstacle_nodes):
            logger.info(
                "Round %d: FORCE_DELIVERY no confirmed blocker at %s, skip FORCED_PASS",
                round_num, target_node_id,
            )
            return make_empty_action(match_id, round_num, player_id)
        if guard_is_active(guard) and not is_own_guard(guard, my_team_id, player_id):
            logger.info(
                "Round %d: FORCE_DELIVERY skip CLEAR at %s (active guard, not obstacle)",
                round_num, target_node_id,
            )
            return make_action(match_id, round_num, player_id, [make_forced_pass_action(target_node_id)])
        t04_task = None
        for task in tasks:
            if (task.get("nodeId") == target_node_id
                    and task.get("active", False)
                    and not task.get("completed", False)
                    and not task.get("failed", False)
                    and get_task_template_id(task).startswith("T04")
                    and task.get("taskId", "") not in failed_task_ids):
                t04_task = task
                break
        if t04_task:
            logger.info("Round %d: FORCE_DELIVERY T04 clear at %s", round_num, target_node_id)
            return make_action(match_id, round_num, player_id, [
                make_claim_task_action(t04_task.get("taskId", ""))
            ])
        if get_good_fruit(player) >= 2:
            logger.info("Round %d: FORCE_DELIVERY CLEAR at %s", round_num, target_node_id)
            return make_action(match_id, round_num, player_id, [make_clear_action(target_node_id)])
        logger.info("Round %d: FORCE_DELIVERY forced pass obstacle at %s", round_num, target_node_id)
        return make_action(match_id, round_num, player_id, [make_forced_pass_action(target_node_id)])

    logger.info("Round %d: FORCE_DELIVERY no live blocker at %s, skip FORCED_PASS", round_num, target_node_id)
    return make_empty_action(match_id, round_num, player_id)


def _resolve_guarded_delivery_hop(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    current_node_id: str,
    target_node_id: str,
    forced_pass_failed_targets: set[str],
    inquire_nodes: list[dict],
    tasks: list[dict],
    failed_task_ids: set[str],
    obstacle_nodes: set[str],
    my_team_id: str,
    route_blocked: set[str],
    last_move_failed: bool = False,
    last_move_error: str = "",
) -> dict | None:
    """Resolve a guarded/obstructed delivery hop using live state, not fixed timers."""
    if last_move_failed and last_move_error == "OBJECT_BUSY":
        logger.info(
            "Round %d: guard setup window at %s, waiting (no FORCED_PASS retry)",
            round_num, target_node_id,
        )
        return make_action(match_id, round_num, player_id, [make_wait_action()])

    if target_node_id in obstacle_nodes:
        if is_enemy_guard(_get_node_guard(inquire_nodes, target_node_id), my_team_id, player_id):
            pass
        elif _confirmed_obstacle(inquire_nodes, target_node_id, obstacle_nodes):
            return _handle_force_delivery_blocker(
                match_id, round_num, player_id, player,
                target_node_id, inquire_nodes, tasks, failed_task_ids,
                obstacle_nodes, my_team_id, route_blocked,
            )

    guard = _get_node_guard(inquire_nodes, target_node_id)
    live_guard = is_enemy_guard(guard, my_team_id, player_id)
    live_obstacle = _confirmed_obstacle(inquire_nodes, target_node_id, obstacle_nodes)
    guarded = target_node_id in route_blocked or live_guard or live_obstacle
    if not guarded:
        return None
    if not live_guard and not live_obstacle:
        logger.info(
            "Round %d: %s is locally blocked only, skip FORCED_PASS probe",
            round_num, target_node_id,
        )
        return None

    if target_node_id in forced_pass_failed_targets:
        return _handle_force_delivery_blocker(
            match_id, round_num, player_id, player,
            target_node_id, inquire_nodes, tasks, failed_task_ids,
            obstacle_nodes, my_team_id, route_blocked,
        )

    logger.info(
        "Round %d: probing guarded hop %s with FORCED_PASS",
        round_num, target_node_id,
    )
    return make_action(match_id, round_num, player_id, [
        make_forced_pass_action(target_node_id)
    ])


def _horse_use_min_hop_cost(player: dict, force_delivery: bool = False) -> float:
    if force_delivery:
        return HORSE_USE_MIN_HOP_COST
    if get_task_score(player) < TASK_SCORE_TARGET:
        return HORSE_USE_MIN_HOP_COST_EARLY
    return HORSE_USE_MIN_HOP_COST


def _use_horse_before_expensive_hop(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node_id: str,
    target_node_id: str,
    weather: dict | None,
    process_nodes: dict[str, dict] | None,
    force_delivery: bool = False,
) -> dict | None:
    """Use horse buff before a high-cost edge."""
    if not target_node_id or current_node_id == target_node_id:
        return None
    if _has_move_speed_buff(player):
        return None
    hop_cost = graph.edge_cost(
        current_node_id, target_node_id, weather, None, process_nodes,
    )
    if hop_cost < _horse_use_min_hop_cost(player, force_delivery):
        return None
    if has_resource(player, "FAST_HORSE"):
        logger.info(
            "Round %d: Using FAST_HORSE before hop %s->%s (cost=%.1f)",
            round_num, current_node_id, target_node_id, hop_cost,
        )
        return make_action(match_id, round_num, player_id, [
            make_use_resource_action("FAST_HORSE")
        ])
    if has_resource(player, "SHORT_HORSE"):
        logger.info(
            "Round %d: Using SHORT_HORSE before hop %s->%s (cost=%.1f)",
            round_num, current_node_id, target_node_id, hop_cost,
        )
        return make_action(match_id, round_num, player_id, [
            make_use_resource_action("SHORT_HORSE")
        ])
    return None


def _use_horse_immediately(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    current_node_id: str,
) -> dict | None:
    """Use a held horse as soon as possible; horse buffs do not stack."""
    if _has_move_speed_buff(player):
        return None
    if has_resource(player, "FAST_HORSE"):
        logger.info(
            "Round %d: Immediately using FAST_HORSE at %s",
            round_num, current_node_id,
        )
        return make_action(match_id, round_num, player_id, [
            make_use_resource_action("FAST_HORSE")
        ])
    if has_resource(player, "SHORT_HORSE"):
        logger.info(
            "Round %d: Immediately using SHORT_HORSE at %s",
            round_num, current_node_id,
        )
        return make_action(match_id, round_num, player_id, [
            make_use_resource_action("SHORT_HORSE")
        ])
    return None


def _should_use_task_aware_routing(
    player: dict, phase: str, force_delivery: bool,
    delivery_slack: float = 999.0,
    round_num: int = 0,
) -> bool:
    if force_delivery:
        return False
    if round_num >= GATE_ENTRY_DEADLINE_ROUND:
        return False
    if round_num >= LATE_GAME_NO_MID_TASK_ROUND and get_task_score(player) >= TASK_SCORE_TARGET:
        return False
    if delivery_slack <= TASK_DETOUR_SLACK_RESERVE:
        return False
    return True


def _task_routable(
    task: dict,
    player: dict,
    obstacle_nodes: set[str] | None,
) -> bool:
    template_id = get_task_template_id(task)
    if template_id.startswith("T06"):
        if not has_resource(player, "FAST_HORSE") and not has_resource(player, "SHORT_HORSE"):
            return False
    if template_id.startswith("T04"):
        task_node = task.get("nodeId", "")
        if obstacle_nodes and task_node and task_node not in obstacle_nodes:
            return False
    return True


def _bucket_available_task_score(
    tasks: list[dict],
    player_id: int,
    player: dict,
    failed_task_ids: set[str] | None,
    enemy_busy_task_ids: set[str] | None,
    obstacle_nodes: set[str] | None,
    route_bucket: str,
) -> int:
    total = 0
    for task in tasks:
        if not is_task_available(task, player_id, failed_task_ids, enemy_busy_task_ids):
            continue
        if not _task_routable(task, player, obstacle_nodes):
            continue
        if (task.get("routeBucket") or "ROAD") != route_bucket:
            continue
        total += get_task_point_value(task)
    return total


def _prefer_water_route(
    player: dict,
    tasks: list[dict],
    player_id: int,
    failed_task_ids: set[str] | None,
    enemy_busy_task_ids: set[str] | None,
    obstacle_nodes: set[str] | None,
    map_gameplay: MapGameplayContext | None = None,
) -> bool:
    """Prefer water when enough WATER-bucket task score is available."""
    prof = _profile(map_gameplay)
    if prof.require_ice_for_water and not has_resource(player, "ICE_BOX"):
        return False
    water_score = _bucket_available_task_score(
        tasks, player_id, player, failed_task_ids, enemy_busy_task_ids,
        obstacle_nodes, "WATER",
    )
    return water_score >= _water_task_min(map_gameplay)


def _task_process_frames(task: dict) -> int:
    template_id = get_task_template_id(task)
    for prefix, (_score, proc_round, _spr) in TASK_PRIORITY.items():
        if template_id.startswith(prefix):
            return proc_round
    return 6


def _task_involves_backtrack(
    task_node: str,
    current_node_id: str,
    visited_node_ids: set[str],
    graph: MapGraph,
    weather: dict | None,
    blocked: set[str] | None,
    process_nodes: dict[str, dict] | None,
) -> bool:
    """任务目标在已访问节点，或前往路径需经过已访问节点。"""
    if task_node != current_node_id and task_node in visited_node_ids:
        return True
    path = graph.weighted_shortest_path(
        current_node_id, task_node, weather, blocked, process_nodes,
    )
    if not path or len(path) < 2:
        return False
    for nid in path[1:]:
        if nid in visited_node_ids:
            return True
    return False


def _task_detour_unacceptable(
    detour: int,
    delivery_slack: float,
    task: dict,
) -> bool:
    """绕路+读条会耗尽交付余量则不做该任务。"""
    if delivery_slack <= TASK_DETOUR_SLACK_RESERVE:
        return True
    cost = detour + _task_process_frames(task)
    return cost >= delivery_slack - TASK_DETOUR_SLACK_RESERVE


def _task_is_away_from_gate(
    graph: MapGraph,
    current_node_id: str,
    task_node: str,
    gate_node_id: str,
    weather: dict | None,
    obstacle_nodes: set[str] | None,
) -> bool:
    """任务节点比当前位置离宫门更远 → 视为折返。"""
    if not gate_node_id or not task_node or task_node == current_node_id:
        return False
    cur_hops = graph.path_length(current_node_id, gate_node_id, weather, obstacle_nodes)
    task_hops = graph.path_length(task_node, gate_node_id, weather, obstacle_nodes)
    if cur_hops == float("inf") or task_hops == float("inf"):
        return False
    return task_hops > cur_hops


def _should_skip_mid_task_detour(
    round_num: int,
    current_node_id: str,
    task_node: str,
    map_gameplay: MapGameplayContext | None,
) -> bool:
    """中后期禁止从水路/入关方向折回官道中段做任务。"""
    if round_num < LATE_GAME_NO_MID_TASK_ROUND:
        return False
    ctx = _map_ctx(map_gameplay)
    if current_node_id in ctx.water_route_nodes and task_node in ctx.official_mid_route_nodes:
        return True
    if current_node_id in GATE_CORRIDOR_NODES and task_node not in GATE_CORRIDOR_NODES:
        return True
    return False


def _try_early_start_move(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node_id: str,
    gate_node_id: str,
    terminal_node_ids: list[str],
    obstacle_nodes: set[str],
    visited_node_ids: set[str],
    map_gameplay: MapGameplayContext | None,
    tasks: list[dict] | None = None,
    weather: dict | None = None,
    process_nodes: dict[str, dict] | None = None,
    failed_task_ids: set[str] | None = None,
    enemy_busy_task_ids: set[str] | None = None,
    max_task_detour: int | None = None,
) -> dict | None:
    """开局先按任务收益选路；没有任务信号时再回退到水路方向。"""
    ctx = _map_ctx(map_gameplay)
    if round_num > EARLY_GAME_MAX_ROUND:
        return None
    if ctx.start_node_id and current_node_id != ctx.start_node_id:
        return None
    if len(visited_node_ids) > 1:
        return None
    tasks = tasks or []
    failed_task_ids = failed_task_ids or set()
    enemy_busy_task_ids = enemy_busy_task_ids or set()
    candidates = [
        n for n in graph.get_neighbors(current_node_id)
        if n not in obstacle_nodes and n not in visited_node_ids
    ]
    if not candidates:
        return None

    if tasks and gate_node_id:
        task_step = _pick_task_aware_neighbor(
            graph, current_node_id, candidates, gate_node_id,
            tasks, player_id, player, gate_node_id, terminal_node_ids,
            weather, None, process_nodes, failed_task_ids,
            enemy_busy_task_ids, obstacle_nodes, visited_node_ids,
            max_task_detour=max_task_detour,
            prefer_water=True,
            map_gameplay=map_gameplay,
        )
        if task_step:
            route_score, route_tasks, route_high = _route_task_stats(
                graph, current_node_id, task_step, gate_node_id,
                tasks, player_id, player, gate_node_id, terminal_node_ids,
                weather, None, process_nodes, failed_task_ids,
                enemy_busy_task_ids, obstacle_nodes,
                max_task_detour=max_task_detour,
            )
            if route_score > 0:
                logger.info(
                    "Round %d: early task-aware move via %s (routeTasks=%d high=%d score=%d)",
                    round_num, task_step, route_tasks, route_high, route_score,
                )
                return make_action(match_id, round_num, player_id, [make_move_action(task_step)])

    best: str | None = None
    best_score = float("inf")
    for neighbor in candidates:
        water_hops = float("inf")
        for w in ctx.water_route_nodes:
            h = graph.path_length(neighbor, w, None, obstacle_nodes)
            if h < water_hops:
                water_hops = h
        gate_hops = (
            graph.path_length(neighbor, gate_node_id, None, obstacle_nodes)
            if gate_node_id else water_hops
        )
        score = water_hops * 10 + gate_hops
        if score < best_score:
            best_score = score
            best = neighbor
    if best:
        logger.info("Round %d: early start move toward water route via %s", round_num, best)
        return make_action(match_id, round_num, player_id, [make_move_action(best)])
    return None


def _is_water_task_location(
    task_node: str,
    task: dict,
    map_gameplay: MapGameplayContext | None,
) -> bool:
    ctx = _map_ctx(map_gameplay)
    return task_node in ctx.water_route_nodes or (task.get("routeBucket") or "ROAD") == "WATER"


def _filter_neighbors_for_route_preference(
    neighbors: list[str],
    current_node_id: str,
    prefer_water: bool,
    map_gameplay: MapGameplayContext | None = None,
    force_delivery: bool = False,
    gate_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
) -> list[str]:
    """未满足水路条件时尽量避开水路，已在链路上时不切断。"""
    if terminal_node_ids is None:
        terminal_node_ids = []
    delivery_goals = {gate_node_id, *terminal_node_ids} - {""}
    if force_delivery or any(n in delivery_goals for n in neighbors):
        return neighbors
    if prefer_water or not neighbors:
        return neighbors
    ctx = _map_ctx(map_gameplay)
    on_water_chain = current_node_id in ctx.water_route_nodes
    if on_water_chain:
        water_neighbors = [n for n in neighbors if n in ctx.water_route_nodes]
        if water_neighbors:
            return water_neighbors
    without_water = [n for n in neighbors if n not in ctx.water_route_nodes]
    if without_water:
        neighbors = without_water
    official = [n for n in neighbors if n in ctx.official_mid_route_nodes]
    if official:
        return official
    return neighbors


def _route_bucket_task_scores(
    tasks: list[dict],
    player_id: int,
    player: dict,
    failed_task_ids: set[str] | None,
    enemy_busy_task_ids: set[str] | None,
    obstacle_nodes: set[str] | None,
    prefer_water: bool = True,
) -> dict[str, int]:
    scores: dict[str, int] = {}
    for task in tasks:
        if not is_task_available(task, player_id, failed_task_ids, enemy_busy_task_ids):
            continue
        if not _task_routable(task, player, obstacle_nodes):
            continue
        bucket = task.get("routeBucket") or "ROAD"
        if bucket == "WATER" and not prefer_water:
            continue
        scores[bucket] = scores.get(bucket, 0) + get_task_point_value(task)
    return scores


def _weighted_path_cost(
    graph: MapGraph,
    start: str,
    end: str,
    weather: dict | None,
    blocked: set[str] | None,
    process_nodes: dict[str, dict] | None,
) -> float:
    path = graph.weighted_shortest_path(start, end, weather, blocked, process_nodes)
    if not path:
        return float("inf")
    return sum(
        graph.edge_cost(path[i], path[i + 1], weather, blocked, process_nodes)
        for i in range(len(path) - 1)
    )


def _route_task_stats(
    graph: MapGraph,
    current_node_id: str,
    neighbor: str,
    goal_node: str,
    tasks: list[dict],
    player_id: int,
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    blocked: set[str] | None,
    process_nodes: dict[str, dict] | None,
    failed_task_ids: set[str] | None,
    enemy_busy_task_ids: set[str] | None,
    obstacle_nodes: set[str] | None,
    max_task_detour: int | None = None,
) -> tuple[int, int, int]:
    """Return (task_score_sum, on_path_task_count, high_value_task_count) for a route via neighbor."""
    if max_task_detour is None:
        max_task_detour = MAX_TASK_DETOUR_COST
    path = graph.weighted_shortest_path(neighbor, goal_node, weather, blocked, process_nodes)
    on_path = set(path) if path else set()
    score_sum = 0
    task_count = 0
    high_count = 0

    for task in tasks:
        if not is_task_available(task, player_id, failed_task_ids, enemy_busy_task_ids):
            continue
        if not _task_routable(task, player, obstacle_nodes):
            continue
        task_node = task.get("nodeId", "")
        if not task_node:
            continue

        on_route = task_node in on_path or task_node == neighbor
        if not on_route:
            detour = _calc_detour_cost(
                graph, current_node_id, task_node, gate_node_id, terminal_node_ids,
                weather, blocked, player, process_nodes,
            )
            if detour > max_task_detour:
                continue
            # 顺路绕一点也算分，但不计入路线任务数量
            score_sum += get_task_point_value(task)
            continue

        pts = get_task_point_value(task)
        score_sum += pts
        task_count += 1
        if pts >= 30:
            high_count += 1

    return score_sum, task_count, high_count


def _neighbor_task_yield(
    graph: MapGraph,
    current_node_id: str,
    neighbor: str,
    goal_node: str,
    tasks: list[dict],
    player_id: int,
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    blocked: set[str] | None,
    process_nodes: dict[str, dict] | None,
    failed_task_ids: set[str] | None,
    enemy_busy_task_ids: set[str] | None,
    obstacle_nodes: set[str] | None,
) -> int:
    """Sum claimable task points on the path via neighbor or within detour budget."""
    score_sum, _, _ = _route_task_stats(
        graph, current_node_id, neighbor, goal_node, tasks, player_id, player,
        gate_node_id, terminal_node_ids, weather, blocked, process_nodes,
        failed_task_ids, enemy_busy_task_ids, obstacle_nodes,
    )
    return score_sum


def _score_navigation_neighbor(
    graph: MapGraph,
    current_node_id: str,
    neighbor: str,
    goal_node: str,
    tasks: list[dict],
    player_id: int,
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    blocked: set[str] | None,
    process_nodes: dict[str, dict] | None,
    failed_task_ids: set[str] | None,
    enemy_busy_task_ids: set[str] | None,
    obstacle_nodes: set[str] | None,
    visited_node_ids: set[str],
    bucket_scores: dict[str, int],
    max_task_detour: int | None = None,
    prefer_water: bool = True,
    map_gameplay: MapGameplayContext | None = None,
) -> float:
    hop_cost = graph.edge_cost(current_node_id, neighbor, weather, blocked, process_nodes)
    tail_cost = _weighted_path_cost(graph, neighbor, goal_node, weather, blocked, process_nodes)
    if tail_cost == float("inf"):
        return float("inf")

    score = hop_cost + tail_cost
    if neighbor in visited_node_ids:
        score += ROUTE_VISITED_BACKTRACK_PENALTY

    if not prefer_water:
        edge_type = graph.get_edge_route_type(current_node_id, neighbor)
        water_nodes = _map_ctx(map_gameplay).water_route_nodes
        if edge_type == "WATER" or neighbor in water_nodes:
            score += WATER_ROUTE_NAV_PENALTY

    task_score_sum, task_count, high_count = _route_task_stats(
        graph, current_node_id, neighbor, goal_node, tasks, player_id, player,
        gate_node_id, terminal_node_ids, weather, blocked, process_nodes,
        failed_task_ids, enemy_busy_task_ids, obstacle_nodes,
        max_task_detour=max_task_detour,
    )
    score -= task_score_sum * ROUTE_TASK_BONUS_PER_SCORE
    score -= task_count * ROUTE_TASK_COUNT_BONUS
    score -= high_count * ROUTE_HIGH_VALUE_TASK_BONUS

    edge_type = graph.get_edge_route_type(current_node_id, neighbor)
    score -= bucket_scores.get(edge_type, 0) * ROUTE_BUCKET_BONUS_PER_SCORE
    return score


def _pick_task_aware_neighbor(
    graph: MapGraph,
    current_node_id: str,
    available: list[str],
    goal_node: str,
    tasks: list[dict],
    player_id: int,
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    blocked: set[str] | None,
    process_nodes: dict[str, dict] | None,
    failed_task_ids: set[str] | None,
    enemy_busy_task_ids: set[str] | None,
    obstacle_nodes: set[str] | None,
    visited_node_ids: set[str],
    max_task_detour: int | None = None,
    prefer_water: bool = True,
    map_gameplay: MapGameplayContext | None = None,
) -> str | None:
    if not available or not goal_node:
        return None

    bucket_scores = _route_bucket_task_scores(
        tasks, player_id, player, failed_task_ids, enemy_busy_task_ids,
        obstacle_nodes, prefer_water=prefer_water,
    )
    best_neighbor = None
    best_score = float("inf")
    best_stats = (0, 0, 0)
    for neighbor in available:
        nav_score = _score_navigation_neighbor(
            graph, current_node_id, neighbor, goal_node, tasks, player_id, player,
            gate_node_id, terminal_node_ids, weather, blocked, process_nodes,
            failed_task_ids, enemy_busy_task_ids, obstacle_nodes,
            visited_node_ids, bucket_scores, max_task_detour=max_task_detour,
            prefer_water=prefer_water, map_gameplay=map_gameplay,
        )
        route_stats = _route_task_stats(
            graph, current_node_id, neighbor, goal_node, tasks, player_id, player,
            gate_node_id, terminal_node_ids, weather, blocked, process_nodes,
            failed_task_ids, enemy_busy_task_ids, obstacle_nodes,
            max_task_detour=max_task_detour,
        )
        if nav_score < best_score or (
            nav_score == best_score and route_stats[1] > best_stats[1]
        ):
            best_score = nav_score
            best_neighbor = neighbor
            best_stats = route_stats
    if best_neighbor:
        logger.debug(
            "route pick %s: score=%d tasks=%d high=%d nav=%.1f",
            best_neighbor, best_stats[0], best_stats[1], best_stats[2], best_score,
        )
    return best_neighbor


def _find_move_target(
    graph: MapGraph,
    current_node_id: str,
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None = None,
    blocked: set[str] | None = None,
    failed_target: str = "",
    obstacle_nodes: set[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    tasks: list[dict] | None = None,
    player_id: int = 0,
    failed_task_ids: set[str] | None = None,
    enemy_busy_task_ids: set[str] | None = None,
    phase: str = "",
    force_delivery: bool = False,
    delivery_slack: float = 999.0,
    max_task_detour: int | None = None,
    map_gameplay: MapGameplayContext | None = None,
    round_num: int = 0,
) -> str | None:
    """Find the best move target using weighted shortest path toward the current goal.

    Filters out obstacle nodes (hasObstacle=true) from move targets,
    since MOVE to an obstacle node will be rejected with TARGET_NOT_REACHABLE.
    """
    if obstacle_nodes is None:
        obstacle_nodes = set()
    if processed_node_ids is None:
        processed_node_ids = set()
    if visited_node_ids is None:
        visited_node_ids = set()

    if _must_wait_for_gate_verify(player, gate_node_id, current_node_id):
        return None

    neighbors = graph.get_neighbors(current_node_id)
    if not neighbors:
        return None

    # Filter out failed target, obstacle nodes, and enemy-guarded nodes when alternatives exist
    guard_blocked = blocked or set()
    all_safe = [n for n in neighbors if n != failed_target and n not in obstacle_nodes]
    if guard_blocked:
        safe = [n for n in all_safe if n not in guard_blocked]
        if safe:
            all_safe = safe
    forward_available = [n for n in all_safe if n not in visited_node_ids]
    prefer_water = _prefer_water_route(
        player, tasks or [], player_id, failed_task_ids or set(),
        enemy_busy_task_ids or set(), obstacle_nodes,
        map_gameplay=map_gameplay,
    )
    if forward_available:
        available = _filter_neighbors_for_route_preference(
            forward_available, current_node_id, prefer_water, map_gameplay,
            force_delivery=force_delivery,
            gate_node_id=gate_node_id,
            terminal_node_ids=terminal_node_ids,
        )
    else:
        available = _filter_neighbors_for_route_preference(
            all_safe or neighbors, current_node_id, prefer_water, map_gameplay,
            force_delivery=force_delivery,
            gate_node_id=gate_node_id,
            terminal_node_ids=terminal_node_ids,
        )
    all_safe = _filter_neighbors_for_route_preference(
        all_safe, current_node_id, prefer_water, map_gameplay,
        force_delivery=force_delivery,
        gate_node_id=gate_node_id,
        terminal_node_ids=terminal_node_ids,
    ) or all_safe
    logger.info(
        "_find_move_target: current=%s neighbors=%s available=%s visited=%s prefer_water=%s",
        current_node_id, neighbors, available, visited_node_ids, prefer_water,
    )

    if tasks is None:
        tasks = []
    if failed_task_ids is None:
        failed_task_ids = set()
    if enemy_busy_task_ids is None:
        enemy_busy_task_ids = set()

    goal_node = _get_goal_node(player, gate_node_id, terminal_node_ids, graph, current_node_id, weather, None, process_nodes)

    # Build remaining process nodes (exclude already-processed nodes at current visit)
    remaining_process_nodes = None
    if process_nodes:
        remaining_process_nodes = {
            nid: info for nid, info in process_nodes.items()
            if nid not in processed_node_ids
        }

    if force_delivery and goal_node:
        step = _find_direct_delivery_step(
            graph, current_node_id, player, gate_node_id, terminal_node_ids,
            weather, process_nodes, processed_node_ids,
        )
        if step:
            return step

    if goal_node and _should_use_task_aware_routing(
        player, phase, force_delivery, delivery_slack, round_num=round_num,
    ):
        if max_task_detour is None:
            max_task_detour = _max_task_detour_budget(delivery_slack, phase)
        task_candidates = all_safe or available
        task_step = _pick_task_aware_neighbor(
            graph, current_node_id, task_candidates, goal_node, tasks, player_id, player,
            gate_node_id, terminal_node_ids, weather, guard_blocked,
            remaining_process_nodes, failed_task_ids, enemy_busy_task_ids,
            obstacle_nodes, visited_node_ids, max_task_detour=max_task_detour,
            prefer_water=prefer_water, map_gameplay=map_gameplay,
        )
        if task_step and task_step in available:
            route_score, route_tasks, route_high = _route_task_stats(
                graph, current_node_id, task_step, goal_node, tasks, player_id, player,
                gate_node_id, terminal_node_ids, weather, guard_blocked,
                remaining_process_nodes, failed_task_ids, enemy_busy_task_ids,
                obstacle_nodes, max_task_detour=max_task_detour,
            )
            logger.info(
                "task-aware move %s->%s (routeTasks=%d high=%d score=%d myTaskScore=%d)",
                current_node_id, task_step, route_tasks, route_high, route_score,
                get_task_score(player),
            )
            return task_step

    if goal_node:
        # Build soft-blocked set: obstacles + visited + enemy guards
        soft_blocked = set(obstacle_nodes)
        soft_blocked.update(visited_node_ids)
        soft_blocked.update(guard_blocked)
        # Don't block the goal itself
        soft_blocked.discard(goal_node)

        # Use weighted Dijkstra first — prefers WATER(1250) over ROAD(1380) over MOUNTAIN(1780)
        step = graph.next_step_toward(current_node_id, goal_node, weather, soft_blocked, use_weighted=True, process_nodes=remaining_process_nodes)
        if step and step in available:
            return step
        # Fallback: unweighted BFS
        step = graph.next_step_toward(current_node_id, goal_node, weather, soft_blocked, use_weighted=False)
        if step and step in available:
            return step
        # Try weighted without soft-blocked (just obstacles)
        step = graph.next_step_toward(current_node_id, goal_node, weather, obstacle_nodes, use_weighted=True, process_nodes=remaining_process_nodes)
        if step and step in available:
            return step
        # Try BFS without any filter
        step = graph.next_step_toward(current_node_id, goal_node, weather, None, use_weighted=False)
        if step and step in available:
            return step
        # Pick neighbor with lowest weighted cost to goal
        best_alt = None
        best_alt_cost = float('inf')
        for n in available:
            path = graph.weighted_shortest_path(n, goal_node, weather, soft_blocked, remaining_process_nodes)
            if path:
                cost = sum(graph.edge_cost(path[i], path[i+1], weather, soft_blocked, remaining_process_nodes)
                           for i in range(len(path)-1))
                if cost < best_alt_cost:
                    best_alt_cost = cost
                    best_alt = n
        if best_alt:
            return best_alt

    # No goal: fall back to first available neighbor
    return available[0]


def _handle_contesting(
    match_id: str, round_num: int, player_id: int,
    player: dict, contests: list[dict] | None,
    events: list[dict] | None, active_contest_id: str,
    my_player: dict, all_players: list[dict], phase: str,
    on_water_route: bool = False,
    graph: MapGraph | None = None,
    gate_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    obstacle_nodes: set[str] | None = None,
) -> dict:
    """Handle CONTESTING state: choose window card (策略文档 §7)."""
    if terminal_node_ids is None:
        terminal_node_ids = []
    if obstacle_nodes is None:
        obstacle_nodes = set()

    contest_id = _find_contest_id(player_id, contests, events, active_contest_id)
    if not contest_id:
        return make_empty_action(match_id, round_num, player_id)

    # Determine contest type and pick card
    contest = _find_contest(contest_id, contests)
    contest_type = ""
    if contest:
        contest_type = contest.get("contestType") or contest.get("type", "")

    on_delivery_path = _contest_on_delivery_path(
        graph, get_current_node_id(my_player), gate_node_id,
        terminal_node_ids, contest, weather=None, obstacle_nodes=obstacle_nodes,
    )
    card = _choose_window_card(
        contest_type, contest, my_player, all_players, phase,
        on_water_route, on_delivery_path=on_delivery_path,
    )
    return make_action(match_id, round_num, player_id, [
        make_window_card_action(contest_id, card)
    ])


def _find_contest_id(
    player_id: int,
    contests: list[dict] | None,
    events: list[dict] | None,
    active_contest_id: str,
) -> str:
    """Find the contest ID for the current player."""
    if contests:
        for c in contests:
            if c.get("resolved", False) or c.get("status") == "SUPPRESSED":
                continue
            if c.get("redPlayerId") == player_id or c.get("bluePlayerId") == player_id:
                cid = c.get("contestId", "")
                if cid:
                    return cid
    if active_contest_id:
        return active_contest_id
    if events:
        for ev in reversed(events):
            if ev.get("type") == "WINDOW_CONTEST_START":
                payload = ev.get("payload", {})
                cid = payload.get("contestId", "")
                if cid:
                    return cid
    return ""


def _find_contest(contest_id: str, contests: list[dict] | None) -> dict | None:
    """Find contest dict by ID."""
    if contests:
        for c in contests:
            if c.get("contestId") == contest_id:
                return c
    return None


# 窗口牌克制表 (任务书 §5.4.4): CARD_BEATS[c] = c 能击败的牌集合。
# 献贡/兵争各克 2 张、只平 1 张(强牌); 验牒/强行各克 1 张、平 2 张(弱牌)。
CARD_BEATS: dict[str, set[str]] = {
    "YAN_DIE": {"QIANG_XING"},
    "QIANG_XING": {"XIAN_GONG"},
    "XIAN_GONG": {"YAN_DIE", "BING_ZHENG"},
    "BING_ZHENG": {"YAN_DIE", "QIANG_XING"},
}

# 各牌的相对成本罚分(资源稀缺度)。价值高的窗口会弱化罚分。
# 兵争消耗护卫行动点(全局仅4点,最稀缺); 强行会消耗宝贵马类(应留给加速);
# 献贡消耗1好果; 验牒消耗文书资源(过所/官凭,本就用于出牌,最廉价)。
CARD_COST_PENALTY: dict[str, float] = {
    "YAN_DIE": 1.0,
    "XIAN_GONG": 1.6,
    "QIANG_XING": 2.6,
    "BING_ZHENG": 3.2,
    "ABSTAIN": 0.0,
}

_ALL_EFFECTIVE_CARDS = ("YAN_DIE", "QIANG_XING", "XIAN_GONG", "BING_ZHENG")


def _window_card_result(mine: str, theirs: str) -> int:
    """本拍胜负 (任务书 §5.4.4): 胜=1, 平=0, 负=-1。"""
    if mine == theirs:
        return 0
    if theirs == "ABSTAIN":
        return 1 if mine != "ABSTAIN" else 0
    if mine == "ABSTAIN":
        return -1
    if theirs in CARD_BEATS.get(mine, ()):
        return 1
    if mine in CARD_BEATS.get(theirs, ()):
        return -1
    return 0


def _affordable_window_cards(p: dict | None) -> set[str]:
    """依据公开状态推断某方本拍买得起哪些有效牌 (任务书 §5.4.3)。"""
    if not p:
        return set()
    cards: set[str] = set()
    resources = get_player_resources(p)
    if resources.get("PASS_TOKEN", 0) + resources.get("OFFICIAL_PERMIT", 0) > 0:
        cards.add("YAN_DIE")
    if has_resource(p, "FAST_HORSE") or has_resource(p, "SHORT_HORSE"):
        cards.add("QIANG_XING")
    if get_good_fruit(p) >= 1 and get_freshness(p) >= 80:
        cards.add("XIAN_GONG")
    if get_action_points(p) > 0:
        cards.add("BING_ZHENG")
    return cards


def _contest_on_delivery_path(
    graph: MapGraph | None,
    current_node_id: str | None,
    gate_node_id: str,
    terminal_node_ids: list[str],
    contest: dict | None,
    weather: dict | None,
    obstacle_nodes: set[str],
) -> bool:
    if not graph or not current_node_id or not contest:
        return False
    target = contest.get("targetNodeId", "")
    if not target:
        return False
    goal = gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "")
    if not goal:
        return False
    return target in _path_nodes_to_goal(
        graph, current_node_id, goal, weather, obstacle_nodes,
    )


def _contest_value_profile(
    contest_type: str, contest: dict | None, on_water_route: bool,
    on_delivery_path: bool = False,
) -> tuple[bool, float]:
    """返回 (是否必争, 价值权重0..1)。价值越高越值得付成本、越可能弃权亏。"""
    if contest_type == "GATE":
        return True, 1.0
    if contest_type == "PASS":
        return False, 0.8
    if contest_type in ("TASK", "OBSTACLE"):
        score = contest.get("taskScore", 0) if contest else 0
        return (False, 0.6) if score >= 30 else (False, 0.0)
    if contest_type == "DOCK":
        if on_delivery_path:
            return True, 0.85
        return (False, 0.5) if on_water_route else (False, 0.0)
    if contest_type == "RESOURCE":
        return False, 0.45
    return False, 0.0


def _choose_window_card(
    contest_type: str, contest: dict | None,
    my_player: dict, all_players: list[dict], phase: str,
    on_water_route: bool = False,
    on_delivery_path: bool = False,
) -> str:
    """博弈式窗口出牌 (任务书 §5.4)。

    核心: 对手资源/好果/鲜度/行动点均为公开状态, 由此推断对手本拍能出哪些牌,
    再用正确克制表做期望胜点最大化, 同时按窗口价值权衡稀缺资源成本。
    不写死固定优先级, 随对手手牌与窗口价值自适应。
    """
    must_win, value = _contest_value_profile(
        contest_type, contest, on_water_route, on_delivery_path,
    )

    my_cards = _affordable_window_cards(my_player)
    # 低价值且非必争的窗口: 直接弃权, 保留资源。
    if not must_win and value <= 0.0:
        return "ABSTAIN"
    if not my_cards:
        return "ABSTAIN"

    opp = _find_opponent(all_players, my_player.get("playerId"))
    opp_cards = _affordable_window_cards(opp)

    # 估计对手本拍出牌的概率权重。
    opp_weights: dict[str, float] = {}
    if opp is None:
        # 完全观测不到对手对象(信息缺失): 保守假设各有效牌均可能。
        for c in _ALL_EFFECTIVE_CARDS:
            opp_weights[c] = 0.5
        opp_weights["ABSTAIN"] = 1.0
    elif opp_cards:
        for c in opp_cards:
            opp_weights[c] = 1.0
        # 窗口越重要, 对手越可能认真出牌 → 弃权权重越低。
        opp_weights["ABSTAIN"] = max(0.1, 1.0 - value)
    else:
        # 公开状态显示对手买不起任何有效牌 → 对手本拍只能弃权,
        # 此时任意有效牌皆胜, 交由成本罚分挑最省的一张。
        opp_weights["ABSTAIN"] = 1.0

    total_w = sum(opp_weights.values()) or 1.0

    def expected(card: str) -> float:
        return sum(w * _window_card_result(card, o) for o, w in opp_weights.items()) / total_w

    best_card = "ABSTAIN"
    best_adj = float("-inf") if must_win else 0.0
    for card in my_cards:
        exp = expected(card)
        # 成本罚分随窗口价值弱化, 但始终保留下限, 使期望相等时优选省资源的牌。
        penalty = CARD_COST_PENALTY.get(card, 1.0) * max(0.03, (1.0 - value) * 0.2)
        adj = exp - penalty
        if adj > best_adj:
            best_adj = adj
            best_card = card

    # 必争窗口(如宫门)绝不弃权: 退化为纯期望最高的有效牌。
    if must_win and best_card == "ABSTAIN":
        best_card = max(my_cards, key=expected)

    return best_card


def _get_pending_station_process_type(
    current_node_id: str | None,
    next_node_id: str,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
) -> str:
    if not current_node_id or next_node_id or not process_nodes:
        return ""
    if current_node_id in processed_node_ids:
        return ""

    process_type = process_nodes.get(current_node_id, {}).get("processType")
    if process_type and not is_verify_process(process_type):
        return process_type
    return ""


def _has_current_process_for_node(player: dict, current_node_id: str | None) -> bool:
    if not current_node_id:
        return False
    current_process = player.get("currentProcess")
    if not isinstance(current_process, dict):
        return False
    target_node_id = current_process.get("targetNodeId", "")
    object_key = current_process.get("objectKey", "")
    return target_node_id == current_node_id or object_key.startswith(f"PROCESS:{current_node_id}:")


def _resolve_guard_block_target(
    player: dict,
    route_blocked: set[str],
    guard_blocked_targets: set[str],
) -> str:
    """Node blocking our in-progress move (next hop with active enemy guard)."""
    next_node = player.get("nextNodeId", "")
    if next_node and next_node in guard_blocked_targets:
        return next_node
    return ""


def _make_squad_weaken_action(
    inquire_nodes: list[dict],
    target_node_id: str,
    my_team_id: str,
    player_id: int,
    player: dict,
) -> dict | None:
    if not target_node_id or get_squad_count(player) < 2:
        return None
    for node in inquire_nodes:
        if node.get("nodeId") != target_node_id:
            continue
        guard = node.get("guard", {})
        if is_enemy_guard(guard, my_team_id, player_id):
            if guard.get("defense", 0) > 0:
                return make_squad_weaken_action(target_node_id)
            return None
    return None


def _wait_and_weaken_guard(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    inquire_nodes: list[dict],
    target_node_id: str,
    my_team_id: str,
    *,
    force_delivery: bool = False,
    graph: MapGraph | None = None,
    gate_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    weather: dict | None = None,
    route_blocked: set[str] | None = None,
    guard_blocked_targets: set[str] | None = None,
    avoid_route_nodes: set[str] | None = None,
    guard_stuck_rounds: int = 0,
    guard_stuck_target: str = "",
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
) -> dict:
    """WAIT (主车队) + SQUAD_WEAKEN (小分队) 每帧削弱设卡直到通行。"""
    if route_blocked is None:
        route_blocked = set()
    if guard_blocked_targets is None:
        guard_blocked_targets = set()
    if avoid_route_nodes is None:
        avoid_route_nodes = set()
    if processed_node_ids is None:
        processed_node_ids = set()
    if terminal_node_ids is None:
        terminal_node_ids = []

    guard = _get_node_guard(inquire_nodes, target_node_id)
    if not is_enemy_guard(guard, my_team_id, player_id):
        player_state = player.get("state", "")
        if player_state == "IDLE" and force_delivery and graph and get_current_node_id(player):
            detour = _find_delivery_detour_step(
                graph, get_current_node_id(player), player, gate_node_id,
                terminal_node_ids, weather, process_nodes, processed_node_ids,
                route_blocked, avoid_nodes=avoid_route_nodes | {target_node_id},
            )
            if detour:
                logger.info(
                    "Round %d: guard cleared/absent at %s, detour via %s",
                    round_num, target_node_id, detour,
                )
                return make_action(match_id, round_num, player_id, [make_move_action(detour)])

    if (
        player.get("state", "") == "IDLE"
        and force_delivery
        and graph
        and _should_detour_force_delivery(
            target_node_id, route_blocked, avoid_route_nodes,
            guard_stuck_rounds, guard_stuck_target,
        )
    ):
        detour = _find_delivery_detour_step(
            graph, get_current_node_id(player) or "", player, gate_node_id,
            terminal_node_ids, weather, process_nodes, processed_node_ids,
            route_blocked, avoid_nodes=avoid_route_nodes | {target_node_id},
        )
        if detour:
            logger.info(
                "Round %d: stuck %d rounds on %s, detour via %s",
                round_num, guard_stuck_rounds, target_node_id, detour,
            )
            return make_action(match_id, round_num, player_id, [make_move_action(detour)])

    msg = make_action(match_id, round_num, player_id, [make_wait_action()])
    squad = _make_squad_weaken_action(
        inquire_nodes, target_node_id, my_team_id, player_id, player,
    )
    if squad:
        logger.info("Round %d: WAIT + squad weaken at %s", round_num, target_node_id)
        return _append_squad_action(msg, squad)
    logger.info(
        "Round %d: WAIT at blocked %s (stuck=%d, no weaken)",
        round_num, target_node_id, guard_stuck_rounds,
    )
    return msg


def _handle_moving(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph, weather: dict | None, phase: str,
) -> dict:
    """Handle MOVING state: can use horse or rush_speed."""
    if _has_move_speed_buff(player):
        return make_empty_action(match_id, round_num, player_id)
    # Use FAST_HORSE if available and on a long road segment
    if has_resource(player, "FAST_HORSE"):
        return make_action(match_id, round_num, player_id, [
            make_use_resource_action("FAST_HORSE")
        ])
    # Use SHORT_HORSE if available
    if has_resource(player, "SHORT_HORSE"):
        return make_action(match_id, round_num, player_id, [
            make_use_resource_action("SHORT_HORSE")
        ])
    return make_empty_action(match_id, round_num, player_id)


def _handle_blocked_by_guard(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph,
    current_node_id: str, gate_node_id: str, terminal_node_ids: list[str],
    weather: dict | None, blocked: set[str] | None, inquire_nodes: list[dict],
    process_nodes: dict[str, dict] | None = None,
) -> dict:
    """Handle MOVE_BLOCKED_BY_GUARD error (策略文档 §3.4)."""
    if blocked is None:
        blocked = set()
    neighbors = graph.get_neighbors(current_node_id)
    goal = gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "")

    # Detour via unblocked neighbor (e.g. S09→S05 when S10 guarded)
    best_detour = None
    best_cost = float("inf")
    for n in neighbors:
        if n in blocked:
            continue
        if not goal:
            return make_action(match_id, round_num, player_id, [make_move_action(n)])
        path = graph.weighted_shortest_path(n, goal, weather, blocked, process_nodes)
        if path:
            cost = sum(
                graph.edge_cost(path[i], path[i + 1], weather, blocked, process_nodes)
                for i in range(len(path) - 1)
            )
            if cost < best_cost:
                best_cost = cost
                best_detour = n
    if best_detour:
        logger.info("Round %d: Detour via %s to avoid guard", round_num, best_detour)
        return make_action(match_id, round_num, player_id, [make_move_action(best_detour)])

    # No detour: BREAK_GUARD or FORCED_PASS on guarded neighbor
    for n in neighbors:
        if n not in blocked:
            continue
        for node in inquire_nodes:
            if node.get("nodeId") != n:
                continue
            guard = node.get("guard", {})
            if is_enemy_guard(guard, get_team_id(player), player_id):
                good, bad = _break_guard_investment(player)
                if good + bad > 0:
                    logger.info(
                        "Round %d: BREAK_GUARD at %s (gf=%d bf=%d)",
                        round_num, n, good, bad,
                    )
                    return make_action(match_id, round_num, player_id, [
                        make_break_guard_action(n, good_fruit=good, bad_fruit=bad)
                    ])
        logger.info("Round %d: FORCED_PASS at guarded %s", round_num, n)
        return make_action(match_id, round_num, player_id, [
            make_forced_pass_action(n)
        ])

    return make_empty_action(match_id, round_num, player_id)


def _task_hold_blocked_by_enemy(
    pending_task_id: str,
    enemy_busy_task_ids: set[str],
) -> bool:
    return bool(pending_task_id and pending_task_id in enemy_busy_task_ids)


def _t04_targets_obstacle(
    task: dict,
    current_node_id: str,
    obstacle_nodes: set[str],
    graph: MapGraph | None,
) -> bool:
    task_node = task.get("nodeId", "")
    if not task_node:
        return False
    if task_node in obstacle_nodes:
        return True
    if graph and current_node_id and task_node in graph.get_neighbors(current_node_id):
        return task_node in obstacle_nodes
    return False


def _player_processing_task(player: dict) -> bool:
    process = player.get("currentProcess")
    if not isinstance(process, dict):
        return False
    if process.get("action") == "CLAIM_TASK" or process.get("type") == "CLAIM_TASK":
        return True
    return str(process.get("objectKey", "")).startswith("TASK:")


def _retry_task_at_current_node(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node_id: str,
    tasks: list[dict],
    failed_task_ids: set[str],
    preferred_task_id: str = "",
    enemy_busy_task_ids: set[str] | None = None,
    map_gameplay: MapGameplayContext | None = None,
) -> dict | None:
    if enemy_busy_task_ids is None:
        enemy_busy_task_ids = set()
    if isinstance(player.get("currentProcess"), dict):
        return None

    neighbors = graph.get_neighbors(current_node_id) if graph else None
    task = None
    if preferred_task_id:
        for candidate in tasks:
            if candidate.get("taskId", "") == preferred_task_id:
                task_node = candidate.get("nodeId", "")
                if (
                    task_node == current_node_id
                    or (neighbors is not None and task_node in neighbors and get_task_template_id(candidate).startswith("T04"))
                ):
                    task = candidate
                break
    if not task:
        task = find_task_at_node(
            tasks, current_node_id, player_id,
            graph_neighbors=neighbors,
            enemy_busy_task_ids=enemy_busy_task_ids,
        )
    if not task:
        return None
    if not task.get("active", False) or task.get("completed", False) or task.get("failed", False):
        return None
    owner = task.get("ownerPlayerId", 0)
    if owner != 0 and owner != player_id:
        return None
    protection = task.get("protectionPlayerId", 0)
    if protection != 0 and protection != player_id:
        return None

    task_id = task.get("taskId", "")
    if not task_id or task_id in failed_task_ids or task_id in enemy_busy_task_ids:
        return None
    template_id = get_task_template_id(task)
    if template_id.startswith("T06") and not has_resource(player, "FAST_HORSE") and not has_resource(player, "SHORT_HORSE"):
        return None
    prefer_water = _prefer_water_route(
        player, tasks, player_id, failed_task_ids, enemy_busy_task_ids, set(),
        map_gameplay=map_gameplay,
    )
    task_node = task.get("nodeId", current_node_id)
    if not prefer_water and _is_water_task_location(task_node, task, map_gameplay):
        logger.info(
            "Round %d: Skip retry water task %s at %s (waterScore<%d)",
            round_num, task_id, task_node, _water_task_min(map_gameplay),
        )
        return None
    expire_round = task.get("expireRound", 0)
    if expire_round > 0 and round_num >= expire_round:
        return None

    logger.info("Round %d: Retrying task %s (template=%s) at %s", round_num, task_id, template_id, current_node_id)
    return make_action(match_id, round_num, player_id, [make_claim_task_action(task_id)])


def _handle_tasks(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph, current_node_id: str,
    tasks: list[dict], my_player_id: int, phase: str,
    weather: dict | None, blocked: set[str] | None,
    goal_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    obstacle_nodes: set[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    failed_task_ids: set[str] | None = None,
    enemy_busy_task_ids: set[str] | None = None,
    max_task_detour: int | None = None,
    allow_detour: bool = True,
    delivery_slack: float = 999.0,
    map_gameplay: MapGameplayContext | None = None,
    task_claimed_this_stop: bool = False,
    inquire_nodes: list[dict] | None = None,
    my_team_id: str = "",
    guard_blocked_targets: set[str] | None = None,
) -> dict | None:
    """Handle task claiming strategy (策略文档 §5).

    Returns action dict or None.
    """
    if terminal_node_ids is None:
        terminal_node_ids = []
    if obstacle_nodes is None:
        obstacle_nodes = set()
    if processed_node_ids is None:
        processed_node_ids = set()
    if visited_node_ids is None:
        visited_node_ids = set()
    if failed_task_ids is None:
        failed_task_ids = set()
    if enemy_busy_task_ids is None:
        enemy_busy_task_ids = set()
    if max_task_detour is None:
        max_task_detour = MAX_TASK_DETOUR_COST
    if inquire_nodes is None:
        inquire_nodes = []
    if guard_blocked_targets is None:
        guard_blocked_targets = set()
    if not my_team_id:
        my_team_id = get_team_id(player)

    water_nodes = _map_ctx(map_gameplay).water_route_nodes
    my_task_score = get_task_score(player)
    prefer_water = _prefer_water_route(
        player, tasks, my_player_id, failed_task_ids, enemy_busy_task_ids, obstacle_nodes,
        map_gameplay=map_gameplay,
    )

    if _player_processing_task(player):
        return None

    # Check if we're currently processing a task (策略文档 §5.2: 同时仅处理1个任务实例)
    for task in tasks:
        if (task.get("ownerPlayerId") == my_player_id
                and task.get("active", False)
                and not task.get("completed", False)
                and not task.get("failed", False)):
            task_id = task.get("taskId", "")
            task_node = task.get("nodeId", "")
            template_id = get_task_template_id(task)
            neighbors = graph.get_neighbors(current_node_id) if graph else []
            can_retry_here = (
                task_node == current_node_id
                or (template_id.startswith("T04") and task_node in neighbors)
            )
            if (
                task_id
                and not task_claimed_this_stop
                and task_id not in failed_task_ids
                and task_id not in enemy_busy_task_ids
                and can_retry_here
            ):
                logger.info(
                    "Round %d: Retrying owned active task %s (template=%s) at %s",
                    round_num, task_id, template_id, current_node_id,
                )
                return make_action(match_id, round_num, player_id, [
                    make_claim_task_action(task_id)
                ])
            return None

    # Try to claim task at current node (prioritized by score/round)
    task = find_task_at_node(
        tasks, current_node_id, my_player_id,
        graph_neighbors=graph.get_neighbors(current_node_id) if graph else None,
        enemy_busy_task_ids=enemy_busy_task_ids,
    )
    if task:
        template_id = get_task_template_id(task)
        if template_id.startswith("T06") and not has_resource(player, "FAST_HORSE") and not has_resource(player, "SHORT_HORSE"):
            logger.debug("Round %d: Skipping T06 task (no horse)", round_num)
            task = None
        if task and template_id.startswith("T04") and not _t04_targets_obstacle(
            task, current_node_id, obstacle_nodes, graph,
        ):
            logger.debug("Round %d: Skipping T04 at %s (no obstacle)", round_num, current_node_id)
            task = None
        if task and not prefer_water:
            task_node = task.get("nodeId", "")
            if _is_water_task_location(task_node, task, map_gameplay):
                logger.info(
                    "Round %d: Skip water task %s at %s (waterScore<%d)",
                    round_num, task.get("taskId", ""), task_node, _water_task_min(map_gameplay),
                )
                task = None

    if task:
        # Check expireRound (策略文档 §5.2: 关注expireRound)
        expire_round = task.get("expireRound", 0)
        if expire_round > 0 and round_num >= expire_round:
            logger.debug("Round %d: Task %s expired", round_num, task.get("taskId", ""))
            task = None

    if task:
        task_id = task.get("taskId", "")
        if task_id and task_id in failed_task_ids:
            logger.debug("Round %d: Skipping failed task %s", round_num, task_id)
            task = None

    if task and task_claimed_this_stop:
        logger.debug("Round %d: Already claimed task at %s this stop", round_num, current_node_id)
        task = None

    if task:
        task_node = task.get("nodeId", current_node_id)
        if _task_is_away_from_gate(
            graph, current_node_id, task_node, goal_node_id, weather, obstacle_nodes,
        ):
            logger.info(
                "Round %d: Skip task %s at %s (away from gate)",
                round_num, task.get("taskId", ""), task_node,
            )
            task = None

    if task:
        task_id = task.get("taskId", "")
        if task_id:
            logger.info("Round %d: Claiming task %s (template=%s) at %s", round_num, task_id, template_id, current_node_id)
            return make_action(match_id, round_num, player_id, [
                make_claim_task_action(task_id)
            ])

    # Look for nearby tasks within detour budget (交付余量内尽量多拿)
    if allow_detour and max_task_detour > 0:
        candidates = []
        for task in tasks:
            if not task.get("active", False) or task.get("completed", False) or task.get("failed", False):
                continue
            owner = task.get("ownerPlayerId", 0)
            if owner != 0 and owner != my_player_id:
                continue
            protection = task.get("protectionPlayerId", 0)
            if protection != 0 and protection != my_player_id:
                continue
            task_node = task.get("nodeId", "")
            if not task_node:
                continue
            if task.get("taskId", "") in enemy_busy_task_ids:
                continue

            # T06: skip if no horse
            tid = get_task_template_id(task)
            if tid.startswith("T06") and not has_resource(player, "FAST_HORSE") and not has_resource(player, "SHORT_HORSE"):
                continue
            if tid.startswith("T04") and not _t04_targets_obstacle(
                task, current_node_id, obstacle_nodes, graph,
            ):
                continue

            # Skip tasks previously rejected with RESOURCE_NOT_ENOUGH
            if task.get("taskId", "") in failed_task_ids:
                continue

            # Check expireRound
            expire_round = task.get("expireRound", 0)
            if expire_round > 0 and round_num >= expire_round:
                continue

            # Check detour cost
            detour = _calc_detour_cost(graph, current_node_id, task_node, goal_node_id, terminal_node_ids, weather, blocked, player, process_nodes)
            if detour > max_task_detour:
                continue
            if _task_involves_backtrack(
                task_node, current_node_id, visited_node_ids, graph,
                weather, blocked, process_nodes,
            ):
                logger.debug(
                    "Round %d: Skip task %s at %s (backtrack via visited)",
                    round_num, task.get("taskId", ""), task_node,
                )
                continue
            if _task_is_away_from_gate(
                graph, current_node_id, task_node, goal_node_id, weather, obstacle_nodes,
            ):
                logger.info(
                    "Round %d: Skip task %s at %s (away from gate)",
                    round_num, task.get("taskId", ""), task_node,
                )
                continue
            if _should_skip_mid_task_detour(
                round_num, current_node_id, task_node, map_gameplay,
            ):
                logger.info(
                    "Round %d: Skip task %s at %s (late-game mid-route detour)",
                    round_num, task.get("taskId", ""), task_node,
                )
                continue
            if _task_detour_unacceptable(detour, delivery_slack, task):
                logger.debug(
                    "Round %d: Skip task %s at %s (detour=%d slack=%.0f)",
                    round_num, task.get("taskId", ""), task_node, detour, delivery_slack,
                )
                continue
            if _node_has_enemy_guard(inquire_nodes, task_node, my_team_id, my_player_id):
                logger.info(
                    "Round %d: Skip task %s at %s (enemy guard)",
                    round_num, task.get("taskId", ""), task_node,
                )
                continue
            if task_node in GATE_CORRIDOR_NODES and task_node in guard_blocked_targets:
                logger.info(
                    "Round %d: Skip task %s at %s (gate corridor blocked)",
                    round_num, task.get("taskId", ""), task_node,
                )
                continue
            if not prefer_water and (
                task_node in _map_ctx(map_gameplay).water_route_nodes
                or (task.get("routeBucket") or "ROAD") == "WATER"
            ):
                continue
            # Score per round priority (策略文档 §5.1)
            spr = 0.0
            for prefix, (score, proc_round, score_per_round) in TASK_PRIORITY.items():
                if tid.startswith(prefix):
                    spr = score_per_round
                    break
            candidates.append((task, detour, spr))

        if candidates:
            # Sort by: score-per-round descending, then detour ascending
            candidates.sort(key=lambda x: (-x[2], x[1]))
            for best_task, _detour, _spr in candidates:
                task_node = best_task.get("nodeId", "")
                template_id = get_task_template_id(best_task)
                soft_blocked = set(obstacle_nodes)
                if blocked:
                    soft_blocked.update(blocked)
                if template_id.startswith("T04"):
                    soft_blocked.discard(task_node)
                elif task_node in soft_blocked:
                    continue
                step = graph.next_step_toward(
                    current_node_id, task_node, weather, soft_blocked,
                    use_weighted=True, process_nodes=process_nodes,
                )
                if not step:
                    step = graph.next_step_toward(
                        current_node_id, task_node, weather, obstacle_nodes,
                        use_weighted=True, process_nodes=process_nodes,
                    )
                if not step or step in visited_node_ids:
                    continue
                if step in soft_blocked or step in guard_blocked_targets:
                    continue
                logger.info(
                    "Round %d: Moving toward task at %s (template=%s), step=%s",
                    round_num, task_node, get_task_template_id(best_task), step,
                )
                return make_action(match_id, round_num, player_id, [make_move_action(step)])

    return None


def _calc_detour_cost(
    graph: MapGraph, current: str, task_node: str,
    gate_node_id: str, terminal_node_ids: list[str],
    weather: dict | None, blocked: set[str] | None,
    player: dict,
    process_nodes: dict[str, dict] | None = None,
) -> int:
    """Calculate the extra weighted cost of detouring to a task node vs direct route."""
    goal = _get_goal_node(player, gate_node_id, terminal_node_ids, graph, current, weather, blocked, process_nodes)
    if not goal:
        return 999

    def _weighted_path_cost(a: str, b: str) -> float:
        path = graph.weighted_shortest_path(a, b, weather, blocked, process_nodes)
        if not path:
            return float('inf')
        return sum(graph.edge_cost(path[i], path[i+1], weather, blocked, process_nodes)
                   for i in range(len(path)-1))

    direct = _weighted_path_cost(current, goal)
    via_task = _weighted_path_cost(current, task_node) + _weighted_path_cost(task_node, goal)

    if direct == float('inf') or via_task == float('inf'):
        return 999

    # Normalize to approximate frame cost (divide by 1000 to get ~frame units)
    return int((via_task - direct) / 1000)


def _should_defer_ice_box(
    player: dict,
    last_move_error: str,
    route_blocked: set[str],
    guard_blocked_targets: set[str] | None = None,
    avoid_route_nodes: set[str] | None = None,
) -> bool:
    """边上移动或交设卡税时不能 USE_RESOURCE，须落地后再用冰鉴。"""
    state = player.get("state", "")
    if state == "MOVING":
        return True
    if state == "WAITING":
        next_node = player.get("nextNodeId", "")
        if _is_paying_guard_travel_tax(
            state, player, last_move_error, next_node, route_blocked,
            guard_blocked_targets=guard_blocked_targets,
            avoid_route_nodes=avoid_route_nodes,
        ):
            return True
        if next_node:
            return True
    return False


def _should_use_ice_box(
    player: dict,
    map_gameplay: MapGameplayContext | None,
    phase: str = "",
    force_delivery: bool = False,
    graph: MapGraph | None = None,
    current_node_id: str | None = None,
    weather: dict | None = None,
    gate_node_id: str = "",
) -> tuple[bool, str]:
    """Decide whether to consume ICE_BOX this frame."""
    if not has_resource(player, "ICE_BOX"):
        return False, ""
    freshness = get_freshness(player)
    threshold = _ice_threshold(map_gameplay)
    rush_threshold = _ice_rush_threshold(map_gameplay)

    if force_delivery and freshness < rush_threshold:
        return True, "force-delivery"

    if phase == "RUSH" and freshness < rush_threshold:
        return True, "rush-consume"

    if graph and current_node_id and gate_node_id:
        hops = graph.path_length(current_node_id, gate_node_id, weather, None)
        if hops <= ICE_BOX_NEAR_GATE_HOPS and freshness < rush_threshold:
            return True, "near-gate"

    if graph and current_node_id and weather:
        penalized = _get_weather_penalized_routes(weather)
        if "MOUNTAIN" in penalized and freshness < threshold + 2:
            next_step = graph.next_step_toward(
                current_node_id, gate_node_id, weather, None, use_weighted=True,
            ) if gate_node_id else None
            if next_step and graph.get_edge_route_type(current_node_id, next_step) == "MOUNTAIN":
                return True, "hot-mountain-ahead"

    if freshness < threshold:
        return True, "below-threshold"
    return False, ""


def _try_use_ice_box(
    match_id: str, round_num: int, player_id: int, player: dict,
    map_gameplay: MapGameplayContext | None = None,
    phase: str = "",
    force_delivery: bool = False,
    graph: MapGraph | None = None,
    current_node_id: str | None = None,
    weather: dict | None = None,
    gate_node_id: str = "",
) -> dict | None:
    """鲜度低于阈值、或冲刺/交付前兜底时使用冰鉴。"""
    should, reason = _should_use_ice_box(
        player, map_gameplay, phase, force_delivery,
        graph, current_node_id, weather, gate_node_id,
    )
    if not should:
        return None
    freshness = get_freshness(player)
    logger.info(
        "Round %d: Using ICE_BOX (freshness=%.1f, reason=%s)",
        round_num, freshness, reason,
    )
    return make_action(match_id, round_num, player_id, [
        make_use_resource_action("ICE_BOX")
    ])


def _has_viable_own_scout_marker(
    inquire_nodes: list[dict],
    node_id: str,
    my_team_id: str,
    frames_needed: float = 0,
) -> bool:
    """节点上是否已有本队探路标记，且剩余有效帧覆盖到达时刻。"""
    node = _get_inquire_node(inquire_nodes, node_id)
    if not node or not my_team_id:
        return False
    for marker in node.get("scouted") or []:
        if not isinstance(marker, dict):
            continue
        if str(marker.get("teamId", "")) != my_team_id:
            continue
        if int(marker.get("remainingTriggers") or 0) <= 0:
            continue
        remain = float(marker.get("remainRound") or 0)
        if remain <= 0:
            continue
        if frames_needed <= 0 or remain >= frames_needed:
            return True
    return False


def _process_scout_value(process_type: str, process_round: int = 0) -> int:
    frames = int(process_round or PROCESS_COST_FRAMES.get(process_type, 0) or 0)
    if frames <= 2:
        return 0
    return min(3, frames - 2)


def _estimate_travel_frames(
    graph: MapGraph,
    from_id: str,
    to_id: str,
    weather: dict | None,
    obstacle_nodes: set[str] | None,
    process_nodes: dict[str, dict] | None,
) -> float:
    path = graph.weighted_shortest_path(from_id, to_id, weather, obstacle_nodes, process_nodes)
    if not path or len(path) < 2:
        return float("inf")
    total = sum(
        graph.edge_cost(path[i], path[i + 1], weather, obstacle_nodes, process_nodes)
        for i in range(len(path) - 1)
    )
    return max(1.0, total / 1000.0)


def _estimate_squad_scout_delay(
    graph: MapGraph,
    current_node_id: str,
    target_id: str,
    weather: dict | None,
) -> int:
    hops = graph.path_length(current_node_id, target_id, weather, None)
    if hops == float("inf") or hops <= 0:
        return SQUAD_SCOUT_MAX_DELAY
    return min(SQUAD_SCOUT_MAX_DELAY, max(SQUAD_SCOUT_MIN_DELAY, int(math.ceil(hops / 3))))


def _planned_route_process_targets(
    graph: MapGraph,
    current_node_id: str,
    goal_node_id: str,
    weather: dict | None,
    obstacle_nodes: set[str] | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str] | None,
    visited_node_ids: set[str] | None,
) -> list[tuple[str, dict, float]]:
    """计划路径上待处理的站点：(node_id, info, travel_frames)。"""
    if not graph or not current_node_id or not goal_node_id or not process_nodes:
        return []
    processed = processed_node_ids or set()
    visited = visited_node_ids or set()
    path = graph.weighted_shortest_path(
        current_node_id, goal_node_id, weather, obstacle_nodes, process_nodes,
    )
    if not path:
        path = graph.shortest_path(current_node_id, goal_node_id, weather, obstacle_nodes) or []
    targets: list[tuple[str, dict, float]] = []
    for nid in path:
        if nid == current_node_id or nid in processed or nid in visited:
            continue
        info = process_nodes.get(nid)
        if not info or not info.get("processType"):
            continue
        travel = _estimate_travel_frames(
            graph, current_node_id, nid, weather, obstacle_nodes, process_nodes,
        )
        if travel == float("inf"):
            continue
        targets.append((nid, info, travel))
    return targets


def _pick_scout_target(
    graph: MapGraph,
    current_node_id: str,
    goal_node_id: str,
    weather: dict | None,
    obstacle_nodes: set[str] | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str] | None,
    visited_node_ids: set[str] | None,
    inquire_nodes: list[dict],
    my_team_id: str,
    gate_corridor_only: bool = False,
) -> tuple[str, float, float, str] | None:
    """选取最佳探路目标：(node_id, route_dist, travel_frames, mode=intel|squad)。"""
    best: tuple[str, float, float, str] | None = None
    best_score = -1.0
    for nid, info, travel in _planned_route_process_targets(
        graph, current_node_id, goal_node_id, weather, obstacle_nodes,
        process_nodes, processed_node_ids, visited_node_ids,
    ):
        if gate_corridor_only:
            hops_to_gate = graph.path_length(nid, goal_node_id, weather, obstacle_nodes)
            if hops_to_gate == float("inf") or hops_to_gate > FINAL_CORRIDOR_GATE_HOPS:
                continue
        marker_needed = max(3.0, travel * 0.5)
        if _has_viable_own_scout_marker(inquire_nodes, nid, my_team_id, marker_needed):
            continue
        if travel > SCOUT_MARKER_VALID_FRAMES:
            continue
        route_dist = graph.min_route_distance(current_node_id, nid, weather, obstacle_nodes)
        if route_dist == float("inf"):
            continue
        pt = info.get("processType", "")
        pr = int(info.get("processRound") or 0)
        value = _process_scout_value(pt, pr)
        if value <= 0:
            continue
        score = value * 10.0 - travel * 0.3
        if nid in GATE_CORRIDOR_NODES:
            score += 8.0
        if score > best_score:
            best_score = score
            mode = "intel" if route_dist <= INTEL_MAX_DISTANCE else "squad"
            best = (nid, route_dist, travel, mode)
    return best


def _try_use_intel(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node_id: str,
    weather: dict | None,
    map_gameplay: MapGameplayContext | None,
    visited_node_ids: set[str] | None,
    inquire_nodes: list[dict] | None = None,
    my_team_id: str = "",
    gate_node_id: str = "",
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
    gate_corridor_only: bool = False,
) -> dict | None:
    """对计划路径上的处理站使用情报探路（距离 ≤15）。"""
    if not has_resource(player, "INTEL"):
        return None
    prof = _profile(map_gameplay)
    if not prof.intel_enabled:
        return None
    if not gate_node_id or not process_nodes or not inquire_nodes:
        return None
    if not my_team_id:
        my_team_id = get_team_id(player)
    pick = _pick_scout_target(
        graph, current_node_id, gate_node_id, weather, None,
        process_nodes, processed_node_ids, visited_node_ids,
        inquire_nodes, my_team_id, gate_corridor_only=gate_corridor_only,
    )
    if not pick:
        return None
    target, route_dist, travel, mode = pick
    if mode != "intel":
        return None
    logger.info(
        "Round %d: Using INTEL on process node %s (routeDist=%.1f, eta=%.0f)",
        round_num, target, route_dist, travel,
    )
    return make_action(match_id, round_num, player_id, [
        make_use_resource_action("INTEL", target)
    ])


def _try_squad_scout(
    match_id: str,
    round_num: int,
    player_id: int,
    graph: MapGraph,
    current_node_id: str,
    gate_node_id: str,
    weather: dict | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str] | None,
    visited_node_ids: set[str] | None,
    inquire_nodes: list[dict],
    my_team_id: str,
    squad_count: int,
    scout_min: int,
    my_task_score: int,
) -> dict | None:
    """远程小分队探路：距离 >15 的处理站，且标记能在到达前生效。"""
    gate_corridor_only = my_task_score >= TASK_SCORE_TARGET
    min_squad = 2 if gate_corridor_only else scout_min
    if squad_count < min_squad:
        return None
    pick = _pick_scout_target(
        graph, current_node_id, gate_node_id, weather, None,
        process_nodes, processed_node_ids, visited_node_ids,
        inquire_nodes, my_team_id, gate_corridor_only=gate_corridor_only,
    )
    if not pick:
        return None
    target, route_dist, travel, mode = pick
    if mode != "squad":
        return None
    delay = _estimate_squad_scout_delay(graph, current_node_id, target, weather)
    if travel <= delay:
        return None
    if travel > delay + SCOUT_MARKER_VALID_FRAMES - 5:
        return None
    logger.info(
        "Round %d: Squad scout process node %s (routeDist=%.1f, eta=%.0f, delay=%d)",
        round_num, target, route_dist, travel, delay,
    )
    return make_action(match_id, round_num, player_id, [
        make_squad_scout_action(target)
    ])


def _squad_clear_allowed(
    round_num: int,
    node_id: str,
    on_my_path: bool,
    force_delivery: bool,
    current_node_id: str,
    gate_node_id: str,
    graph: MapGraph | None,
    weather: dict | None,
    is_next_hop: bool = False,
) -> bool:
    """统一的小分队清障时机判断。"""
    if not on_my_path:
        return False
    if is_next_hop:
        return True
    if force_delivery:
        return True
    if node_id in GATE_CORRIDOR_NODES:
        if round_num >= SQUAD_CLEAR_MIN_ROUND:
            return True
        if graph and gate_node_id and current_node_id:
            hops = graph.path_length(current_node_id, gate_node_id, weather, None)
            return hops != float("inf") and hops <= NEAR_GATE_RESOURCE_HOPS
        return False
    return round_num >= SQUAD_CLEAR_MIDMAP_MIN_ROUND


def _try_claim_ice_box(
    match_id: str, round_num: int, player_id: int,
    player: dict, current_node: dict | None,
) -> dict | None:
    """路途中遇到冰鉴就领取，不受距宫门距离等限制。"""
    if current_node is None or has_resource(player, "ICE_BOX"):
        return None
    current_node_id = current_node.get("nodeId", "")
    for rtype, _count in find_available_resources(current_node):
        if rtype == "ICE_BOX":
            logger.info("Round %d: Claiming ICE_BOX at %s", round_num, current_node_id)
            return make_action(match_id, round_num, player_id, [
                make_claim_resource_action(current_node_id, rtype)
            ])
    return None


def _handle_resources(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph, current_node_id: str,
    current_node: dict | None, phase: str, weather: dict | None,
    gate_node_id: str = "",
    process_nodes: dict[str, dict] | None = None,
    map_gameplay: MapGameplayContext | None = None,
) -> dict | None:
    """Handle resource claiming strategy (策略文档 §6).

    Returns action dict or None.
    """
    if current_node is None:
        return None
    if phase == "RUSH" and get_task_score(player) >= TASK_SCORE_STRETCH:
        return None
    if round_num >= 520:
        return None

    ctx = _map_ctx(map_gameplay)
    prof = _profile(map_gameplay)
    hops_to_gate = _hops_to_gate(graph, current_node_id, gate_node_id, weather, None)
    near_gate = hops_to_gate <= prof.near_gate_skip_permit_hops
    on_final_approach = hops_to_gate <= FINAL_CORRIDOR_GATE_HOPS
    at_palace_transfer = _is_palace_transfer_node(process_nodes, current_node_id)
    in_gate_corridor = current_node_id in GATE_CORRIDOR_NODES

    resources = find_available_resources(current_node)
    if not resources:
        return None

    my_resources = get_player_resources(player)

    HIGH_VALUE_RESOURCES = {"FAST_HORSE", "SHORT_HORSE", "ICE_BOX"}
    WINDOW_RESOURCES = {"OFFICIAL_PERMIT", "PASS_TOKEN"}

    for rtype, count in resources:
        if my_resources.get(rtype, 0) >= 1 and rtype in HIGH_VALUE_RESOURCES:
            continue
        if rtype in HIGH_VALUE_RESOURCES:
            logger.info("Round %d: Claiming resource %s at %s", round_num, rtype, current_node_id)
            return make_action(match_id, round_num, player_id, [
                make_claim_resource_action(current_node_id, rtype)
            ])
        if rtype == "INTEL" and prof.intel_enabled:
            if get_task_score(player) >= TASK_SCORE_TARGET or near_gate:
                continue
            if my_resources.get("INTEL", 0) >= 1:
                continue
            logger.info("Round %d: Claiming INTEL at %s", round_num, current_node_id)
            return make_action(match_id, round_num, player_id, [
                make_claim_resource_action(current_node_id, rtype)
            ])
        if rtype in WINDOW_RESOURCES:
            # 变种地图过所/官凭可能在 S07 等中段节点，仅入关走廊才跳过
            if near_gate and in_gate_corridor:
                continue
            if (
                near_gate
                and rtype == "PASS_TOKEN"
                and current_node_id not in ctx.pass_token_nodes
            ):
                continue
            total_permits = (
                my_resources.get("OFFICIAL_PERMIT", 0)
                + my_resources.get("PASS_TOKEN", 0)
            )
            if on_final_approach:
                if total_permits >= GUARD_RESERVE_FOR_GATE:
                    continue
                logger.info(
                    "Round %d: Claiming %s at %s (final corridor gate reserve)",
                    round_num, rtype, current_node_id,
                )
                return make_action(match_id, round_num, player_id, [
                    make_claim_resource_action(current_node_id, rtype)
                ])
            if at_palace_transfer and total_permits < GUARD_RESERVE_FOR_GATE:
                logger.info(
                    "Round %d: Claiming %s at %s (palace transfer gate reserve)",
                    round_num, rtype, current_node_id,
                )
                return make_action(match_id, round_num, player_id, [
                    make_claim_resource_action(current_node_id, rtype)
                ])
            if total_permits < GUARD_RESERVE_FOR_GATE + 1:
                logger.info("Round %d: Claiming resource %s at %s (for window contests)", round_num, rtype, current_node_id)
                return make_action(match_id, round_num, player_id, [
                    make_claim_resource_action(current_node_id, rtype)
                ])
            continue
        if rtype == "BOAT_RIGHT" and my_resources.get("BOAT_RIGHT", 0) < 1:
            logger.info("Round %d: Claiming BOAT_RIGHT at %s", round_num, current_node_id)
            return make_action(match_id, round_num, player_id, [
                make_claim_resource_action(current_node_id, rtype)
            ])

    return None


def _handle_force_delivery_resource(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node_id: str,
    current_node: dict | None,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
) -> dict | None:
    """Claim route-shortening resources on the current node when the next hop is expensive."""
    if current_node is None or has_resource(player, "FAST_HORSE"):
        return None
    direct_target = _find_direct_delivery_step(
        graph, current_node_id, player, gate_node_id, terminal_node_ids,
        weather, process_nodes, processed_node_ids,
    )
    if not direct_target:
        return None
    hop_cost = graph.edge_cost(
        current_node_id, direct_target, weather, None, process_nodes,
    )
    if hop_cost < 40:
        return None
    for rtype, _count in find_available_resources(current_node):
        if rtype == "FAST_HORSE":
            logger.info(
                "Round %d: FORCE_DELIVERY claiming FAST_HORSE at %s (next hop cost=%.1f)",
                round_num, current_node_id, hop_cost,
            )
            return make_action(match_id, round_num, player_id, [
                make_claim_resource_action(current_node_id, rtype)
            ])
    return None


def _handle_use_resources(
    match_id: str, round_num: int, player_id: int,
    player: dict, current_node_id: str, graph: MapGraph,
    weather: dict | None, phase: str,
    gate_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
    map_gameplay: MapGameplayContext | None = None,
    inquire_nodes: list[dict] | None = None,
    visited_node_ids: set[str] | None = None,
) -> dict | None:
    """Handle using resources: horses, intel (冰鉴由 _try_use_ice_box 优先处理)."""
    intel_action = _try_use_intel(
        match_id, round_num, player_id, player, graph,
        current_node_id, weather, map_gameplay, visited_node_ids,
        inquire_nodes=inquire_nodes,
        my_team_id=get_team_id(player),
        gate_node_id=gate_node_id,
        process_nodes=process_nodes,
        processed_node_ids=processed_node_ids,
        gate_corridor_only=get_task_score(player) >= TASK_SCORE_TARGET,
    )
    if intel_action is not None:
        return intel_action

    force_delivery = _should_force_delivery(
        round_num, phase, player, graph, current_node_id,
        gate_node_id, terminal_node_ids, weather,
        process_nodes, processed_node_ids,
        map_gameplay=map_gameplay,
    )

    if force_delivery:
        direct_target = _find_direct_delivery_step(
            graph, current_node_id, player, gate_node_id,
            terminal_node_ids or [], weather, process_nodes,
            processed_node_ids or set(),
        )
        if direct_target:
            horse_action = _use_horse_before_expensive_hop(
                match_id, round_num, player_id, player, graph,
                current_node_id, direct_target, weather, process_nodes,
                force_delivery=True,
            )
            if horse_action is not None:
                return horse_action
    elif graph and gate_node_id and current_node_id:
        next_step = graph.next_step_toward(
            current_node_id, gate_node_id, weather, None,
            use_weighted=True, process_nodes=process_nodes,
        )
        if next_step:
            horse_action = _use_horse_before_expensive_hop(
                match_id, round_num, player_id, player, graph,
                current_node_id, next_step, weather, process_nodes,
                force_delivery=False,
            )
            if horse_action is not None:
                return horse_action

    return None


def _node_has_planned_t04(
    tasks: list[dict],
    node_id: str,
    failed_task_ids: set[str],
    my_task_score: int,
) -> bool:
    """Skip squad clear when we still plan to claim T04 at this node for task points."""
    if my_task_score >= TASK_SCORE_TARGET:
        return False
    for task in tasks:
        if task.get("nodeId") != node_id:
            continue
        if not task.get("active", False) or task.get("completed", False) or task.get("failed", False):
            continue
        if not get_task_template_id(task).startswith("T04"):
            continue
        task_id = task.get("taskId", "")
        if task_id and task_id in failed_task_ids:
            continue
        return True
    return False


def _path_nodes_to_goal(
    graph: MapGraph,
    start_node_id: str,
    goal_node_id: str,
    weather: dict | None,
    obstacle_nodes: set[str],
) -> list[str]:
    if not start_node_id or not goal_node_id:
        return []
    path = graph.shortest_path(start_node_id, goal_node_id, weather, obstacle_nodes)
    return path or []


def _get_planned_next_hop(
    graph: MapGraph,
    current_node_id: str,
    goal_node_id: str,
    weather: dict | None,
    process_nodes: dict[str, dict] | None = None,
) -> str | None:
    """Preferred weighted next hop toward goal (before obstacle avoidance reroute)."""
    if not current_node_id or not goal_node_id or current_node_id == goal_node_id:
        return None
    return graph.next_step_toward(
        current_node_id, goal_node_id, weather, None,
        use_weighted=True, process_nodes=process_nodes,
    )


def _node_is_route_obstacle(
    node_id: str,
    inquire_nodes: list[dict],
    obstacle_nodes: set[str],
) -> bool:
    if node_id in obstacle_nodes:
        return True
    node = _get_inquire_node(inquire_nodes, node_id)
    return node is not None and node_has_obstacle(node)


def _min_squad_for_clear_node(node_id: str, *, is_next_hop: bool) -> int:
    if node_id in GATE_CORRIDOR_NODES:
        return GATE_CORRIDOR_SQUAD_MIN
    if is_next_hop:
        return SQUAD_CLEAR_NEXT_HOP_MIN_SQUAD
    return SQUAD_CLEAR_MIN_SQUAD


def _can_squad_clear_obstacle(
    node_id: str,
    *,
    is_next_hop: bool,
    on_my_path: bool,
    round_num: int,
    force_delivery: bool,
    current_node_id: str,
    gate_node_id: str,
    graph: MapGraph | None,
    weather: dict | None,
    tasks: list[dict],
    failed_task_ids: set[str],
    my_task_score: int,
    squad_count: int,
) -> bool:
    if not _squad_clear_allowed(
        round_num, node_id, on_my_path, force_delivery,
        current_node_id, gate_node_id, graph, weather,
        is_next_hop=is_next_hop,
    ):
        return False
    if (
        node_id not in GATE_CORRIDOR_NODES
        and _node_has_planned_t04(tasks, node_id, failed_task_ids, my_task_score)
    ):
        return False
    return squad_count >= _min_squad_for_clear_node(node_id, is_next_hop=is_next_hop)


def _find_next_hop_obstacle_clear_target(
    graph: MapGraph,
    current_node_id: str,
    goal_node_id: str,
    inquire_nodes: list[dict],
    obstacle_nodes: set[str],
    weather: dict | None,
    tasks: list[dict],
    failed_task_ids: set[str],
    my_task_score: int,
    squad_count: int,
    squad_clear_pending: set[str],
    round_num: int = 0,
    force_delivery: bool = False,
    gate_node_id: str = "",
    process_nodes: dict[str, dict] | None = None,
) -> str | None:
    """Prioritize clearing an obstacle on the immediate planned next hop."""
    if squad_count < GATE_CORRIDOR_SQUAD_MIN:
        return None
    next_hop = _get_planned_next_hop(
        graph, current_node_id, goal_node_id, weather, process_nodes,
    )
    if not next_hop or next_hop == current_node_id or next_hop in squad_clear_pending:
        return None
    if not _node_is_route_obstacle(next_hop, inquire_nodes, obstacle_nodes):
        return None
    my_path = _path_nodes_to_goal(
        graph, current_node_id, goal_node_id, weather, obstacle_nodes,
    )
    on_my_path = next_hop in my_path
    if not _can_squad_clear_obstacle(
        next_hop,
        is_next_hop=True,
        on_my_path=on_my_path,
        round_num=round_num,
        force_delivery=force_delivery,
        current_node_id=current_node_id,
        gate_node_id=gate_node_id or goal_node_id,
        graph=graph,
        weather=weather,
        tasks=tasks,
        failed_task_ids=failed_task_ids,
        my_task_score=my_task_score,
        squad_count=squad_count,
    ):
        return None
    return next_hop


def _score_squad_clear_target(
    node_id: str,
    my_path: list[str],
    opp_path: list[str],
    obstacle_candidate_node_ids: frozenset[str] | None = None,
    next_hop_id: str = "",
) -> float:
    """Higher score = more valuable to clear on own route only."""
    on_my_path = node_id in my_path
    if not on_my_path:
        return 0.0
    if node_id in GATE_CORRIDOR_NODES:
        score = 25.0
    else:
        score = 12.0
    on_opp_path = bool(opp_path) and node_id in opp_path
    if on_my_path and on_opp_path:
        score += 4.0
    if obstacle_candidate_node_ids and node_id in obstacle_candidate_node_ids:
        score += 6.0
    if next_hop_id and node_id == next_hop_id:
        score += 50.0
    return score


def _find_squad_clear_target(
    graph: MapGraph,
    current_node_id: str,
    goal_node_id: str,
    inquire_nodes: list[dict],
    obstacle_nodes: set[str],
    weather: dict | None,
    opp_player: dict | None,
    tasks: list[dict],
    failed_task_ids: set[str],
    my_task_score: int,
    squad_count: int,
    squad_clear_pending: set[str],
    map_gameplay: MapGameplayContext | None = None,
    round_num: int = 0,
    force_delivery: bool = False,
    gate_node_id: str = "",
    process_nodes: dict[str, dict] | None = None,
) -> str | None:
    """Pick obstacle node for SQUAD_CLEAR on own route with unified timing gates."""
    if squad_count < GATE_CORRIDOR_SQUAD_MIN:
        return None

    next_hop_clear = _find_next_hop_obstacle_clear_target(
        graph, current_node_id, goal_node_id, inquire_nodes, obstacle_nodes,
        weather, tasks, failed_task_ids, my_task_score, squad_count,
        squad_clear_pending, round_num=round_num, force_delivery=force_delivery,
        gate_node_id=gate_node_id, process_nodes=process_nodes,
    )
    if next_hop_clear:
        logger.info(
            "Round %d: Priority next-hop squad clear at %s (squad=%d)",
            round_num, next_hop_clear, squad_count,
        )
        return next_hop_clear

    ctx = _map_ctx(map_gameplay)
    my_path = _path_nodes_to_goal(
        graph, current_node_id, goal_node_id, weather, obstacle_nodes,
    )
    planned_next_hop = _get_planned_next_hop(
        graph, current_node_id, goal_node_id, weather, process_nodes,
    )
    opp_path: list[str] = []
    if opp_player:
        opp_node = opp_player.get("currentNodeId", "")
        if opp_node:
            opp_path = _path_nodes_to_goal(
                graph, opp_node, goal_node_id, weather, obstacle_nodes,
            )

    best_node = ""
    best_score = float("-inf")
    for node in inquire_nodes:
        if not node.get("hasObstacle", False):
            continue
        nid = node.get("nodeId", "")
        if not nid or nid == current_node_id or nid in squad_clear_pending:
            continue
        on_my_path = nid in my_path
        is_next_hop = nid == planned_next_hop
        if not _squad_clear_allowed(
            round_num, nid, on_my_path, force_delivery,
            current_node_id, gate_node_id or goal_node_id, graph, weather,
            is_next_hop=is_next_hop,
        ):
            continue
        if not _can_squad_clear_obstacle(
            nid,
            is_next_hop=is_next_hop,
            on_my_path=on_my_path,
            round_num=round_num,
            force_delivery=force_delivery,
            current_node_id=current_node_id,
            gate_node_id=gate_node_id or goal_node_id,
            graph=graph,
            weather=weather,
            tasks=tasks,
            failed_task_ids=failed_task_ids,
            my_task_score=my_task_score,
            squad_count=squad_count,
        ):
            continue

        score = _score_squad_clear_target(
            nid, my_path, opp_path, ctx.obstacle_candidate_node_ids,
            next_hop_id=planned_next_hop or "",
        )
        if score <= 0:
            continue

        if (
            not is_next_hop
            and nid not in GATE_CORRIDOR_NODES
            and squad_count <= SQUAD_CLEAR_MIN_SQUAD + 1
            and score < 12.0
        ):
            continue
        if not is_next_hop and nid not in GATE_CORRIDOR_NODES and score < 9.0:
            continue

        if score > best_score:
            best_score = score
            best_node = nid

    return best_node or None


def _list_own_active_guard_nodes(
    inquire_nodes: list[dict], my_team_id: str, player_id: int,
) -> list[str]:
    nodes: list[str] = []
    for node in inquire_nodes:
        nid = node.get("nodeId", "")
        if nid and is_own_guard(node.get("guard"), my_team_id, player_id):
            nodes.append(nid)
    return nodes


def _count_own_active_guards(
    inquire_nodes: list[dict], my_team_id: str, player_id: int,
) -> int:
    return len(_list_own_active_guard_nodes(inquire_nodes, my_team_id, player_id))


def _own_guard_active_at(
    inquire_nodes: list[dict], node_id: str,
    my_team_id: str, player_id: int,
) -> bool:
    return is_own_guard(_get_node_guard(inquire_nodes, node_id), my_team_id, player_id)


def _get_inquire_node_type(node: dict | None) -> str:
    if not node:
        return ""
    return str(node.get("nodeType") or node.get("type") or "")


def _is_key_pass_node(inquire_nodes: list[dict], node_id: str) -> bool:
    return _get_inquire_node_type(_get_inquire_node(inquire_nodes, node_id)) == "KEY_PASS"


def _is_pass_transfer_node(process_nodes: dict[str, dict] | None, node_id: str) -> bool:
    if not process_nodes or not node_id:
        return False
    return process_nodes.get(node_id, {}).get("processType") == "PASS_TRANSFER"


def _is_palace_transfer_node(process_nodes: dict[str, dict] | None, node_id: str) -> bool:
    if not process_nodes or not node_id:
        return False
    return process_nodes.get(node_id, {}).get("processType") == "PALACE_TRANSFER"


def _hops_to_gate(
    graph: MapGraph,
    node_id: str,
    gate_node_id: str,
    weather: dict | None,
    obstacle_nodes: set[str] | None,
) -> float:
    if not graph or not node_id or not gate_node_id:
        return float("inf")
    return graph.path_length(node_id, gate_node_id, weather, obstacle_nodes)


def _is_in_final_corridor(
    graph: MapGraph,
    node_id: str,
    gate_node_id: str,
    weather: dict | None,
    obstacle_nodes: set[str] | None,
) -> bool:
    hops = _hops_to_gate(graph, node_id, gate_node_id, weather, obstacle_nodes)
    return hops != float("inf") and hops <= FINAL_CORRIDOR_GATE_HOPS


def _is_final_corridor_guard_site(
    graph: MapGraph,
    node_id: str,
    gate_node_id: str,
    inquire_nodes: list[dict],
    process_nodes: dict[str, dict] | None,
    weather: dict | None,
    obstacle_nodes: set[str] | None,
) -> bool:
    if not _is_in_final_corridor(graph, node_id, gate_node_id, weather, obstacle_nodes):
        return False
    if _is_key_pass_node(inquire_nodes, node_id):
        return True
    if _is_pass_transfer_node(process_nodes, node_id):
        return True
    node_type = _get_inquire_node_type(_get_inquire_node(inquire_nodes, node_id))
    return node_type in ("PASS", "KEY_PASS")


def _final_corridor_guard_priority(
    graph: MapGraph,
    node_id: str,
    gate_node_id: str,
    inquire_nodes: list[dict],
    process_nodes: dict[str, dict] | None,
    weather: dict | None,
    obstacle_nodes: set[str] | None,
) -> int:
    score = 0
    if _is_in_final_corridor(graph, node_id, gate_node_id, weather, obstacle_nodes):
        score += 3
    if _is_key_pass_node(inquire_nodes, node_id):
        score += 10
    if _is_pass_transfer_node(process_nodes, node_id):
        score += 6
    if gate_node_id and node_id == gate_node_id:
        score += 8
    return score


def _is_key_choke_node(graph: MapGraph, node_id: str) -> bool:
    return len(graph.get_neighbors(node_id)) <= 4


def _guard_extra_good_fruit(
    player: dict,
    inquire_nodes: list[dict] | None = None,
    current_node_id: str = "",
    task_score: int = 0,
) -> int:
    good = get_good_fruit(player)
    reserve = 1 + GUARD_GOOD_FRUIT_RESERVE
    if (
        inquire_nodes
        and current_node_id
        and _is_key_pass_node(inquire_nodes, current_node_id)
        and task_score >= FINAL_CORRIDOR_GUARD_TASK_MIN
        and good >= 1 + 2 + reserve
    ):
        return 2
    if good >= 1 + 1 + reserve:
        return 1
    return 0


def _guard_good_fruit_sufficient(
    player: dict,
    inquire_nodes: list[dict] | None = None,
    current_node_id: str = "",
    task_score: int = 0,
) -> bool:
    extra = _guard_extra_good_fruit(player, inquire_nodes, current_node_id, task_score)
    return get_good_fruit(player) >= 1 + extra + GUARD_GOOD_FRUIT_RESERVE


def _evaluate_dual_guard_slot(
    graph: MapGraph,
    current_node_id: str,
    goal: str,
    weather: dict | None,
    obstacle_nodes: set[str],
    player: dict,
    inquire_nodes: list[dict],
    my_team_id: str,
    player_id: int,
    opp_player: dict,
    process_nodes: dict[str, dict] | None = None,
) -> tuple[bool, str]:
    """Dual-guard plan: guard1 then guard2, each on opp path at a choke."""
    own_guards = _list_own_active_guard_nodes(inquire_nodes, my_team_id, player_id)
    if len(own_guards) >= MAX_ACTIVE_GUARDS:
        return False, "max-guards"
    if _own_guard_active_at(inquire_nodes, current_node_id, my_team_id, player_id):
        return False, "already-guarded"
    task_score = get_task_score(player)
    if not _guard_good_fruit_sufficient(
        player, inquire_nodes, current_node_id, task_score,
    ):
        return False, "low-fruit"
    if not _is_key_choke_node(graph, current_node_id):
        return False, "not-choke"

    opp_node = opp_player.get("currentNodeId", "")
    if not opp_node:
        return False, "no-opp-node"

    my_hops = graph.path_length(current_node_id, goal, weather, obstacle_nodes)
    opp_hops = graph.path_length(opp_node, goal, weather, obstacle_nodes)
    if my_hops == float("inf") or opp_hops == float("inf"):
        return False, "unreachable"

    opp_path = _path_nodes_to_goal(graph, opp_node, goal, weather, obstacle_nodes)
    if current_node_id not in opp_path:
        return False, "not-on-opp-path"

    lead = opp_hops - my_hops
    if lead < 1:
        return False, "not-leading"

    final_guard_site = _is_final_corridor_guard_site(
        graph, current_node_id, goal, inquire_nodes, process_nodes,
        weather, obstacle_nodes,
    )
    min_lead_first = GUARD_MIN_LEAD_FIRST
    if final_guard_site and task_score >= FINAL_CORRIDOR_GUARD_TASK_MIN:
        min_lead_first = FINAL_CORRIDOR_GUARD_MIN_LEAD

    if len(own_guards) == 0:
        if lead < min_lead_first:
            return False, "guard1-insufficient-lead"
        if final_guard_site and _is_key_pass_node(inquire_nodes, current_node_id):
            return True, f"final-guard1,key-pass,ahead({int(lead)})"
        return True, f"guard1,ahead({int(lead)}),opp-path,choke"

    first_node = own_guards[0]
    if current_node_id == first_node:
        return False, "guard2-same-node"
    first_guard = _get_node_guard(inquire_nodes, first_node)
    if not is_own_guard(first_guard, my_team_id, player_id):
        return False, "guard1-lost"
    if first_guard.get("defense", 0) <= 0:
        return False, "guard1-not-active"

    if (
        final_guard_site
        and _is_pass_transfer_node(process_nodes, current_node_id)
        and _is_key_pass_node(inquire_nodes, first_node)
    ):
        return True, f"final-guard2,pass-transfer,after-{first_node}"

    if final_guard_site and task_score >= FINAL_CORRIDOR_GUARD_TASK_MIN:
        return True, f"final-guard2,after-{first_node},opp-path,choke"

    return True, f"guard2,after-{first_node},opp-path,choke"


def _should_set_guard_dynamic(
    graph: MapGraph,
    current_node_id: str,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    obstacle_nodes: set[str],
    player: dict,
    inquire_nodes: list[dict],
    my_team_id: str,
    player_id: int,
    opp_player: dict | None,
    mode: str,
    phase: str,
    process_nodes: dict[str, dict] | None = None,
    force_delivery: bool = False,
    map_gameplay: MapGameplayContext | None = None,
) -> tuple[bool, str]:
    """Decide whether to SET_GUARD at the fleet's current node."""
    goal = gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "")
    if not goal or not current_node_id:
        return False, ""

    if force_delivery:
        return False, "force-delivery"

    task_score = get_task_score(player)
    guard_min = _profile(map_gameplay).guard_min_task_score
    final_guard_site = _is_final_corridor_guard_site(
        graph, current_node_id, gate_node_id, inquire_nodes, process_nodes,
        weather, obstacle_nodes,
    )
    final_guard_exception = (
        final_guard_site
        and task_score >= FINAL_CORRIDOR_GUARD_TASK_MIN
    )
    if task_score < guard_min and mode != "GATE_FIGHT" and not final_guard_exception:
        return False, f"task<{guard_min}"
    hops_to_gate = float("inf")
    if graph and gate_node_id:
        hops_to_gate = graph.path_length(current_node_id, gate_node_id, weather, obstacle_nodes)

    if phase == "RUSH":
        if is_verified(player):
            return False, "rush-verified"
        if gate_node_id and current_node_id == gate_node_id:
            return False, "rush-priority-verify-deliver"

    if mode == "GATE_FIGHT":
        if task_score >= 60 and hops_to_gate <= 4:
            return False, "gate-fight-near-end"
        if _count_own_active_guards(inquire_nodes, my_team_id, player_id) >= MAX_ACTIVE_GUARDS:
            return False, "max-guards"
        if _own_guard_active_at(inquire_nodes, current_node_id, my_team_id, player_id):
            return False, "already-guarded"
        if not _guard_good_fruit_sufficient(
            player, inquire_nodes, current_node_id, task_score,
        ):
            return False, "low-fruit"
        if final_guard_site:
            return True, "gate-fight-final-choke"
        if _is_key_choke_node(graph, current_node_id):
            return True, "gate-fight-choke"
        if gate_node_id and current_node_id == gate_node_id:
            return True, "gate-fight-gate"
        return False, "gate-fight-skip"

    if not opp_player:
        return False, "no-opp"

    return _evaluate_dual_guard_slot(
        graph, current_node_id, goal, weather, obstacle_nodes,
        player, inquire_nodes, my_team_id, player_id, opp_player,
        process_nodes=process_nodes,
    )


def _try_leave_own_guard_node(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node_id: str,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    route_blocked: set[str],
    avoid_route_nodes: set[str],
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
    own_guard_sites: set[str],
) -> dict | None:
    """After SET_GUARD, leave the node via detour so we do not block ourselves."""
    if not current_node_id or current_node_id not in own_guard_sites:
        return None

    detour = _find_delivery_detour_step(
        graph, current_node_id, player, gate_node_id, terminal_node_ids,
        weather, process_nodes, processed_node_ids, route_blocked,
        avoid_nodes=avoid_route_nodes,
    )
    if detour:
        logger.info(
            "Round %d: leaving own guard at %s via %s (guard/advance split)",
            round_num, current_node_id, detour,
        )
        return make_action(match_id, round_num, player_id, [make_move_action(detour)])

    for neighbor in graph.get_neighbors(current_node_id):
        if neighbor in avoid_route_nodes:
            continue
        logger.info(
            "Round %d: leaving own guard at %s via %s (fallback)",
            round_num, current_node_id, neighbor,
        )
        return make_action(match_id, round_num, player_id, [make_move_action(neighbor)])
    return None


def _try_set_guard_action(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph,
    current_node_id: str, gate_node_id: str, terminal_node_ids: list[str],
    weather: dict | None, obstacle_nodes: set[str],
    inquire_nodes: list[dict], my_team_id: str,
    opp_player: dict | None, mode: str, phase: str,
    process_nodes: dict[str, dict] | None = None,
    force_delivery: bool = False,
    map_gameplay: MapGameplayContext | None = None,
) -> dict | None:
    should, reason = _should_set_guard_dynamic(
        graph, current_node_id, gate_node_id, terminal_node_ids,
        weather, obstacle_nodes, player, inquire_nodes,
        my_team_id, player_id, opp_player, mode, phase,
        process_nodes=process_nodes,
        force_delivery=force_delivery,
        map_gameplay=map_gameplay,
    )
    if not should:
        return None
    task_score = get_task_score(player)
    extra = _guard_extra_good_fruit(
        player, inquire_nodes, current_node_id, task_score,
    )
    logger.info("Round %d: Setting guard at %s (%s, extra=%d)", round_num, current_node_id, reason, extra)
    return make_action(match_id, round_num, player_id, [
        make_set_guard_action(current_node_id, extra_good_fruit=extra)
    ])


def _handle_combat(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph,
    current_node_id: str, gate_node_id: str, terminal_node_ids: list[str],
    weather: dict | None, blocked: set[str] | None,
    mode: str, phase: str, inquire_nodes: list[dict],
    opp_player: dict | None,
    obstacle_nodes: set[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
    visited_node_ids: set[str] | None = None,
    my_team_id: str = "",
    tasks: list[dict] | None = None,
    failed_task_ids: set[str] | None = None,
    squad_clear_pending: set[str] | None = None,
    map_gameplay: MapGameplayContext | None = None,
    processed_node_ids: set[str] | None = None,
) -> dict | None:
    """Handle combat: guard, break, squad (策略文档 §8)."""
    if obstacle_nodes is None:
        obstacle_nodes = set()
    if process_nodes is None:
        process_nodes = {}
    if visited_node_ids is None:
        visited_node_ids = set()
    if tasks is None:
        tasks = []
    if failed_task_ids is None:
        failed_task_ids = set()
    if squad_clear_pending is None:
        squad_clear_pending = set()
    if processed_node_ids is None:
        processed_node_ids = set()
    if not my_team_id:
        my_team_id = get_team_id(player)

    # --- BREAK_GUARD with optional BREAK_ORDER (策略文档 §8.2, §10) ---
    goal = gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "")
    if goal and blocked:
        optimal_path = graph.shortest_path(current_node_id, goal, weather, obstacle_nodes)
        if optimal_path and len(optimal_path) >= 2:
            next_hop = optimal_path[1]
            if next_hop in blocked:
                reroute_blocked = set(blocked) | set(obstacle_nodes)
                reroute_blocked.discard(goal)
                alt_step = graph.next_step_toward(
                    current_node_id, goal, weather, reroute_blocked,
                    use_weighted=True, process_nodes=process_nodes,
                )
                if alt_step and alt_step not in blocked:
                    logger.info(
                        "Round %d: Reroute via %s to avoid blocked %s",
                        round_num, alt_step, next_hop,
                    )
                    return make_action(match_id, round_num, player_id, [make_move_action(alt_step)])
                for node in inquire_nodes:
                    if node.get("nodeId") != next_hop:
                        continue
                    guard = node.get("guard", {})
                    if is_enemy_guard(guard, my_team_id, player_id):
                        good, bad = _break_guard_investment(player)
                        if good + bad > 0:
                            action = make_break_guard_action(next_hop, good_fruit=good, bad_fruit=bad)
                            rush_used = int(player.get("rushTacticUsedCount", 0) or 0)
                            if phase == "RUSH" and rush_used == 0 and (bad >= 2 or get_good_fruit(player) >= 2):
                                action["rushTactic"] = "BREAK_ORDER"
                                logger.info("Round %d: Breaking guard at %s with BREAK_ORDER", round_num, next_hop)
                            else:
                                logger.info("Round %d: Breaking guard at %s (blocking path)", round_num, next_hop)
                            return make_action(match_id, round_num, player_id, [action])
                    logger.info("Round %d: Forced pass at %s", round_num, next_hop)
                    return make_action(match_id, round_num, player_id, [
                        make_forced_pass_action(next_hop)
                    ])

    # --- Squad actions (策略文档 §8.4) — only if not RUSH ---
    if phase != "RUSH":
        squad_count = get_squad_count(player)
        my_task_score = get_task_score(player)
        scout_min = _profile(map_gameplay).squad_scout_min_squad

        # SQUAD_CLEAR before scout: prioritize next-hop obstacles on own route.
        if squad_count >= GATE_CORRIDOR_SQUAD_MIN and goal:
            clear_target = _find_squad_clear_target(
                graph, current_node_id, goal, inquire_nodes, obstacle_nodes,
                weather, opp_player, tasks, failed_task_ids, my_task_score,
                squad_count, squad_clear_pending, map_gameplay,
                round_num=round_num, force_delivery=False,
                gate_node_id=gate_node_id or goal,
                process_nodes=process_nodes,
            )
            if clear_target:
                logger.info(
                    "Round %d: Squad clear at %s (own-route, squad=%d)",
                    round_num, clear_target, squad_count,
                )
                return make_action(match_id, round_num, player_id, [
                    make_squad_clear_action(clear_target)
                ])

        scout_action = _try_squad_scout(
            match_id, round_num, player_id, graph,
            current_node_id, gate_node_id or goal, weather,
            process_nodes, processed_node_ids, visited_node_ids,
            inquire_nodes, my_team_id, squad_count, scout_min, my_task_score,
        )
        if scout_action is not None:
            return scout_action

        # SQUAD_REINFORCE: Reinforce our own guard at final corridor key nodes first
        if squad_count >= SQUAD_CLEAR_MIN_SQUAD:
            best_reinforce = ""
            best_reinforce_score = -1
            for node in inquire_nodes:
                guard = node.get("guard", {})
                owner_team = guard.get("ownerTeamId") if guard else ""
                nid = node.get("nodeId", "")
                if not nid or nid == current_node_id:
                    continue
                if not (guard and owner_team == my_team_id and guard_is_active(guard)):
                    continue
                score = _final_corridor_guard_priority(
                    graph, nid, goal, inquire_nodes, process_nodes,
                    weather, obstacle_nodes,
                )
                if score > best_reinforce_score:
                    best_reinforce_score = score
                    best_reinforce = nid
            if best_reinforce and best_reinforce_score > 0:
                logger.info(
                    "Round %d: Squad reinforce at %s (final-corridor priority=%d)",
                    round_num, best_reinforce, best_reinforce_score,
                )
                return make_action(match_id, round_num, player_id, [
                    make_squad_reinforce_action(best_reinforce)
                ])
            for node in inquire_nodes:
                guard = node.get("guard", {})
                owner_team = guard.get("ownerTeamId") if guard else ""
                if (guard and owner_team == my_team_id
                        and guard_is_active(guard)
                        and node.get("nodeId") != current_node_id):
                    nid = node.get("nodeId", "")
                    logger.info("Round %d: Squad reinforce at %s", round_num, nid)
                    return make_action(match_id, round_num, player_id, [
                        make_squad_reinforce_action(nid)
                    ])

        # SQUAD_WEAKEN: Prefer enemy guards on the final corridor (KEY_PASS / PASS_TRANSFER)
        if squad_count >= 2 and opp_player:
            best_weaken = ""
            best_weaken_score = -1
            for node in inquire_nodes:
                guard = node.get("guard", {})
                nid = node.get("nodeId", "")
                if not nid or nid == current_node_id:
                    continue
                if not is_enemy_guard(guard, my_team_id, player_id):
                    continue
                score = _final_corridor_guard_priority(
                    graph, nid, goal, inquire_nodes, process_nodes,
                    weather, obstacle_nodes,
                )
                if score > best_weaken_score:
                    best_weaken_score = score
                    best_weaken = nid
            if best_weaken and best_weaken_score > 0:
                logger.info(
                    "Round %d: Squad weaken at %s (final-corridor priority=%d)",
                    round_num, best_weaken, best_weaken_score,
                )
                return make_action(match_id, round_num, player_id, [
                    make_squad_weaken_action(best_weaken)
                ])
            for node in inquire_nodes:
                guard = node.get("guard", {})
                if (is_enemy_guard(guard, my_team_id, player_id)
                        and node.get("nodeId") != current_node_id):
                    nid = node.get("nodeId", "")
                    logger.info("Round %d: Squad weaken at %s", round_num, nid)
                    return make_action(match_id, round_num, player_id, [
                        make_squad_weaken_action(nid)
                    ])

    return None


def _handle_rush_tactics(
    match_id: str, round_num: int, player_id: int,
    player: dict, current_node_id: str, phase: str, mode: str,
    graph: MapGraph | None = None,
    gate_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    weather: dict | None = None,
    obstacle_nodes: set[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    rush_speed_failed: bool = False,
    map_gameplay: MapGameplayContext | None = None,
    force_delivery: bool = False,
    delivery_critical: bool = False,
) -> dict | None:
    """Handle rush tactics: RUSH_SPEED, RUSH_PROTECT (策略文档 §10).

    Only available after RUSH phase. Each can be used once per match.
    RUSH_SPEED与马互斥: 有马buff时不使用.
    Reserve rush quota for gate VERIFY / BREAK_ORDER during delivery push.
    """
    if phase != "RUSH":
        return None

    if force_delivery or delivery_critical:
        return None

    if int(player.get("rushTacticUsedCount", 0) or 0) >= 1:
        return None

    state = player.get("state", "")
    if state != "IDLE":
        return None

    if graph and gate_node_id and current_node_id:
        hops = graph.path_length(current_node_id, gate_node_id, weather, obstacle_nodes)
        if hops != float("inf") and hops <= 3:
            return None

    freshness = get_freshness(player)

    if freshness < RUSH_PROTECT_FRESHNESS:
        logger.info("Round %d: Using RUSH_PROTECT (freshness=%.1f)", round_num, freshness)
        return make_action(match_id, round_num, player_id, [make_rush_protect_action()])

    if rush_speed_failed or _has_move_speed_buff(player):
        return None
    if get_good_fruit(player) < 3:
        return None

    if graph and gate_node_id and current_node_id:
        next_step = graph.next_step_toward(
            current_node_id, gate_node_id, weather, obstacle_nodes,
            use_weighted=True, process_nodes=process_nodes,
        )
        if next_step:
            hop_cost = graph.edge_cost(
                current_node_id, next_step, weather, obstacle_nodes, process_nodes,
            )
            if hop_cost >= HORSE_USE_MIN_HOP_COST:
                logger.info(
                    "Round %d: Using RUSH_SPEED before expensive hop (cost=%.1f)",
                    round_num, hop_cost,
                )
                return make_action(match_id, round_num, player_id, [make_rush_speed_action()])

    if freshness >= _ice_threshold(map_gameplay):
        logger.info("Round %d: Using RUSH_SPEED (freshness=%.1f)", round_num, freshness)
        return make_action(match_id, round_num, player_id, [make_rush_speed_action()])

    return None
