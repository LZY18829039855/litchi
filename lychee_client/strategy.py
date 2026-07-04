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
from typing import Any

from lychee_client.map_graph import MapGraph, ROUTE_FRESHNESS_LOSS
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
    TASK_SCORE_TARGET, TASK_SCORE_STRETCH, MAX_TASK_DETOUR_COST,
    ROUTE_TASK_BONUS_PER_SCORE, ROUTE_TASK_COUNT_BONUS, ROUTE_HIGH_VALUE_TASK_BONUS,
    ROUTE_VISITED_BACKTRACK_PENALTY,
    ROUTE_BUCKET_BONUS_PER_SCORE, NEAR_GATE_RESOURCE_HOPS,
    HORSE_USE_MIN_HOP_COST, HORSE_USE_MIN_HOP_COST_EARLY,
    ICE_BOX_FRESHNESS_THRESHOLD, RUSH_PROTECT_FRESHNESS,
    RESOURCE_CLAIM_PRIORITY, TASK_PRIORITY, MAX_ROUND,
    SQUAD_CLEAR_COST, SQUAD_RESERVE_FOR_LATE, SQUAD_CLEAR_MIN_SQUAD,
    MAX_ACTIVE_GUARDS, GUARD_GOOD_FRUIT_RESERVE, GUARD_MIN_LEAD_FIRST,
    GUARD_STUCK_AVOID_ROUNDS, GUARD_SILENT_WAIT_LIMIT,
    GUARD_RESERVE_FOR_GATE, FINAL_CORRIDOR_GATE_HOPS,
    FINAL_CORRIDOR_GUARD_MIN_LEAD, FINAL_CORRIDOR_GUARD_TASK_MIN,
    FORCE_DELIVERY_ETA_BUFFER, FORCE_DELIVERY_MIN_REMAINING,
    FORCE_DELIVERY_ETA_REMAINING_MAX,
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
    make_rush_protect_action,
)

logger = logging.getLogger("lychee_client.strategy")


def _make_process_action(
    match_id: str,
    round_num: int,
    player_id: int,
    process_type: str,
    current_node_id: str,
    phase: str,
) -> dict:
    """Map processType to the correct protocol action."""
    if process_type == "DOCK":
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
        return _decide_action_impl(
            match_id, round_num, player_id, player, graph,
            current_node, process_nodes, contests, events,
            active_contest_id, last_move_failed, last_move_error,
            gate_node_id, terminal_node_ids, tasks, phase,
            processed_node_ids, visited_node_ids, weather, all_players, inquire_nodes,
            failed_task_ids, rush_speed_failed, guard_blocked_targets, avoid_route_nodes,
            pending_task_hold_task_id, pending_task_hold_node_id, pending_task_hold_until_round,
            forced_pass_failed_targets, squad_clear_pending,
            guard_stuck_rounds, guard_stuck_target, own_guard_sites,
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
    force_delivery = _should_force_delivery(
        round_num, phase, player, graph, current_node_id,
        gate_node_id, terminal_node_ids, weather,
        process_nodes, processed_node_ids,
    )
    guard_wait_kwargs = {
        "force_delivery": force_delivery,
        "graph": graph,
        "gate_node_id": gate_node_id,
        "terminal_node_ids": terminal_node_ids,
        "weather": weather,
        "route_blocked": route_blocked,
        "avoid_route_nodes": avoid_route_nodes,
        "guard_stuck_rounds": guard_stuck_rounds,
        "guard_stuck_target": guard_stuck_target,
        "process_nodes": process_nodes,
        "processed_node_ids": processed_node_ids,
    }

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
                    )
                    if task_retry is not None:
                        return task_retry
                task_retry = _retry_task_at_current_node(
                    match_id, round_num, player_id, player, graph,
                    current_node_id, tasks, failed_task_ids,
                    enemy_busy_task_ids=enemy_busy_task_ids,
                )
                if task_retry is not None:
                    return task_retry

            if force_delivery and current_node_id and not next_node:
                move_action = _plan_force_delivery_move(
                    match_id, round_num, player_id, player, graph,
                    current_node_id, gate_node_id, terminal_node_ids,
                    weather, process_nodes, processed_node_ids,
                    route_blocked, avoid_route_nodes, guard_stuck_rounds, guard_stuck_target,
                    inquire_nodes, tasks, failed_task_ids, obstacle_nodes, my_team_id,
                    forced_pass_failed_targets, last_move_failed, last_move_error,
                    log_prefix="FORCE_DELIVERY",
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
                if next_node in route_blocked:
                    return _handle_limited_state_guard_block(
                        match_id, round_num, player_id, player, state,
                        next_node, last_move_failed, last_move_error,
                        inquire_nodes, my_team_id, guard_wait_kwargs,
                    )
                return make_action(match_id, round_num, player_id, [make_move_action(next_node)])
            if current_node_id:
                move_target = _find_move_target(
                    graph, current_node_id, player, gate_node_id, terminal_node_ids,
                    weather, route_blocked, obstacle_nodes=obstacle_nodes,
                    process_nodes=process_nodes,
                    processed_node_ids=processed_node_ids, visited_node_ids=visited_node_ids,
                    tasks=tasks, player_id=player_id, failed_task_ids=failed_task_ids,
                    enemy_busy_task_ids=enemy_busy_task_ids, phase=phase,
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
            if guard_target or last_move_failed and last_move_error == "MOVE_BLOCKED_BY_GUARD":
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

    # At S14 (gate): VERIFY_GATE in RUSH phase
    if gate_node_id and is_at_node(player, gate_node_id) and not is_verified(player):
        if phase == "RUSH":
            action = make_verify_gate_action(current_node_id)
            for node in inquire_nodes:
                if node.get("nodeId") == gate_node_id:
                    guard = node.get("guard")
                    if is_enemy_guard(guard, my_team_id, player_id):
                        if get_bad_fruit(player) >= 2 or get_good_fruit(player) >= 1:
                            action["rushTactic"] = "BREAK_ORDER"
                    break
            return make_action(match_id, round_num, player_id, [action])
        # Not RUSH yet: don't submit VERIFY_GATE (will be rejected)
        # Continue doing other things until RUSH

    # --- Fixed processing (策略文档 §4.1: 再次到达同一站需重新处理) ---
    # Process at current node ONLY if not already processed this visit.
    # processed_node_ids tracks nodes where we completed processing this session.
    # If already processed, skip to MOVE (even if node has processType).
    already_processed_here = current_node_id in processed_node_ids
    process_type = None if already_processed_here else _get_process_type(current_node, process_nodes, current_node_id)

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

    # --- Dual guard (before optional tasks): set at choke on opp path, then advance separately ---
    if not force_delivery:
        guard_action = _try_set_guard_action(
            match_id, round_num, player_id, player, graph,
            current_node_id, gate_node_id, terminal_node_ids,
            weather, obstacle_nodes, inquire_nodes, my_team_id,
            opp_player, mode, phase, process_nodes=process_nodes,
            force_delivery=force_delivery,
        )
        if guard_action is not None:
            return guard_action

    # --- P2/P3: Task strategy (策略文档 §5) — keep collecting until task_score ≥ 90 ---
    if not force_delivery or get_task_score(player) < TASK_SCORE_TARGET:
        task_action = _handle_tasks(
            match_id, round_num, player_id, player, graph,
            current_node_id, tasks, player_id, phase, weather, blocked,
            goal_node_id=gate_node_id, terminal_node_ids=terminal_node_ids,
            obstacle_nodes=obstacle_nodes, process_nodes=process_nodes,
            processed_node_ids=processed_node_ids, visited_node_ids=visited_node_ids,
            failed_task_ids=failed_task_ids,
            enemy_busy_task_ids=enemy_busy_task_ids,
        )
        if task_action is not None:
            return task_action

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
        )
        if combat_action is not None:
            return combat_action

    # --- Rush tactics (策略文档 §10) ---
    rush_action = _handle_rush_tactics(
        match_id, round_num, player_id, player,
        current_node_id, phase, mode,
        graph=graph, gate_node_id=gate_node_id,
        terminal_node_ids=terminal_node_ids, weather=weather,
        obstacle_nodes=obstacle_nodes, process_nodes=process_nodes,
        processed_node_ids=processed_node_ids, visited_node_ids=visited_node_ids,
        rush_speed_failed=rush_speed_failed,
    )
    if rush_action is not None:
        return rush_action

    # --- NAVIGATION: Move toward goal ---
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
        force_delivery=force_delivery,
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


def _is_paying_guard_travel_tax(
    state: str,
    player: dict,
    last_move_error: str,
    block_node: str,
    route_blocked: set[str],
) -> bool:
    """WAITING/MOVING 交设卡时间税时只能 WAIT/EMPTY，不能 BREAK/CLEAR。"""
    if last_move_error == "MOVING_ACTION_FORBIDDEN":
        return True
    if state != "WAITING":
        return False
    next_node = player.get("nextNodeId", "")
    if next_node and next_node in route_blocked:
        return True
    return bool(block_node and block_node in route_blocked)


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
    if _is_paying_guard_travel_tax(state, player, last_move_error, block_node, route_blocked):
        logger.info(
            "Round %d: %s paying guard tax at %s, WAIT only (no BREAK/CLEAR)",
            round_num, state, block_node,
        )
        return make_action(match_id, round_num, player_id, [make_wait_action()])
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
) -> float:
    """Estimate frames needed to reach delivery-ready state along the planned route."""
    if not current_node_id:
        return float("inf")

    remaining_process = process_nodes
    if process_nodes and processed_node_ids is not None:
        remaining_process = {
            nid: info for nid, info in process_nodes.items()
            if nid not in processed_node_ids
        }

    goal = _get_goal_node(
        player, gate_node_id, terminal_node_ids, graph,
        current_node_id, weather, None, remaining_process,
    )
    if not goal:
        return 0.0

    path = graph.weighted_shortest_path(
        current_node_id, goal, weather, None, remaining_process,
    )
    if not path and remaining_process is not process_nodes:
        path = graph.weighted_shortest_path(
            current_node_id, goal, weather, None, process_nodes,
        )
    if not path:
        return float("inf")

    total = sum(
        graph.edge_cost(path[i], path[i + 1], weather, None, remaining_process)
        for i in range(len(path) - 1)
    )
    if not is_verified(player) and gate_node_id:
        total += 18
    elif is_verified(player):
        total += 4
    return total


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
) -> bool:
    """Stop optional scoring when remaining time is tight for delivery ETA."""
    if phase == "RUSH":
        return True

    task_score = get_task_score(player)
    if task_score >= TASK_SCORE_TARGET:
        return True

    remaining = max(0, max_round - round_num)
    if remaining <= 0:
        return True

    if remaining <= int(max_round * 0.33):
        return True

    if graph is None or not current_node_id:
        return False

    if task_score < 60:
        return False

    eta = _estimate_delivery_eta(
        graph, current_node_id, player, gate_node_id,
        terminal_node_ids or [], weather, process_nodes, processed_node_ids,
    )
    buffer = FORCE_DELIVERY_ETA_BUFFER

    if remaining <= FORCE_DELIVERY_ETA_REMAINING_MAX and eta >= remaining - buffer:
        return True
    return False


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
    direct_target = _find_direct_delivery_step(
        graph, current_node_id, player, gate_node_id, terminal_node_ids,
        weather, process_nodes, processed_node_ids,
        avoid_nodes=avoid_route_nodes,
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
        good = min(get_good_fruit(player), 2)
        bad = min(get_bad_fruit(player), 2)
        if good + bad > 0:
            action = make_break_guard_action(target_node_id, good_fruit=good, bad_fruit=bad)
            logger.info("Round %d: FORCE_DELIVERY break guard at %s", round_num, target_node_id)
            return make_action(match_id, round_num, player_id, [action])
        logger.info("Round %d: FORCE_DELIVERY forced pass guard at %s", round_num, target_node_id)
        return make_action(match_id, round_num, player_id, [make_forced_pass_action(target_node_id)])

    if target_node_id in obstacle_nodes:
        if not _confirmed_obstacle(inquire_nodes, target_node_id, obstacle_nodes):
            logger.info(
                "Round %d: FORCE_DELIVERY skip CLEAR at %s (no confirmed obstacle)",
                round_num, target_node_id,
            )
            return make_action(match_id, round_num, player_id, [make_forced_pass_action(target_node_id)])
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

    logger.info("Round %d: FORCE_DELIVERY forced pass blocked %s", round_num, target_node_id)
    return make_action(match_id, round_num, player_id, [make_forced_pass_action(target_node_id)])


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
    guarded = (
        target_node_id in route_blocked
        or is_enemy_guard(guard, my_team_id, player_id)
    )
    if not guarded:
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


def _should_use_task_aware_routing(player: dict, phase: str, force_delivery: bool) -> bool:
    if force_delivery:
        return False
    my_task_score = get_task_score(player)
    if my_task_score >= TASK_SCORE_TARGET:
        return False
    if phase == "RUSH" and my_task_score >= 60:
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


def _route_bucket_task_scores(
    tasks: list[dict],
    player_id: int,
    player: dict,
    failed_task_ids: set[str] | None,
    enemy_busy_task_ids: set[str] | None,
    obstacle_nodes: set[str] | None,
) -> dict[str, int]:
    scores: dict[str, int] = {}
    for task in tasks:
        if not is_task_available(task, player_id, failed_task_ids, enemy_busy_task_ids):
            continue
        if not _task_routable(task, player, obstacle_nodes):
            continue
        bucket = task.get("routeBucket") or "ROAD"
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
) -> tuple[int, int, int]:
    """Return (task_score_sum, on_path_task_count, high_value_task_count) for a route via neighbor."""
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
            if detour > MAX_TASK_DETOUR_COST:
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
) -> float:
    hop_cost = graph.edge_cost(current_node_id, neighbor, weather, blocked, process_nodes)
    tail_cost = _weighted_path_cost(graph, neighbor, goal_node, weather, blocked, process_nodes)
    if tail_cost == float("inf"):
        return float("inf")

    score = hop_cost + tail_cost
    if neighbor in visited_node_ids:
        score += ROUTE_VISITED_BACKTRACK_PENALTY

    task_score_sum, task_count, high_count = _route_task_stats(
        graph, current_node_id, neighbor, goal_node, tasks, player_id, player,
        gate_node_id, terminal_node_ids, weather, blocked, process_nodes,
        failed_task_ids, enemy_busy_task_ids, obstacle_nodes,
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
) -> str | None:
    if not available or not goal_node:
        return None

    bucket_scores = _route_bucket_task_scores(
        tasks, player_id, player, failed_task_ids, enemy_busy_task_ids, obstacle_nodes,
    )
    best_neighbor = None
    best_score = float("inf")
    best_stats = (0, 0, 0)
    for neighbor in available:
        nav_score = _score_navigation_neighbor(
            graph, current_node_id, neighbor, goal_node, tasks, player_id, player,
            gate_node_id, terminal_node_ids, weather, blocked, process_nodes,
            failed_task_ids, enemy_busy_task_ids, obstacle_nodes,
            visited_node_ids, bucket_scores,
        )
        route_stats = _route_task_stats(
            graph, current_node_id, neighbor, goal_node, tasks, player_id, player,
            gate_node_id, terminal_node_ids, weather, blocked, process_nodes,
            failed_task_ids, enemy_busy_task_ids, obstacle_nodes,
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
    if forward_available:
        available = forward_available
    else:
        available = all_safe or neighbors
    logger.info("_find_move_target: current=%s neighbors=%s available=%s visited=%s failed_target=%s",
                current_node_id, neighbors, available, visited_node_ids, failed_target)

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

    if goal_node and _should_use_task_aware_routing(player, phase, force_delivery):
        task_candidates = all_safe or available
        task_step = _pick_task_aware_neighbor(
            graph, current_node_id, task_candidates, goal_node, tasks, player_id, player,
            gate_node_id, terminal_node_ids, weather, guard_blocked,
            remaining_process_nodes, failed_task_ids, enemy_busy_task_ids,
            obstacle_nodes, visited_node_ids,
        )
        if task_step and task_step in available:
            route_score, route_tasks, route_high = _route_task_stats(
                graph, current_node_id, task_step, goal_node, tasks, player_id, player,
                gate_node_id, terminal_node_ids, weather, guard_blocked,
                remaining_process_nodes, failed_task_ids, enemy_busy_task_ids,
                obstacle_nodes,
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
    if active_contest_id:
        return active_contest_id
    if contests:
        for c in contests:
            if c.get("redPlayerId") == player_id or c.get("bluePlayerId") == player_id:
                if not c.get("resolved", False) and c.get("status") != "SUPPRESSED":
                    return c.get("contestId", "")
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
    """Node blocking our in-progress move (next hop or known guard)."""
    next_node = player.get("nextNodeId", "")
    if next_node and next_node in route_blocked:
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
    # inquire 可能未包含远程节点，仍尝试削弱
    return make_squad_weaken_action(target_node_id)


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
    avoid_route_nodes: set[str] | None = None,
    guard_stuck_rounds: int = 0,
    guard_stuck_target: str = "",
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
) -> dict:
    """WAIT (主车队) + SQUAD_WEAKEN (小分队) 每帧削弱设卡直到通行。"""
    if route_blocked is None:
        route_blocked = set()
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
                good = min(get_good_fruit(player), 2)
                bad = min(get_bad_fruit(player), 2)
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
) -> dict | None:
    if enemy_busy_task_ids is None:
        enemy_busy_task_ids = set()
    if get_task_score(player) >= TASK_SCORE_TARGET:
        return None
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

    my_task_score = get_task_score(player)
    if my_task_score >= TASK_SCORE_STRETCH and phase != "RUSH":
        return None

    hops_to_goal = float("inf")
    goal = goal_node_id or (terminal_node_ids[0] if terminal_node_ids else "")
    if goal and graph and current_node_id:
        hops_to_goal = graph.path_length(current_node_id, goal, weather, blocked)
    push_phase = my_task_score < 40 and hops_to_goal > 5

    if _player_processing_task(player):
        return None

    # Check if we're currently processing a task (策略文档 §5.2: 同时仅处理1个任务实例)
    for task in tasks:
        if (task.get("ownerPlayerId") == my_player_id
                and task.get("active", False)
                and not task.get("completed", False)
                and not task.get("failed", False)):
            return None

    # Try to claim task at current node (prioritized by score/round)
    task = find_task_at_node(
        tasks, current_node_id, my_player_id,
        graph_neighbors=graph.get_neighbors(current_node_id) if graph else None,
        enemy_busy_task_ids=enemy_busy_task_ids,
    )
    if task:
        template_id = get_task_template_id(task)
        if push_phase and not template_id.startswith(("T01", "T06", "T04")):
            task = None
        if template_id.startswith("T06") and not has_resource(player, "FAST_HORSE") and not has_resource(player, "SHORT_HORSE"):
            logger.debug("Round %d: Skipping T06 task (no horse)", round_num)
            task = None
        if task and template_id.startswith("T04") and not _t04_targets_obstacle(
            task, current_node_id, obstacle_nodes, graph,
        ):
            logger.debug("Round %d: Skipping T04 at %s (no obstacle)", round_num, current_node_id)
            task = None

    if task:
        # Check expireRound (策略文档 §5.2: 关注expireRound)
        expire_round = task.get("expireRound", 0)
        if expire_round > 0 and round_num >= expire_round:
            logger.debug("Round %d: Task %s expired", round_num, task.get("taskId", ""))
            task = None

    if task:
        # Skip tasks previously rejected with RESOURCE_NOT_ENOUGH
        task_id = task.get("taskId", "")
        if task_id and task_id in failed_task_ids:
            logger.debug("Round %d: Skipping failed task %s", round_num, task_id)
            task = None

    if task:
        task_id = task.get("taskId", "")
        if task_id:
            logger.info("Round %d: Claiming task %s (template=%s) at %s", round_num, task_id, template_id, current_node_id)
            return make_action(match_id, round_num, player_id, [
                make_claim_task_action(task_id)
            ])

    # Look for nearby tasks within detour cost (策略文档 §5.2 顺路原则)
    if my_task_score < TASK_SCORE_TARGET and not push_phase:
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
            if detour <= MAX_TASK_DETOUR_COST:
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
            best_task = candidates[0][0]
            task_node = best_task.get("nodeId", "")
            # Move toward task: avoid obstacles/guards, but do not block visited nodes
            soft_blocked = set(obstacle_nodes)
            if blocked:
                soft_blocked.update(blocked)
            soft_blocked.discard(task_node)
            step = graph.next_step_toward(
                current_node_id, task_node, weather, soft_blocked,
                use_weighted=True, process_nodes=process_nodes,
            )
            if not step:
                step = graph.next_step_toward(
                    current_node_id, task_node, weather, obstacle_nodes,
                    use_weighted=True, process_nodes=process_nodes,
                )
            if step:
                logger.info("Round %d: Moving toward task at %s (template=%s), step=%s", round_num, task_node, get_task_template_id(best_task), step)
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


def _handle_resources(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph, current_node_id: str,
    current_node: dict | None, phase: str, weather: dict | None,
    gate_node_id: str = "",
    process_nodes: dict[str, dict] | None = None,
) -> dict | None:
    """Handle resource claiming strategy (策略文档 §6).

    Returns action dict or None.
    """
    if current_node is None:
        return None
    if phase == "RUSH" or round_num >= 360:
        return None

    hops_to_gate = _hops_to_gate(graph, current_node_id, gate_node_id, weather, None)
    near_gate = hops_to_gate <= NEAR_GATE_RESOURCE_HOPS
    on_final_approach = hops_to_gate <= FINAL_CORRIDOR_GATE_HOPS
    at_palace_transfer = _is_palace_transfer_node(process_nodes, current_node_id)

    resources = find_available_resources(current_node)
    if not resources:
        return None

    my_resources = get_player_resources(player)

    # Filter to only high-value resources worth claiming
    HIGH_VALUE_RESOURCES = {"FAST_HORSE", "SHORT_HORSE", "ICE_BOX"}
    WINDOW_RESOURCES = {"OFFICIAL_PERMIT", "PASS_TOKEN"}

    for rtype, count in resources:
        # Skip if already have this resource
        if my_resources.get(rtype, 0) >= 1 and rtype in HIGH_VALUE_RESOURCES:
            continue
        # Only claim high-value resources (FAST_HORSE, SHORT_HORSE, ICE_BOX)
        if rtype in HIGH_VALUE_RESOURCES:
            logger.info("Round %d: Claiming resource %s at %s", round_num, rtype, current_node_id)
            return make_action(match_id, round_num, player_id, [
                make_claim_resource_action(current_node_id, rtype)
            ])
        # Claim OFFICIAL_PERMIT/PASS_TOKEN for window contests
        if rtype in WINDOW_RESOURCES:
            if near_gate:
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
        # Claim BOAT_RIGHT (策略文档 §6.1: 仅领取, passive)
        if rtype == "BOAT_RIGHT" and my_resources.get("BOAT_RIGHT", 0) < 1:
            logger.info("Round %d: Claiming BOAT_RIGHT at %s", round_num, current_node_id)
            return make_action(match_id, round_num, player_id, [
                make_claim_resource_action(current_node_id, rtype)
            ])
        # Skip INTEL — low value, not worth the frames early game

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
) -> dict | None:
    """Handle using resources: ice box, horses (策略文档 §6.1)."""
    freshness = get_freshness(player)
    force_delivery = _should_force_delivery(
        round_num, phase, player, graph, current_node_id,
        gate_node_id, terminal_node_ids, weather,
        process_nodes, processed_node_ids,
    )
    ice_threshold = ICE_BOX_FRESHNESS_THRESHOLD
    if force_delivery:
        ice_threshold = ICE_BOX_FRESHNESS_THRESHOLD + 8

    if has_resource(player, "ICE_BOX") and freshness <= ice_threshold:
        logger.info("Round %d: Using ICE_BOX (freshness=%.1f)", round_num, freshness)
        return make_action(match_id, round_num, player_id, [
            make_use_resource_action("ICE_BOX")
        ])

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


def _score_squad_clear_target(
    node_id: str,
    my_path: list[str],
    opp_path: list[str],
) -> float:
    """Higher score = more valuable to clear (own route, opponent tax, shared choke)."""
    on_my_path = node_id in my_path
    on_opp_path = bool(opp_path) and node_id in opp_path
    score = 0.0
    if on_my_path:
        score += 12.0
    if on_opp_path:
        score += 9.0
    if on_my_path and on_opp_path:
        score += 4.0
    elif on_opp_path and not on_my_path:
        score += 3.0
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
) -> str | None:
    """Pick obstacle node for SQUAD_CLEAR: own route first, then opponent tax routes."""
    if squad_count < SQUAD_CLEAR_MIN_SQUAD:
        return None

    my_path = _path_nodes_to_goal(
        graph, current_node_id, goal_node_id, weather, obstacle_nodes,
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
        if _node_has_planned_t04(tasks, nid, failed_task_ids, my_task_score):
            continue

        score = _score_squad_clear_target(nid, my_path, opp_path)
        if score <= 0:
            continue

        # Tight squad budget: only clear obstacles on our own planned route.
        if squad_count <= SQUAD_CLEAR_MIN_SQUAD + 1 and score < 12.0:
            continue
        # Moderate budget: require value on at least one meaningful path.
        if score < 9.0:
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
) -> tuple[bool, str]:
    """Decide whether to SET_GUARD at the fleet's current node."""
    goal = gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "")
    if not goal or not current_node_id:
        return False, ""

    if force_delivery:
        return False, "force-delivery"

    task_score = get_task_score(player)
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
        if _is_final_corridor_guard_site(
            graph, current_node_id, gate_node_id, inquire_nodes, process_nodes,
            weather, obstacle_nodes,
        ):
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
) -> dict | None:
    should, reason = _should_set_guard_dynamic(
        graph, current_node_id, gate_node_id, terminal_node_ids,
        weather, obstacle_nodes, player, inquire_nodes,
        my_team_id, player_id, opp_player, mode, phase,
        process_nodes=process_nodes,
        force_delivery=force_delivery,
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
    if not my_team_id:
        my_team_id = get_team_id(player)

    # --- BREAK_GUARD with optional BREAK_ORDER (策略文档 §8.2, §10) ---
    goal = gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "")
    if goal and blocked:
        optimal_path = graph.shortest_path(current_node_id, goal, weather, obstacle_nodes)
        if optimal_path and len(optimal_path) >= 2:
            next_hop = optimal_path[1]
            if next_hop in blocked:
                for node in inquire_nodes:
                    if node.get("nodeId") == next_hop:
                        guard = node.get("guard", {})
                        if is_enemy_guard(guard, my_team_id, player_id):
                            good = min(get_good_fruit(player), 2)
                            bad = min(get_bad_fruit(player), 2)
                            if good + bad > 0:
                                action = make_break_guard_action(next_hop, good_fruit=good, bad_fruit=bad)
                                # Bind BREAK_ORDER if in RUSH and have resources (策略文档 §10: +3攻坚)
                                if phase == "RUSH" and (bad >= 2 or good >= 1):
                                    action["rushTactic"] = "BREAK_ORDER"
                                    logger.info("Round %d: Breaking guard at %s with BREAK_ORDER", round_num, next_hop)
                                else:
                                    logger.info("Round %d: Breaking guard at %s (blocking path)", round_num, next_hop)
                                return make_action(match_id, round_num, player_id, [action])
                        # Try FORCED_PASS instead
                        logger.info("Round %d: Forced pass at %s", round_num, next_hop)
                        return make_action(match_id, round_num, player_id, [
                            make_forced_pass_action(next_hop)
                        ])

    # --- Squad actions (策略文档 §8.4) — only if not RUSH ---
    if phase != "RUSH":
        squad_count = get_squad_count(player)
        my_task_score = get_task_score(player)

        # Reserve squads for late key-pass guard weakening; scouting is optional.
        if squad_count >= 10 and my_task_score < TASK_SCORE_TARGET and process_nodes:
            for nid, info in process_nodes.items():
                if nid not in visited_node_ids and nid != current_node_id:
                    dist = graph.path_length(current_node_id, nid, weather, None)
                    if 0 < dist <= 15:
                        logger.info("Round %d: Squad scout at %s", round_num, nid)
                        return make_action(match_id, round_num, player_id, [
                            make_squad_scout_action(nid)
                        ])

        # SQUAD_CLEAR: own-route opening + opponent-route tax; keep late-game reserve.
        if squad_count >= SQUAD_CLEAR_MIN_SQUAD and goal:
            clear_target = _find_squad_clear_target(
                graph, current_node_id, goal, inquire_nodes, obstacle_nodes,
                weather, opp_player, tasks, failed_task_ids, my_task_score,
                squad_count, squad_clear_pending,
            )
            if clear_target:
                on_my = clear_target in _path_nodes_to_goal(
                    graph, current_node_id, goal, weather, obstacle_nodes,
                )
                on_opp = bool(opp_player) and clear_target in _path_nodes_to_goal(
                    graph, opp_player.get("currentNodeId", ""), goal, weather, obstacle_nodes,
                )
                reason = "own-route"
                if on_my and on_opp:
                    reason = "shared-choke"
                elif on_opp and not on_my:
                    reason = "opp-tax"
                logger.info(
                    "Round %d: Squad clear at %s (%s, squad=%d, reserve>=%d)",
                    round_num, clear_target, reason, squad_count, SQUAD_RESERVE_FOR_LATE,
                )
                return make_action(match_id, round_num, player_id, [
                    make_squad_clear_action(clear_target)
                ])

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
) -> dict | None:
    """Handle rush tactics: RUSH_SPEED, RUSH_PROTECT (策略文档 §10).

    Only available after RUSH phase. Each can be used once per match.
    RUSH_SPEED与马互斥: 有马buff时不使用.
    """
    if phase != "RUSH":
        return None

    # RUSH_SPEED can only be used when IDLE
    state = player.get("state", "")
    if state != "IDLE":
        return None

    freshness = get_freshness(player)

    # RUSH_PROTECT: 鲜度<50, 停靠节点使用 (策略文档 §10: 0成本)
    if freshness < RUSH_PROTECT_FRESHNESS:
        logger.info("Round %d: Using RUSH_PROTECT (freshness=%.1f)", round_num, freshness)
        return make_action(match_id, round_num, player_id, [make_rush_protect_action()])

    # The current server rejects standalone RUSH_SPEED as INVALID_ACTION_TYPE.
    return None
