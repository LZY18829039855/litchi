"""Map graph: building adjacency from start/inquire map data with weather-aware routing."""

from __future__ import annotations

import heapq
import logging
from collections import deque
from typing import Any


# Route type cost multipliers (策略文档 §3.2)
ROUTE_COST_FACTOR = {
    "ROAD": 1380,
    "WATER": 1250,
    "MOUNTAIN": 1780,
    "BRANCH": 1550,
}

# Freshness loss per frame by route type
ROUTE_FRESHNESS_LOSS = {
    "ROAD": 0.055,
    "WATER": 0.045,
    "MOUNTAIN": 0.07,
    "BRANCH": 0.065,
}

# Weather penalty multipliers (策略文档 §3.2)
WEATHER_ROUTE_PENALTY = {
    "HOT": {"MOUNTAIN": 1.5, "ROAD": 1.2},     # 酷暑: 山路鲜度×1.5
    "HEAVY_RAIN": {"WATER": 1.5},               # 暴雨: 水路耗时×1.5
    "MOUNTAIN_FOG": {"MOUNTAIN": 1.3},           # 山雾: 山路耗时×1.3
}

# Process type cost in frames (策略文档 §4.1)
PROCESS_COST_FRAMES = {
    "TRANSFER": 4,         # S02 前段交接
    "BOARD": 7,            # S04 登船
    "WATER_TRANSFER": 6,   # S05 水路换运
    "PASS_TRANSFER": 5,    # S11 入关交接
    "PALACE_TRANSFER": 5,  # S13 宫前交接
    "VERIFY": 6,           # S14 宫门验核
}

# Process cost penalty for route selection — directly in frame units
# Since edge_cost returns distance * routeCostFactor / 1000 (≈ frames),
# process penalty is just the frame count
PROCESS_COST_PENALTY = {k: v for k, v in PROCESS_COST_FRAMES.items()}

logger = logging.getLogger("lychee_client.map_graph")


class MapGraph:
    """Adjacency graph built from nodes[] and edges[] in start or inquire messages.

    Does NOT hardcode any node IDs. All structure comes from the provided data.
    Supports weighted shortest path with route-type and weather awareness.
    """

    def __init__(self, nodes: list[dict], edges: list[dict]):
        self.node_ids: set[str] = set()
        self.adjacency: dict[str, list[str]] = {}  # nodeId -> [neighbor nodeIds]
        self.edge_info: dict[tuple[str, str], dict] = {}  # (from, to) -> edge data
        self.node_info: dict[str, dict] = {}  # nodeId -> node data

        for node in nodes:
            nid = node.get("nodeId", "")
            if nid:
                self.node_ids.add(nid)
                self.adjacency.setdefault(nid, [])
                self.node_info[nid] = node

        for edge in edges:
            from_id = edge.get("fromNodeId") or edge.get("fromNode", "")
            to_id = edge.get("toNodeId") or edge.get("toNode", "")
            bidirectional = edge.get("bidirectional", False)
            if not from_id or not to_id:
                continue
            self.adjacency.setdefault(from_id, [])
            self.adjacency.setdefault(to_id, [])
            if to_id not in self.adjacency[from_id]:
                self.adjacency[from_id].append(to_id)
            self.edge_info[(from_id, to_id)] = edge
            if bidirectional:
                if from_id not in self.adjacency[to_id]:
                    self.adjacency[to_id].append(from_id)
                self.edge_info[(to_id, from_id)] = edge

    def get_neighbors(self, node_id: str) -> list[str]:
        """Return list of neighbor node IDs reachable from node_id."""
        return self.adjacency.get(node_id, [])

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self.node_ids

    def get_edge(self, from_id: str, to_id: str) -> dict | None:
        """Return edge data for the given direction, or None."""
        return self.edge_info.get((from_id, to_id))

    def get_node(self, node_id: str) -> dict | None:
        """Return node data, or None."""
        return self.node_info.get(node_id)

    def remove_edge(self, from_id: str, to_id: str) -> None:
        """Remove an edge from the graph (e.g., when TARGET_NOT_REACHABLE)."""
        # Remove from adjacency
        if to_id in self.adjacency.get(from_id, []):
            self.adjacency[from_id].remove(to_id)
        # Remove edge info
        self.edge_info.pop((from_id, to_id), None)
        # If bidirectional, also remove reverse
        reverse_edge = self.edge_info.get((to_id, from_id))
        if reverse_edge and reverse_edge.get("bidirectional", False):
            if from_id in self.adjacency.get(to_id, []):
                self.adjacency[to_id].remove(from_id)
            self.edge_info.pop((to_id, from_id), None)
        logger.info("Removed edge %s -> %s", from_id, to_id)

    def get_edge_route_type(self, from_id: str, to_id: str) -> str:
        """Get the route type of an edge (ROAD/WATER/MOUNTAIN/BRANCH)."""
        edge = self.get_edge(from_id, to_id)
        if edge:
            return edge.get("routeType", "ROAD")
        return "ROAD"

    def edge_cost(
        self, from_id: str, to_id: str,
        weather: dict | None = None,
        blocked_nodes: set[str] | None = None,
        process_nodes: dict[str, dict] | None = None,
    ) -> float:
        """Calculate edge traversal cost for weighted pathfinding.

        Cost = distance * routeType_factor, plus weather and process penalties.
        Returns infinity if the target node is blocked.
        """
        if blocked_nodes and to_id in blocked_nodes:
            return float('inf')

        route_type = self.get_edge_route_type(from_id, to_id)
        base_factor = ROUTE_COST_FACTOR.get(route_type, 1380)

        # Get edge distance
        edge = self.get_edge(from_id, to_id)
        distance = edge.get("distance", 30) if edge else 30

        # Cost in frame units: distance * routeCostFactor / 1000
        base_cost = distance * base_factor / 1000

        # Apply weather penalty (协议: region=ALL/WATER/MOUNTAIN + type=HOT/HEAVY_RAIN/MOUNTAIN_FOG)
        if weather:
            weather_events = list(weather.get("active", [])) + list(weather.get("forecast", []))
            for fw in weather_events:
                weather_type = fw.get("type", "")
                region = fw.get("region", "")
                route_penalties = WEATHER_ROUTE_PENALTY.get(weather_type, {})
                if not route_penalties:
                    continue
                if region == "ALL" or region == weather_type:
                    if route_type in route_penalties:
                        base_cost *= route_penalties[route_type]
                elif region == "WATER" and route_type == "WATER":
                    if route_type in route_penalties:
                        base_cost *= route_penalties[route_type]
                elif region == "MOUNTAIN" and route_type == "MOUNTAIN":
                    if route_type in route_penalties:
                        base_cost *= route_penalties[route_type]

        # Add process cost penalty for the target node
        if process_nodes and to_id in process_nodes:
            pt = process_nodes[to_id].get("processType", "")
            if pt in PROCESS_COST_PENALTY:
                base_cost += PROCESS_COST_PENALTY[pt]

        return base_cost

    def shortest_path(
        self, from_id: str, to_id: str,
        weather: dict | None = None,
        blocked_nodes: set[str] | None = None,
    ) -> list[str]:
        """BFS shortest path (unweighted) from from_id to to_id.

        Returns list of node IDs (including from_id and to_id), or empty list if unreachable.
        """
        if from_id == to_id:
            return [from_id]
        if from_id not in self.adjacency or to_id not in self.adjacency:
            return []
        visited = {from_id}
        prev: dict[str, str] = {}
        queue = deque([from_id])
        while queue:
            node = queue.popleft()
            for neighbor in self.adjacency.get(node, []):
                if neighbor not in visited:
                    if blocked_nodes and neighbor in blocked_nodes and neighbor != to_id:
                        continue
                    visited.add(neighbor)
                    prev[neighbor] = node
                    if neighbor == to_id:
                        path = [to_id]
                        cur = to_id
                        while cur in prev:
                            cur = prev[cur]
                            path.append(cur)
                        path.reverse()
                        return path
                    queue.append(neighbor)
        return []

    def weighted_shortest_path(
        self, from_id: str, to_id: str,
        weather: dict | None = None,
        blocked_nodes: set[str] | None = None,
        process_nodes: dict[str, dict] | None = None,
    ) -> list[str]:
        """Dijkstra weighted shortest path considering route costs, weather, and process costs.

        Returns list of node IDs (including from_id and to_id), or empty list if unreachable.
        """
        if from_id == to_id:
            return [from_id]
        if from_id not in self.adjacency or to_id not in self.adjacency:
            return []

        dist: dict[str, float] = {from_id: 0.0}
        prev: dict[str, str] = {}
        heap = [(0.0, from_id)]
        visited: set[str] = set()

        while heap:
            d, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            if node == to_id:
                path = [to_id]
                cur = to_id
                while cur in prev:
                    cur = prev[cur]
                    path.append(cur)
                path.reverse()
                return path
            for neighbor in self.adjacency.get(node, []):
                if neighbor in visited:
                    continue
                cost = self.edge_cost(node, neighbor, weather, blocked_nodes, process_nodes)
                new_dist = d + cost
                if new_dist < dist.get(neighbor, float('inf')):
                    dist[neighbor] = new_dist
                    prev[neighbor] = node
                    heapq.heappush(heap, (new_dist, neighbor))
        return []

    def next_step_toward(
        self, from_id: str, to_id: str,
        weather: dict | None = None,
        blocked_nodes: set[str] | None = None,
        use_weighted: bool = False,
        process_nodes: dict[str, dict] | None = None,
    ) -> str | None:
        """Return the first step of the shortest path from from_id toward to_id.

        Returns the neighbor node ID to move to, or None if unreachable.
        """
        if use_weighted:
            path = self.weighted_shortest_path(from_id, to_id, weather, blocked_nodes, process_nodes)
        else:
            path = self.shortest_path(from_id, to_id, weather, blocked_nodes)
        if len(path) >= 2:
            return path[1]
        return None

    def path_length(
        self, from_id: str, to_id: str,
        weather: dict | None = None,
        blocked_nodes: set[str] | None = None,
    ) -> int:
        """Return the number of hops in the shortest path, or infinity if unreachable."""
        path = self.shortest_path(from_id, to_id, weather, blocked_nodes)
        return len(path) - 1 if path else float('inf')