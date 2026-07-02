package com.litchi.game;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class GameMap {

    public static final class Edge {
        private final String edgeId;
        private final String from;
        private final String to;
        private final String routeType;
        private final int distance;
        private final boolean bidirectional;

        public Edge(String edgeId, String from, String to, String routeType, int distance, boolean bidirectional) {
            this.edgeId = edgeId;
            this.from = from;
            this.to = to;
            this.routeType = routeType;
            this.distance = distance;
            this.bidirectional = bidirectional;
        }

        public String getFrom() {
            return from;
        }

        public String getTo() {
            return to;
        }

        public String getRouteType() {
            return routeType;
        }

        public int getDistance() {
            return distance;
        }

        public boolean isBidirectional() {
            return bidirectional;
        }
    }

    public static final class NodeInfo {
        private final String nodeId;
        private final String processType;
        private final int processRound;

        public NodeInfo(String nodeId, String processType, int processRound) {
            this.nodeId = nodeId;
            this.processType = processType;
            this.processRound = processRound;
        }

        public String getNodeId() {
            return nodeId;
        }

        public String getProcessType() {
            return processType;
        }

        public int getProcessRound() {
            return processRound;
        }
    }

    private final Map<String, NodeInfo> nodes = new HashMap<>();
    private final List<Edge> edges = new ArrayList<>();
    private String startNodeId = "S01";
    private String gateNodeId = "S14";
    private String terminalNodeId = "S15";

    public void loadFromStart(JsonObject startData) {
        nodes.clear();
        edges.clear();
        if (startData.has("map") && startData.get("map").isJsonObject()) {
            JsonObject map = startData.getAsJsonObject("map");
            if (map.has("gameplay") && map.get("gameplay").isJsonObject()) {
                JsonObject gameplay = map.getAsJsonObject("gameplay");
                if (gameplay.has("roles") && gameplay.get("roles").isJsonObject()) {
                    JsonObject roles = gameplay.getAsJsonObject("roles");
                    if (roles.has("startNodeId")) {
                        startNodeId = roles.get("startNodeId").getAsString();
                    }
                    if (roles.has("gateNodeId")) {
                        gateNodeId = roles.get("gateNodeId").getAsString();
                    }
                    if (roles.has("terminalNodeIds") && roles.get("terminalNodeIds").isJsonArray()
                            && !roles.getAsJsonArray("terminalNodeIds").isEmpty()) {
                        terminalNodeId = roles.getAsJsonArray("terminalNodeIds").get(0).getAsString();
                    }
                }
                if (gameplay.has("processNodes") && gameplay.get("processNodes").isJsonArray()) {
                    for (JsonElement element : gameplay.getAsJsonArray("processNodes")) {
                        JsonObject processNode = element.getAsJsonObject();
                        String nodeId = processNode.get("nodeId").getAsString();
                        String processType = processNode.has("processType") ? processNode.get("processType").getAsString() : null;
                        int processRound = processNode.has("processRound") ? processNode.get("processRound").getAsInt() : 0;
                        nodes.put(nodeId, new NodeInfo(nodeId, processType, processRound));
                    }
                }
            }
        }
        if (startData.has("nodes") && startData.get("nodes").isJsonArray()) {
            for (JsonElement element : startData.getAsJsonArray("nodes")) {
                JsonObject node = element.getAsJsonObject();
                String nodeId = node.get("nodeId").getAsString();
                String processType = node.has("processType") && !node.get("processType").isJsonNull()
                        ? node.get("processType").getAsString() : null;
                int processRound = node.has("processRound") ? node.get("processRound").getAsInt() : 0;
                nodes.putIfAbsent(nodeId, new NodeInfo(nodeId, processType, processRound));
            }
        }
        if (startData.has("edges") && startData.get("edges").isJsonArray()) {
            for (JsonElement element : startData.getAsJsonArray("edges")) {
                edges.add(parseEdge(element.getAsJsonObject()));
            }
        }
    }

    public void updateEdges(JsonArray edgeArray) {
        if (edgeArray == null) {
            return;
        }
        edges.clear();
        for (JsonElement element : edgeArray) {
            edges.add(parseEdge(element.getAsJsonObject()));
        }
    }

    private Edge parseEdge(JsonObject edge) {
        String from = edge.has("fromNodeId") ? edge.get("fromNodeId").getAsString() : edge.get("fromNode").getAsString();
        String to = edge.has("toNodeId") ? edge.get("toNodeId").getAsString() : edge.get("toNode").getAsString();
        String edgeId = edge.has("edgeId") ? edge.get("edgeId").getAsString() : from + "->" + to;
        String routeType = edge.has("routeType") ? edge.get("routeType").getAsString() : "ROAD";
        int distance = edge.has("distance") ? edge.get("distance").getAsInt() : 1;
        boolean bidirectional = edge.has("bidirectional") && edge.get("bidirectional").getAsBoolean();
        return new Edge(edgeId, from, to, routeType, distance, bidirectional);
    }

    public String getStartNodeId() {
        return startNodeId;
    }

    public String getGateNodeId() {
        return gateNodeId;
    }

    public String getTerminalNodeId() {
        return terminalNodeId;
    }

    public NodeInfo getNode(String nodeId) {
        return nodes.get(nodeId);
    }

    public List<String> neighbors(String nodeId, Set<String> blockedNodes) {
        List<String> result = new ArrayList<>();
        for (Edge edge : edges) {
            if (edge.getFrom().equals(nodeId) && !blockedNodes.contains(edge.getTo())) {
                result.add(edge.getTo());
            }
            if (edge.isBidirectional() && edge.getTo().equals(nodeId) && !blockedNodes.contains(edge.getFrom())) {
                result.add(edge.getFrom());
            }
        }
        return result;
    }

    public Edge findEdge(String from, String to) {
        for (Edge edge : edges) {
            if (edge.getFrom().equals(from) && edge.getTo().equals(to)) {
                return edge;
            }
            if (edge.isBidirectional() && edge.getFrom().equals(to) && edge.getTo().equals(from)) {
                return edge;
            }
        }
        return null;
    }

    public List<String> shortestPath(String from, String to, Set<String> blockedNodes) {
        if (from.equals(to)) {
            return java.util.Collections.singletonList(from);
        }
        Map<String, String> previous = new HashMap<>();
        Map<String, Integer> distance = new HashMap<>();
        Set<String> visited = new HashSet<>();
        List<String> queue = new ArrayList<>();
        queue.add(from);
        distance.put(from, 0);
        while (!queue.isEmpty()) {
            String current = queue.remove(0);
            if (!visited.add(current)) {
                continue;
            }
            if (current.equals(to)) {
                break;
            }
            for (String next : neighbors(current, blockedNodes)) {
                int nextDistance = distance.getOrDefault(current, Integer.MAX_VALUE / 4) + edgeCost(current, next);
                if (nextDistance < distance.getOrDefault(next, Integer.MAX_VALUE / 4)) {
                    distance.put(next, nextDistance);
                    previous.put(next, current);
                    queue.add(next);
                }
            }
        }
        if (!from.equals(to) && !previous.containsKey(to)) {
            return Collections.emptyList();
        }
        List<String> path = new ArrayList<>();
        String cursor = to;
        while (cursor != null) {
            path.add(cursor);
            cursor = previous.get(cursor);
        }
        Collections.reverse(path);
        return path;
    }

    private int edgeCost(String from, String to) {
        Edge edge = findEdge(from, to);
        if (edge == null) {
            return 1000;
        }
        int factor;
        switch (edge.getRouteType()) {
            case "WATER":
                factor = 1250;
                break;
            case "MOUNTAIN":
                factor = 1780;
                break;
            case "BRANCH":
                factor = 1550;
                break;
            default:
                factor = 1380;
                break;
        }
        return edge.getDistance() * factor;
    }
}
