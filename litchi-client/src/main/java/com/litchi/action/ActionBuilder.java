package com.litchi.action;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

public final class ActionBuilder {

    private ActionBuilder() {
    }

    public static JsonObject of(String action) {
        JsonObject obj = new JsonObject();
        obj.addProperty("action", action);
        return obj;
    }

    public static JsonObject move(String targetNodeId) {
        JsonObject obj = of("MOVE");
        obj.addProperty("targetNodeId", targetNodeId);
        return obj;
    }

    public static JsonObject claimResource(String targetNodeId, String resourceType) {
        JsonObject obj = of("CLAIM_RESOURCE");
        obj.addProperty("targetNodeId", targetNodeId);
        obj.addProperty("resourceType", resourceType);
        return obj;
    }

    public static JsonObject useResource(String resourceType, String targetNodeId) {
        JsonObject obj = of("USE_RESOURCE");
        obj.addProperty("resourceType", resourceType);
        if (targetNodeId != null) {
            obj.addProperty("targetNodeId", targetNodeId);
        }
        return obj;
    }

    public static JsonObject claimTask(String taskId) {
        JsonObject obj = of("CLAIM_TASK");
        obj.addProperty("taskId", taskId);
        return obj;
    }

    public static JsonObject process(String targetNodeId) {
        JsonObject obj = of("PROCESS");
        if (targetNodeId != null) {
            obj.addProperty("targetNodeId", targetNodeId);
        }
        return obj;
    }

    public static JsonObject verifyGate(String targetNodeId, String rushTactic) {
        JsonObject obj = of("VERIFY_GATE");
        obj.addProperty("targetNodeId", targetNodeId);
        if (rushTactic != null) {
            obj.addProperty("rushTactic", rushTactic);
        }
        return obj;
    }

    public static JsonObject deliver() {
        return of("DELIVER");
    }

    public static JsonObject windowCard(String contestId, String card) {
        JsonObject obj = of("WINDOW_CARD");
        obj.addProperty("contestId", contestId);
        obj.addProperty("card", card);
        return obj;
    }

    public static JsonObject squadScout(String targetNodeId) {
        JsonObject obj = of("SQUAD_SCOUT");
        obj.addProperty("targetNodeId", targetNodeId);
        return obj;
    }

    public static JsonObject rushSpeed() {
        return of("RUSH_SPEED");
    }

    public static JsonObject rushProtect() {
        return of("RUSH_PROTECT");
    }

    public static JsonArray single(JsonObject action) {
        JsonArray actions = new JsonArray();
        actions.add(action);
        return actions;
    }

    public static JsonArray pair(JsonObject mainAction, JsonObject squadAction) {
        JsonArray actions = new JsonArray();
        actions.add(mainAction);
        if (squadAction != null) {
            actions.add(squadAction);
        }
        return actions;
    }
}
