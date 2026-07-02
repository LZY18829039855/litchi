package com.litchi.net;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

public final class ProtocolMessages {

    private ProtocolMessages() {
    }

    public static JsonObject registration(int playerId, String playerName, String version) {
        JsonObject data = new JsonObject();
        data.addProperty("playerId", playerId);
        data.addProperty("playerName", playerName);
        data.addProperty("version", version);
        return wrap("registration", data);
    }

    public static JsonObject ready(String matchId, int round, int playerId) {
        JsonObject data = new JsonObject();
        data.addProperty("matchId", matchId);
        data.addProperty("round", round);
        data.addProperty("playerId", playerId);
        return wrap("ready", data);
    }

    public static JsonObject action(String matchId, int round, int playerId, JsonArray actions) {
        JsonObject data = new JsonObject();
        data.addProperty("matchId", matchId);
        data.addProperty("round", round);
        data.addProperty("playerId", playerId);
        data.add("actions", actions);
        return wrap("action", data);
    }

    public static JsonObject wrap(String name, JsonObject data) {
        JsonObject root = new JsonObject();
        root.addProperty("msg_name", name);
        root.add("msg_data", data);
        return root;
    }
}
