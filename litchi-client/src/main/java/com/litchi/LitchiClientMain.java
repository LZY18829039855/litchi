package com.litchi;

import com.litchi.config.ClientConfig;
import com.litchi.game.GameSession;

public final class LitchiClientMain {

    public static void main(String[] args) {
        try {
            ClientConfig config = ClientConfig.fromArgs(args);
            GameSession session = new GameSession(config);
            session.run();
        } catch (Exception e) {
            System.err.println("客户端启动失败: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }
}
