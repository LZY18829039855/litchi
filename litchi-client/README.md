# 荔枝争运战 Java 客户端（初版）

## 功能

- TCP 5 位长度前缀拆包/粘包
- 完整协议流程：`registration -> start -> ready -> inquire/action -> over`
- 每帧自动决策（基础策略）：
  - 地图寻路前往宫门/终点
  - 自动处理站点流程、领取资源、顺路做皇榜任务
  - 宫宴冲刺后验核与交付
  - 窗口争夺默认出牌
  - 小分队探路

## 本地编译

```bash
cd litchi-client
mvn -q package
```

## 本地调试（Windows PowerShell）

```powershell
cd litchi-client
mvn -q package
java -jar target/litchi-client.jar <playerId> <host> <port>
```

示例：

```powershell
java -jar target/litchi-client.jar 1001 127.0.0.1 8081
```

## Linux 提交包

```bash
cd litchi-client
mvn -q package
cp target/litchi-client.jar .
chmod +x start.sh
```

将 `start.sh` 与 `litchi-client.jar` 放在 ZIP 根目录提交。

## 注意事项

1. 每帧必须回包，空动作发送 `actions: []`
2. `action.round` 必须等于 `inquire.round`
3. 地图、节点、路线以服务端 `start/inquire` 下发为准，不要写死
4. 当前策略为初版，后续可重点增强：设卡干扰、攻坚破卡、任务 90 分路线优化
