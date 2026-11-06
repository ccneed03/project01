# 系统架构

## 分层

```text
用户前端
  ↓
Agent 编排
  ├── 知识能力：RAG 检索、引用、政策文本
  ├── 事实能力：订单、物流、商品只读 API
  └── 决策能力：ReturnPolicyEngine
  ↓
安全控制面
身份 → 归属 → 参数校验 → 动作预览 → 用户确认
→ 提交前复核 → 幂等执行 → 状态回查 → 审计
  ↓
退货申请服务（唯一写入方）
```

知识层、行动层和安全层是产品叙事。代码边界上，资格决策和控制面是独立组件，避免把“安全”写成一句提示词。

## 编排循环

1. 用户消息进入 `AgentRuntime`；
2. 模型按需调用 `knowledge_search`、`query_order`、`query_logistics` 或 `search_product`；
3. 模型必须知道商品状态和原因类型，才能调用 `prepare_return`；
4. `ReturnPolicyEngine` 选择政策版本并生成逐商品决策；
5. 合格时创建 `PendingAction`，SSE 发送确认卡片并暂停；
6. 用户确认时提交确认令牌；
7. 服务端重新校验订单、商品、金额和规则快照；
8. `AfterSalesService.commit_return` 原子创建退货申请并更新商品状态；
9. Agent 根据服务端结果生成最终答复。

## 政策选择

每条规则包含：

- `policy_version`；
- `category`；
- `effective_from` / `effective_to`；
- 退货窗口；
- 是否允许无理由退货；
- 无理由退货是否要求未拆封；
- 是否允许质量问题人工核验；
- 政策文档和章节来源。

资格判断以订单签收时间对应的有效规则为准。类别规则优先于 `*` 默认规则。相同版本和品类由唯一约束防止冲突。

## 决策结果

每条订单商品都会得到：

- 是否可提交；
- 是否进入人工核验；
- 结果码；
- 用户可读原因；
- 政策版本；
- 规则快照。

`PendingAction.payload_json` 保存预览时的决策，`ReturnRequest.decision_json` 保存最终决策，便于审计和复现。

## 状态机

确认动作：

```text
pending → executing → success
                   ↘ failed
pending → cancelled
pending → expired
```

退货申请当前使用：

```text
submitted       # 普通申请已受理
manual_review   # 质量问题待人工核验
approved        # 预留：人工核验通过
refunded        # 预留：退款系统回调
failed          # 预留：处理失败
```

在 MVP 中 `commit_return` 只负责创建申请和更新订单商品状态，不负责退款。

## 幂等和一致性

- `PendingAction.idempotency_key` 防止重复生成同一待确认动作；
- 确认前使用条件更新把 `pending` 原子切换为 `executing`；
- `ReturnRequest.idempotency_key` 唯一约束防止重复创建申请；
- 成功动作收到重复确认时返回已保存的同一结果，不再次生成解释消息；
- 同一商品的状态更新和退货申请创建在同一个数据库事务中；
- 跨订单、物流、退货和财务系统的全局原子性不在当前范围；生产接入时需要状态机、Outbox/Saga 和对账。

## 主要失败模式

| 场景 | 行为 |
|---|---|
| 没有有效政策 | `POLICY_UNAVAILABLE`，不创建动作 |
| 超过窗口 | `RETURN_WINDOW_EXPIRED`，不创建动作 |
| 已拆封且无理由 | `ITEM_CONDITION_NOT_ELIGIBLE` |
| 质量问题 | 窗口内允许提交 `manual_review` |
| 商品已在退货流程 | `ITEM_ALREADY_IN_RETURN` |
| 确认后状态变化 | `CONFIRMATION_STALE` |
| 确认令牌错误 | `CONFIRMATION_TOKEN_INVALID` |
| 重复确认 | `ALREADY_PROCESSED` |
| 工具超限 | 停止循环并转人工 |
