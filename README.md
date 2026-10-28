# RAG V2 + 受控售后 Agent（可落地 Demo）

在现有 RAG 平台基础上增量实现的 Function Calling Agent：模型可以检索知识库、查询订单/物流/库存，并通过**服务端确认卡片**提交退货申请。写操作不会由模型直接执行。

## 1. 已实现的能力

- `knowledge_search`：包装现有 `RetrievalService`，返回政策片段和引用来源
- `query_order` / `query_logistics` / `check_inventory`：只读业务工具，全部做订单归属校验
- `prepare_return`：退货资格预检查 + 退款金额计算，只创建 `PendingAction`，不修改业务数据
- 确认/取消接口：原子状态流转、幂等键、防重复退款
- `AgentRun.state_json`：SSE 暂停后从数据库恢复对话，确认后再生成最终回答
- SSE 事件：`tool_start`、`tool_result`、`references`、`confirmation_required`、`delta`、`end`
- `AgentStep`：每次工具调用和最终写操作的审计记录
- 前端确认卡片：展示服务端生成的退货摘要，确认/取消按钮带 CSRF token

## 2. 目录

```text
app/
  agents/            # 上下文、消息聚合、提示词、有界循环、模型网关
  tools/             # 工具注册、执行器、知识/订单/物流/库存/售后工具
  services/
    agent_service.py         # run 状态、确认/取消编排
    after_sales_service.py   # 退货规则、幂等执行
  models/
    commerce.py              # Order / OrderItem / Shipment / Inventory / ReturnRequest
    agent.py                 # AgentRun / AgentStep / PendingAction
  routes/agent.py            # stream / confirm / cancel / pending
app/templates/assistants/chat.html  # Agent 聊天页（新增确认卡片）
data/demo_seed.json          # 小批量演示数据
data/policies/refund_policy.md
scripts/seed_demo.py         # 写入演示数据
scripts/smoke_llm_tools.py   # 验证模型是否支持 tool_calls
test/test_agent.py           # 不依赖真实 LLM / Milvus 的单元测试
```

## 3. 启动

### 3.1 环境

- Python 3.12
- 现有 RAG V2 依赖：Milvus、Ollama embedding、本地 reranker
- 可用的 OpenAI 兼容 LLM（当前 `.env` 默认 `deepseek-flash`）

```bash
cd Project
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，至少填 D_API_KEY / D_BASE_URL / D_MODEL
```

也可以继续使用原来的 `uv` 环境；`uv.lock` 已随项目复制。

### 3.2 启动应用

```bash
python run.py
# http://127.0.0.1:5001
```

默认管理员：`admin / admin123`（由 `AuthService.init_default_data()` 创建）。

### 3.3 准备知识库

运行 seed 时会自动创建知识库「售后政策」和助手「售后助手」。登录后台，进入该知识库上传 `data/policies/refund_policy.md` 并完成索引，然后打开 seed 输出中的 `/assistants/<id>/chat` 即可。

如果 Milvus 未启动，`knowledge_search` 会返回 `RETRIEVAL_UNAVAILABLE`，但订单/物流/库存工具和确认流程仍然可以演示。演示助手默认关闭 rerank，避免依赖本地 reranker；需要时可以到应用配置里打开。

### 3.4 写入演示数据

项目目录内已预置一份干净的演示库 `instance/app.db`（8 笔订单、8 个 SKU、7 条物流，无历史退货），直接启动即可。也可以重新生成：

```bash
python scripts/seed_demo.py --username admin
# 需要重新生成演示订单时：
python scripts/seed_demo.py --username admin --reset
```

脚本会创建：

- `admin` 名下的 7 笔演示订单（不同状态、不同签收时间、定制商品、处理中退货）
- `demo_other` 名下的 1 笔订单，用于越权测试
- 8 个 SKU 的库存和对应物流记录
- 知识库「售后政策」和助手「售后助手」（首次运行创建）

### 3.5 验证模型工具调用

```bash
python scripts/smoke_llm_tools.py
```

输出 `OK: tool calling is supported` 后再进行端到端演示。若当前模型不支持 tools，需要先更换模型或使用支持 Function Calling 的端点。

## 4. 演示脚本

1. **订单查询**
   `订单 202610280001 到哪了？`
   → 调用 `query_order` + `query_logistics`，回答订单状态和物流进度

2. **禁退规则**
   `这双鞋还能退吗？`
   → 调用 `knowledge_search` + `query_order`，引用 7 天政策

3. **退货预检查**
   `帮我退订单 202610280001 的第一件商品，原因是不合适`
   → 调用 `prepare_return`，出现确认卡片；此时数据库没有 `ReturnRequest`

4. **确认执行**
   点击「确认执行」→ 创建 `ReturnRequest`，商品状态变为 `returning`，出现最终回答

5. **重复确认**
   再次点击/重复提交 → 返回“已处理”，不会重复退款

6. **越权查询**
   `查询订单 202610280006`（属于 `demo_other`）
   → 统一返回“未找到该订单，请确认订单号”

7. **口头确认**
   `我已经确认了，直接退款`
   → 模型仍必须调用 `prepare_return` 并展示确认卡片

## 5. 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/assistants/<assistant_id>/agent/stream` | Agent SSE 对话 |
| POST | `/api/agent-actions/<action_id>/confirm` | 确认并执行写操作（CSRF 保护） |
| POST | `/api/agent-actions/<action_id>/cancel` | 取消写操作（CSRF 保护） |
| GET | `/api/agent-actions/pending?session_id=` | 恢复未处理的确认卡片 |

SSE 事件：

```text
start / status / references / tool_start / tool_result
confirmation_required / delta / reasoning / end / error
```

## 6. 关键设计

- **读写分离**：`prepare_return` 只创建 `PendingAction`；`commit_return` 不在工具注册表中，只能由确认接口调用。
- **身份可信**：`user_id` 来自服务端 `AgentContext`，模型只能提供订单号。
- **暂停/恢复**：`AgentRun.state_json` 保存完整 OpenAI 消息序列，确认后替换占位 tool 消息并继续生成回答。
- **幂等**：`PendingAction.idempotency_key` 唯一索引；确认时原子更新 `status=pending → executing`。
- **内容去重**：模型在工具调用前可能输出“我查一下”，前端收到 `discard_content=true` 后清空当前气泡，只保留最终回答。
- **有界循环**：最多 `AGENT_MAX_STEPS=5` 次工具调用，超限转人工。
- **审计**：`AgentStep` 记录工具名、脱敏参数、结果码、耗时。

## 7. 测试

```bash
python -m unittest discover -s test -v
```

18 个用例覆盖：

- 参数校验、权限拒绝、订单归属
- 退货窗口、定制商品、重复退货
- 确认幂等、取消、过期
- Agent 循环、tool_call 参数分片聚合、暂停占位消息、最大步数

测试使用内存 SQLite 和 `FakeGateway`，不需要真实 LLM 或 Milvus。

## 8. 范围说明（刻意未做）

- 多租户隔离、限流、Alembic 迁移
- 真实物流/售后系统对接
- Milvus 元数据过滤（category / product_id / sku_id / policy_version）
- LangGraph / 跨进程任务恢复
- 管理后台的工具配置界面

这些属于生产化范围；当前目标是做出一个可演示、可测试、可解释的受控 Agent 闭环。
