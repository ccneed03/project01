# 售后 Agent

在现有 RAG 平台基础上增量实现的 Function Calling Agent：模型可以检索知识库、查询订单/物流/库存，并通过**服务端确认卡片**提交退货申请。写操作不会由模型直接执行。

## 1. 已实现的能力

- `knowledge_search`：包装现有 `RetrievalService`，返回政策片段和引用来源；Milvus/Ollama 不可用时降级到本地 `refund_policy.md`，来源标记为 `local_fallback`
- `query_order` / `query_logistics` / `check_inventory`：只读业务工具，全部做订单归属校验
- `search_product`：查询商品目录（产品说明、材质/尺码属性、洗涤/发货/退换说明），只读
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
scripts/bootstrap_demo.py   # 一键重置演示数据 + 可选索引
scripts/preflight.py         # 演示前自检（数据、窗口、LLM、Milvus）
scripts/seed_demo.py         # 写入演示数据
scripts/index_products.py    # 商品向量索引（默认写入助手绑定 KB）
scripts/smoke_llm_tools.py   # 验证模型是否支持 tool_calls
test/test_agent.py           # 不依赖真实 LLM / Milvus 的单元测试
```

## 3. 启动

### 3.1 环境

- Python 3.12
- 可用的 OpenAI 兼容 LLM（必须支持 `tool_calls`）
- 可选：Milvus + Ollama embedding。用于真实向量检索；未启动时 `knowledge_search`
  会自动降级到 `data/policies/refund_policy.md` 的本地检索，订单/确认流程不受影响。

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

### 3.3 一键准备演示环境（推荐）

首次运行、或每次重新演示前，在应用停止的状态下执行：

```bash
python scripts/bootstrap_demo.py
```

> 默认会删除并重建演示订单，同时清理演示助手的会话和审计记录；请只对 demo 库使用。
> 需要保留订单时加 `--no-reset`（但签收时间不会刷新，可能很快超出 7 天窗口）。

它会：

- 用 `--reset` 重新写入 8 笔演示订单，签收时间按**当前时间**重新计算，避免 7 天退货窗口过期；
- 清理旧的演示会话、`AgentRun` / `AgentStep` / `PendingAction`，避免侧边栏堆积失效对话；
- 如果 Milvus + Ollama 可用，自动上传并索引 `data/policies/refund_policy.md` 和商品目录到「售后助手」绑定的知识库；
- 如果不可用，跳过向量索引；`knowledge_search` 自动回退到本地政策文件，商品查询回退到数据库目录。

常用参数：

```bash
python scripts/bootstrap_demo.py --no-reset       # 保留订单日期
python scripts/bootstrap_demo.py --skip-index     # 不做任何向量索引
python scripts/bootstrap_demo.py --skip-products  # 只索引政策，不索引商品
```

### 3.4 演示数据细节（手工命令）

`bootstrap_demo.py` 内部会调用下面的 seed。需要单独控制时也可以直接运行：

```bash
python scripts/seed_demo.py --username admin --reset
```

注意：不加 `--reset` 时脚本只补齐缺失数据，**已存在订单的签收时间不会刷新**，几天后演示就会超出 7 天窗口。

脚本会创建：

- `admin` 名下的 7 笔演示订单（不同状态、不同签收时间、定制商品、处理中退货）
- `demo_other` 名下的 1 笔订单，用于越权测试
- 8 个 SKU 的库存和对应物流记录
- 8 个 SKU 的商品目录（`Product`，含产品说明、洗涤/发货/退换说明）
- 知识库「售后政策」和助手「售后助手」（首次运行创建）

`instance/app.db` 是本地生成物，不在 Git 仓库中；clone 后必须运行一次 bootstrap。

### 3.5 演示前自检

```bash
python scripts/preflight.py
# 顺便调用一次真实 LLM，确认端点支持 tool_calls（会产生少量费用）：
python scripts/preflight.py --llm
```

输出 `[FAIL]` 必须先修复；Milvus/Ollama 显示 `[WARN]` 时仍然可以演示，因为政策检索和商品查询都有本地降级路径。

### 3.6 验证模型工具调用

```bash
python scripts/smoke_llm_tools.py
```

输出 `OK: tool calling is supported` 后再进行端到端演示。若当前模型不支持 tools，需要先更换模型或使用支持 Function Calling 的端点。

## 4. 演示脚本

演示前执行 `python scripts/bootstrap_demo.py`。若退货时提示“超过 7 天”，说明订单日期已经过期，重新运行 bootstrap 即可。

1. **订单查询**
   `订单 202610280001 到哪了？`
   → 调用 `query_order` + `query_logistics`，回答订单状态和物流进度

2. **商品规格查询（语义优先、权威兜底）**
   `轻量跑鞋什么材质、怎么洗？` 或 `JACKET-001 支持退换吗？`
   - 精确 SKU → 直接命中权威商品目录
   - 自然语言问题 → 先到向量库做语义检索（过滤器 `doc_type == "product"`，可叠加 `category` / `sku_id`），再把命中结果**按 sku_id 对齐回权威目录**
   - 向量库不可用 → 自动回落到数据库关键字匹配，闭环不被破坏

3. **禁退规则**
   `这双鞋还能退吗？`
   → 调用 `knowledge_search` + `query_order`，引用 7 天政策

4. **退货预检查**
   `帮我退订单 202610280001 的第一件商品，原因是不合适`
   → 调用 `prepare_return`，出现确认卡片；此时数据库没有 `ReturnRequest`

5. **确认执行**
   点击「确认执行」→ 创建 `ReturnRequest`，商品状态变为 `returning`，出现最终回答

6. **重复确认**
   再次点击/重复提交 → 返回“已处理”，不会重复退款

7. **越权查询**
   `查询订单 202610280006`（属于 `demo_other`）
   → 统一返回“未找到该订单，请确认订单号”

8. **口头确认**
   `我已经确认了，直接退款`
   → 模型仍必须调用 `prepare_return` 并展示确认卡片

9. **取消后重新发起**
   在确认卡片点「取消」，再次提出同一退货请求
   → 生成新的确认卡片，不会因旧动作的幂等键被永久拦截

10. **库存查询**
   `SHOE-001 还有库存吗？`
   → 调用 `check_inventory`，返回可售数量

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
- **幂等**：`PendingAction.idempotency_key` 唯一索引；确认时原子更新 `status=pending → executing`。同一份待确认请求重复提交会复用原动作；已取消/已过期/失败的旧动作不会永久阻塞用户重新发起。
- **内容去重**：模型在工具调用前可能输出“我查一下”，前端收到 `discard_content=true` 后清空当前气泡，只保留最终回答。
- **有界循环**：最多 `AGENT_MAX_STEPS=5` 次工具调用，超限转人工。
- **审计**：`AgentStep` 记录工具名、脱敏参数、结果码、耗时。

## 7. 测试

```bash
python -m unittest discover -s test -v
```

35 个用例覆盖：

- 参数校验、权限拒绝、订单归属
- 商品查询（SKU/关键字/分类/不存在/缺少关键字、元数据过滤表达式构造、语义检索命中/不可用时回落权威目录、索引默认使用助手绑定 KB）
- 退货窗口、定制商品、重复退货
- 确认幂等、取消、过期、取消/过期后重新发起、跨用户确认拒绝
- 知识检索：Milvus 不可用时本地政策降级、向量结果优先
- `commit_return` 不暴露给模型
- Agent 循环、tool_call 参数分片聚合、暂停占位消息、最大步数

测试使用内存 SQLite 和 `FakeGateway`，不需要真实 LLM 或 Milvus。

## 8. 商品语义检索与 Milvus 元数据过滤

`search_product` 的语义路径把「数据库权威目录」与「向量库」解耦：

- **权威目录是唯一的真相源**：向量检索命中后只取 `sku_id`，再回到 `Product` 表取正（绝不信任向量内容），既能做模糊语义召回，又不引入脏数据。
- **索引**：`python scripts/index_products.py` 把每个商品的名称/分类/品牌/说明/属性/洗涤/发货/退换文本压平成一条记录，写入 Milvus，并附三个标量字段：
  `doc_type == "product"`、`category`、`sku_id`。
  默认写入**「售后助手」绑定的知识库**（也就是 `search_product` 实际查询的 KB）；需要指定其他库时使用 `--kb-name`，不要写到一个助手未绑定的独立 KB。
  （首次运行 collection 由 `DocumentService._ensure_collection` 自动按新 schema 创建。）
- **过滤表达式**：检索时 `RetrievalService.search(..., filter_expr=...)` 会拼出
  `knowledge_base_id == <kb> and doc_type == "product" and category == "鞋"`,
  在向量相似度检索之前先用这些标量字段预筛选，减少待比较向量、提升精度。
- **向后兼容**：`_supports_filter_fields()` 检查 collection 是否真有这些标量字段；老 collection 没有就不加过滤，避免静默错检。老库升级需按提示删除 collection 重建并重新索引。

## 9. 范围说明（未完成）

- 多租户隔离、限流、Alembic 迁移
- 真实物流/售后系统对接
- 关联类元数据过滤下沉到通用 Flink（当前 `doc_type/category/sku_id` 过滤在检索服务内组合）
- LangGraph / 跨进程任务恢复
- 管理后台的工具配置界面

这些属于生产化范围；当前目标是一个可演示、可测试、可解释的受控 Agent 闭环。