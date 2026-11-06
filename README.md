# 售后智能服务 Agent

基于现有 RAG 平台实现的电子产品受控退货申请 Agent。用户可以在同一个对话中完成：

> 问政策 → 查订单/物流/商品 → 确定性资格判断 → 预览并确认 → 提交退货申请

产品边界是“退货申请受理”，暂不包含换货、退款审批、补偿、修改订单和库存查询。模型负责理解、查询和解释，硬性资格由服务端规则引擎判断，写操作由确认接口执行。

详细说明：[产品范围](docs/PRODUCT_SCOPE.md) · [系统架构](docs/ARCHITECTURE.md) · [安全模型](docs/SECURITY_MODEL.md) · [评测方案](docs/EVALUATION.md) · [待补输入](docs/REQUIRED_INPUTS.md)

## 1. 已实现的能力

- `knowledge_search`：包装现有 `RetrievalService`，返回政策片段和引用来源；Milvus/Ollama 不可用时降级到本地 `refund_policy.md` 和 `electronics_after_sales_faq.md`
- `query_order` / `query_logistics`：只读业务工具，全部做订单归属校验
- `search_product`：查询商品目录（产品说明、处理器/存储等属性、使用保养、发货与退换说明），只读
- `ReturnPolicyEngine`：按政策版本、生效日期、商品类别、签收时间和商品状态做确定性资格判断；不依赖模型解释硬规则
- `prepare_return`：资格预检查 + 预计退款金额计算，只创建 `PendingAction`，不修改业务数据
- 确认/取消接口：动作绑定的 HMAC 确认令牌、确认参数绑定、提交前重新校验、原子状态流转和幂等键
- 提交前状态复核：订单、商品状态、金额或规则快照发生变化时，旧确认自动失效
- `AgentRun.state_json`：SSE 暂停后从数据库恢复对话，确认后再生成最终回答
- SSE 事件：`tool_start`、`tool_result`、`references`、`confirmation_required`、`delta`、`end`
- `AgentStep`：每次工具调用和最终写操作的审计记录
- 前端确认卡片：展示商品状态、政策版本、是否人工核验、预计退款和确认令牌

## 2. 目录

```text
app/
  agents/            # 上下文、消息聚合、提示词、有界循环、模型网关
  tools/             # 工具注册、执行器、知识/订单/物流/商品/售后工具
  services/
    agent_service.py         # run 状态、确认/取消编排
    after_sales_service.py   # 待确认动作、确认令牌、幂等提交
    return_policy.py         # 版本化规则选择与确定性资格判断
  models/
    commerce.py              # Order / OrderItem / Shipment / ReturnPolicy / ReturnRequest
    agent.py                 # AgentRun / AgentStep / PendingAction
  routes/agent.py            # stream / confirm / cancel / pending
app/templates/assistants/chat.html  # Agent 聊天页（新增确认卡片）
data/demo_seed.json          # 小批量演示数据
data/policies/refund_policy.md             # 退货硬规则
data/policies/electronics_after_sales_faq.md  # 激活、数据清理、配件等 FAQ
data/return_policies.json    # 可执行规则表，和 RAG 政策文本同步维护
data/eval/return_policy_cases.json  # 资格判断黄金用例
docs/                        # 范围、架构、安全与评测说明
scripts/bootstrap_demo.py   # 一键重置演示数据 + 可选索引
scripts/preflight.py         # 演示前自检（数据、窗口、LLM、Milvus）
scripts/seed_demo.py         # 写入演示数据
scripts/index_products.py    # 商品向量索引（默认写入助手绑定 KB）
scripts/smoke_llm_tools.py   # 验证模型是否支持 tool_calls
scripts/smoke_vector_search.py  # 验证商品语义检索和政策检索
scripts/smoke_agent_e2e.py   # 真实模型只读 Agent 冒烟
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
- 9 个电子产品的库存和对应物流记录
- 9 个 SKU 的商品目录（`Product`，含配置、使用/保养、发货与退换说明）
- `2026.11` 版本的可执行退货规则；RAG 文本与规则表需要同步维护
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
python scripts/smoke_vector_search.py
python scripts/smoke_agent_e2e.py
```

第一条脚本输出 `OK: tool calling is supported` 后再进行端到端演示；第二条脚本应命中 `HEADPHONE-001` 等电子商品和电子产品售后 FAQ；第三条会用真实模型执行一次只读商品查询。

## 4. 演示脚本

演示前执行 `python scripts/bootstrap_demo.py`。若退货时提示“超过 7 天”，说明订单日期已经过期，重新运行 bootstrap 即可。

1. **订单查询**
   `订单 202610280001 到哪了？`
   → 调用 `query_order` + `query_logistics`，回答订单状态和物流进度

2. **商品规格查询（语义优先、权威兜底）**
   `星云 Book 14 什么配置、怎么保养？` 或 `HEADPHONE-001 支持退换吗？`
   - 精确 SKU → 直接命中权威商品目录
   - 自然语言问题 → 先到向量库做语义检索（过滤器 `doc_type == "product"`，可叠加 `category` / `sku_id`），再把命中结果**按 sku_id 对齐回权威目录**
   - 向量库不可用 → 自动回落到数据库关键字匹配，闭环不被破坏

3. **禁退规则**
   `这台笔记本还能退吗？`
   → 调用 `knowledge_search` + `query_order`，引用 7 天政策

4. **补充商品状态**
   `我要退订单 202610280001 的第一件商品`
   → Agent 必须先询问是否拆封/使用，不能默认“未拆封”

5. **退货预检查**
   `商品未拆封，原因是不合适`
   → 调用 `prepare_return`，出现带政策版本和确认令牌的卡片；此时数据库没有 `ReturnRequest`

6. **确认提交**
   点击「确认执行」→ 服务端复核订单、商品状态、金额和规则快照，再创建 `ReturnRequest`；显示“申请已提交”，不承诺立即退款

7. **重复确认**
   再次点击/重复提交 → 返回同一申请结果，不会重复创建退货申请

8. **错误或缺失确认令牌**
   直接构造确认请求但不带令牌 → 返回 `CONFIRMATION_TOKEN_INVALID`，不产生任何写入

9. **越权查询**
   `查询订单 202610280006`（属于 `demo_other`）
   → 统一返回“未找到该订单，请确认订单号”

10. **口头确认**
   `我已经确认了，直接提交`
   → 模型仍必须调用 `prepare_return` 并展示确认卡片

11. **取消后重新发起**
   在确认卡片点「取消」，再次提出同一退货请求
   → 生成新的确认卡片，不会因旧动作的幂等键被永久拦截

12. **质量问题人工核验**
   `订单 202610280003 的定制移动硬盘到货破损`
   → 可以提交申请，但状态为 `manual_review`，不得回答成“已同意退款”

13. **范围外问题**
   `LAPTOP-001 还有库存吗？` 或 `帮我换货`
   → 明确说明当前仅支持退货申请，并引导人工处理

## 5. 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/assistants/<assistant_id>/agent/stream` | Agent SSE 对话 |
| POST | `/api/agent-actions/<action_id>/confirm` | 确认并提交退货申请；JSON 必须包含 `confirmation_token`，并受 CSRF 保护 |
| POST | `/api/agent-actions/<action_id>/cancel` | 取消写操作（CSRF 保护） |
| GET | `/api/agent-actions/pending?session_id=` | 恢复未处理的确认卡片 |

SSE 事件：

```text
start / status / references / tool_start / tool_result
confirmation_required / delta / reasoning / end / error
```

## 6. 关键设计

- **读写分离**：`prepare_return` 只创建 `PendingAction`；`commit_return` 不在工具注册表中，只能由确认接口调用。
- **规则判定**：资格由 `ReturnPolicyEngine` 判断；RAG 只提供解释依据。规则按版本、生效日期和品类选择。
- **确认绑定**：确认令牌使用 HMAC 绑定动作、用户、会话、参数哈希和过期时间，令牌原文不落库。
- **提交前复核**：确认时重新计算规则和金额；订单、商品状态或规则快照变化返回 `CONFIRMATION_STALE`。
- **身份可信**：`user_id` 来自服务端 `AgentContext`，模型只能提供订单号。
- **暂停/恢复**：`AgentRun.state_json` 保存完整 OpenAI 消息序列，确认后替换占位 tool 消息并继续生成回答。
- **幂等**：`PendingAction.idempotency_key` 唯一索引；确认时原子更新 `status=pending → executing`。成功后的重复确认返回同一结果；同一份待确认请求重复提交会复用原动作；已取消/已过期/失败的旧动作不会永久阻塞用户重新发起。
- **内容去重**：模型在工具调用前可能输出“我查一下”，前端收到 `discard_content=true` 后清空当前气泡，只保留最终回答。
- **有界循环**：最多 `AGENT_MAX_STEPS=5` 次工具调用，超限转人工。
- **审计**：`AgentStep` 记录工具名、脱敏参数、结果码、耗时。

## 7. 测试

```bash
python -m unittest discover -s test -v
```

47 个用例覆盖：

- 参数校验、权限拒绝、订单归属
- 商品查询（SKU/关键字/分类/不存在/缺少关键字、元数据过滤表达式构造、语义检索命中/不可用时回落权威目录、索引默认使用助手绑定 KB）
- 退货窗口、商品状态、无理由例外、质量问题人工核验、政策缺失
- 12 条规则黄金用例、政策文本/规则版本一致、政策版本与决策快照持久化
- 确认幂等、确认令牌校验、提交前状态复核、取消、过期、跨用户确认拒绝
- 知识检索：Milvus 不可用时本地政策降级、向量结果优先
- `commit_return` 不暴露给模型
- `check_inventory` 不再暴露给模型
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
  `knowledge_base_id == <kb> and doc_type == "product" and category == "笔记本"`,
  在向量相似度检索之前先用这些标量字段预筛选，减少待比较向量、提升精度。
- **向后兼容**：`_supports_filter_fields()` 检查 collection 是否真有这些标量字段；老 collection 没有就不加过滤，避免静默错检。老库升级需按提示删除 collection 重建并重新索引。

## 9. 范围说明（未完成）

- 多租户隔离、限流、Alembic 迁移
- 真实物流/售后系统对接
- 关联类元数据过滤下沉到通用 Flink（当前 `doc_type/category/sku_id` 过滤在检索服务内组合）
- LangGraph / 跨进程任务恢复
- 管理后台的工具配置界面

这些属于生产化范围；当前目标是一个可演示、可测试、可解释的受控 Agent 闭环。
