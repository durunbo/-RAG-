# 企业内部制度问答 RAG 系统 项目说明

这份文档讲系统做什么、怎么跑、谁和谁通信。读者是没做过 AI 的后端、产品、测试。

---

# 第一部分：项目概述

## 1. 项目背景与痛点

公司制度散在几十个 Word 和 PDF 里。员工查"工牌丢了去哪补""私用公司快递罚多少"，只能问人或者自己翻。原来的全文搜索按字面匹配：制度原文写"严禁使用公司快递服务邮寄个人物品"，员工问"拿公司快递寄东西会怎样"，字面几乎不重合，搜不到。

**RAG（检索增强生成）**：先从文档里找出相关段落，再让模型照着这几段回答，不让它凭记忆写。

## 2. 项目目标

1. 把制度文档变成可检索的知识。上传 PDF/Word，自动解析、切块、转向量、入库。
2. 用员工自己的话提问，答案里的数字、房间号、条款号从原文取。

不做：通用聊天、制度以外的问答、替人下判断（"我这种情况算不算旷工"只给条款，不给结论）。

## 3. 适用场景与使用者

**适用**：制度查询、人事行政流程咨询、处罚标准确认。问题有明确出处，库里有原文。

**不适用**：跨文档推理、需要外部信息的（法规、行业标准）、库里没有依据的开放问答。

| 使用者 | 怎么用 |
|---|---|
| 普通员工 | 网页聊天框提问，看答案和引用来源 |
| 行政 / 人事 | 上传新制度、删过期文档 |
| 开发 / 测试 | 调 HTTP 接口，做集成和回归 |
| 运维 | 看健康检查、指标、日志 |

## 4. 项目核心能力总览

- 文档上传：PDF / DOCX，单文件 20 MB 上限，后台异步处理
- 解析、清洗、分类、分块、向量化、入库
- 语义检索：按意思找段落，不按字面
- 两种回答方式：页面走 Dify，直连接口走两阶段（模型出 JSON，程序核验后套模板）
- 引用溯源：答案标出来自哪个文件哪一节
- 多轮会话：链路 B 每次取最近 3 条消息
- 检索接口对外开放，兼容 Dify 外部知识库协议
- 健康检查、运行指标、请求日志

---

# 第二部分：总体方案设计

## 1. 总体架构图说明

**链路**：一次提问从发出到拿到答案，中间经过哪些程序。同一句话可以走两条路，写答案的不是同一个东西。

| | 链路 A（页面） | 链路 B（直连接口） |
|---|---|---|
| 入口 | `http://127.0.0.1:5500` | `POST /api/v1/chat` |
| 谁用 | 员工日常提问 | 开发、测试 |
| 谁写答案 | Dify 里的 qwen2.5 | 本项目程序 |
| 本项目负责 | 只找段落 | 找段落 + 抽条款 + 核验数字 + 排版 |
| 线上真实调用 | 有 | 无 |

```
链路 A：浏览器 → dev-proxy(5500) → Dify(80) → POST /retrieval 到 FastAPI(8000)
        → bge-m3 编码 → pgvector 取前 8 条 → 回 Dify → qwen2.5 写答案 → 浏览器

链路 B：调用方 → POST /api/v1/chat → 改写问题 → 检索 → 候选加减分
        → 模型输出 JSON → 程序逐字核验 → 套模板渲染 → 落库 → 返回
```

dev-proxy 是开发用的转发脚本，托管静态页，转发时给 Dify 加上 API Key。链路 A 上本项目只做一件事：把问题编码，取 8 段原文返回。写答案、标出处由 Dify 和 qwen2.5 完成。

链路 B 的代码完整，但页面不走它。查 2026-10-30 的 8000 端口日志：`/retrieval` 被 Dify 容器调了 8 次，都是真实问题；`/api/v1/chat` 只有本机自己测的 2 次。所以 `rag_service.py` 里的两阶段逻辑、四段提示词、逐字核验、模板渲染，目前一次没在生产跑过。

两条路答案不一样是正常的，写答案的不是同一个东西，对比测试时别混着比。

## 2. 整体业务流程

以「我上班玩电脑扣钱吗」为例，走链路 A。

| 步 | 谁 | 做什么 | 代码 |
|---|---|---|---|
| 1 | 浏览器 | 把这句话发给 5500 | —— |
| 2 | dev-proxy | 去掉 `/dify` 前缀，加 Dify 的 Key，转发到 `127.0.0.1/v1/chat-messages` | `dev-proxy.py` 143–158 |
| 3 | Dify | 检索节点按知识库设置发 `POST /retrieval`，带 `knowledge_id`、`query`、`top_k=8`、`score_threshold=0` | —— |
| 4 | FastAPI | 比对 Key 和 `DIFY_RETRIEVAL_API_KEY`，不匹配返回 401 | `dify_retrieval.py:99` |
| 5 | FastAPI | 校验参数；`query` 为空串返回 `{"records": []}`（Dify 点"测试连接"就发空串） | 第 104–126 行 |
| 6 | FastAPI | 编码成 1024 个数，pgvector 取前 8 条 | 第 132 行 |
| 7 | FastAPI | `score_threshold` 是 0，判定"不过滤"，8 条全留，打包返回 | 第 137–181 行 |
| 8 | Dify | 8 段塞进 472 字提示词，交给 qwen2.5 | —— |
| 9 | qwen2.5 | 读完判断有没有能答的条款，没有就说固定话术 | —— |

整条链路实测 2 秒左右，检索约 0.1 秒，其余是模型生成。

**代码怎么判断"查得到"？不判断。** 第 6、7 步就是全部：编码、取前 8、原样返回，不管像不像。库里 191 个块，永远凑得齐 8 条，所以链路 A 上"检索返回空"几乎不会发生，除非服务挂了。

判断由模型做，在第 9 步：8 段原文进提示词，qwen2.5 自己读，没一条对得上就说"知识库中未收录"。

**为什么不用分数当裁判**：实测两个区间重叠。该答得出的最低 0.5618，不该答的（停车场、年假天数、出差补贴）最高 0.6918。定 0.6 会误杀真问题，定 0.7 会放过假问题。

**链路 B 查不到时**：模型先抽条款输出 JSON，程序拿 `evidence` 回原文逐字找，找不到就丢。全丢光触发改写重查一次，还空就返回常量那句话，不让模型参与。

**什么时候带上文去检索**：

- 链路 A 永远不带。Dify 检索节点的 query 变量设的是 `sys.query`，就是当前这一句。页面上追问"那第二次呢"，这句话原样查库，容易查偏。
- 链路 B 有条件带。取最近 3 条消息（`context_window = 3`），且两个条件同时满足才拼上一条用户问题：当前问题 ≤ 15 字（`_FOLLOWUP_MAX_LEN`），且含 12 个承接词之一——那、这、它、他、她、呢、还、再、刚才、上面、上述、前面。

实测：`那第二次呢` 单独查 top1=0.6016，命中无关章节标题；拼成 `我上班打游戏扣多少钱 那第二次呢` 后 top1=0.6881，命中"严禁上班打游戏…乐捐200"。

这套规则有缺陷：承接词只做字面包含判断。"这周能请假吗"含"这"，会被判成追问并拼上文，其实它自己说得清。

## 3. 技术选型原则

"本地能跑、不花钱、中文效果够用"，没做过多方案对比。

| 组件 | 用它做什么 |
|---|---|
| FastAPI | 所有 HTTP 接口的框架，支持异步，自带 `/docs` 页面 |
| PostgreSQL + pgvector | 业务数据和向量放一个库，不用维护两套存储 |
| bge-m3 | 把文字转成 1024 维向量，本地跑，不调外部 API |
| qwen2.5:7b | 生成答案、抽条款，本地 7B，单卡能跑 |
| Ollama | 本地模型管理，对外提供 HTTP 推理接口 |
| Dify | 页面聊天的编排、会话管理、引用展示 |

**Chroma** 是一个独立的向量库。项目装过、`docker-compose.yml` 里挂过目录，代码零引用，实际存储只有 pgvector 一套。

## 4. 系统分层划分

| 层 | 组件 | 代码位置 | 链路 A | 链路 B |
|---|---|---|---|---|
| 接入层 | dev-proxy(5500)、Dify nginx(80) | `RAG-frontend/dev-proxy.py` | 转发 + 注入 Key | 不参与 |
| 应用服务层 | FastAPI(8000) | `app/main.py`、`app/api/routers/` | 只暴露 `/retrieval` | 全部接口 |
| 检索层 | `RAGService`、`VectorStore` | `app/services/rag_service.py`、`src/vector_store/` | 编码 + 取 8 条 | 编码 + 取 5 条 + 加减分 |
| 模型层 | Ollama(11434) | 外部进程 | 不参与，Dify 直连 | 抽取 JSON |
| 存储层 | PostgreSQL 18.6 + pgvector 0.8.6 | 本机 5432 | 只读 `chunks` | 读写全部表 |
| 运维层 | `/health/*`、`/metrics` | `app/api/routers/health.py` | 探活 | 探活 |

---

# 第三部分：核心功能模块说明

## 1. 对话交互

**输入**：一句话，1–2000 字。**输出**：答案 + 引用来源 + 会话号。

- 会话号由前端保存，第二轮起带上，Dify 据此取历史
- 页面单次请求等 60 秒，超时提示"请求超过 60 秒未响应，请稍后重试"
- 前端调 5500 端口的 `/dify/v1/chat-messages`，不直接打后端

## 2. 知识库检索问答

**输入**：问题文本 + 取几条。**输出**：若干段原文和相似度分数。

**Embedding（把一段文字压成一串数字，数字接近代表意思接近）**：bge-m3 把问题和每段文字都压成 1024 个数字，算余弦距离，越近排越前。

**向量库**：专存这种数字串、能按接近程度快速查的数据库，本项目用 PostgreSQL 的 pgvector 扩展。

**分块（chunk）**：把长文档切成小块，每块单独存、单独检索。库现状：191 个块、35,220 字，来自 5 份文档。

**召回（从库里找出候选段落这一步）**：按向量距离取前 N 条。**重排（对召回结果再排序）**：本项目没有装重排模型，只按章节加减分——同一章节加 0.04~0.05，跨文档扣分。

已知短板：108 个块（57%）不足 150 字，碎片偏多。18 个块装了 8 条以上处罚条款，最密的 16 条。这种块里问其中一条，整块不突出，容易排到后面。

## 3. 文档接入与处理

**输入**：`.pdf` 或 `.docx`，最大 20 MB。**输出**：任务号，结果写进 `chunks` 表。

上传立即返回 202，字段 `task_id`、`status`、`stage`、`source_file`、`poll_url`。之后轮询 `GET /api/v1/documents/{task_id}` 拿结果。

- `status` 五个值：`processing`、`succeeded`、`needs_review`、`skipped`、`failed`
- `stage` 七个值：`queued`、`parsing`、`cleaning`、`classifying`、`chunking`、`indexing`、`finished`

任务记录只在进程内存里，保留 60 分钟，服务重启就没了。已入库的数据不受影响。

## 4. 知识库管理

能做：

- 查任务状态（`GET /api/v1/documents/{task_id}`）
- 删文档（`DELETE /api/v1/documents/{source_file}`），按文件名删掉该文件所有块
- 重建：重传同一文件，旧块先删后插

不能做：

- 不能建多个知识库，只有一个库
- 不能改单个块，只能删了重传
- 没有增量同步，改了源文件要整个重传
- 没有文档级权限，能调接口就能删，只校验是不是管理员
- 没有"列出所有文档"的接口。要看库里有哪几份文件，直接查 `select distinct source_file from chunks`

## 5. 提示词与工作流编排

分两块。

**Dify 那边（页面在用）**：可视化编排，4 个节点——用户输入 → 知识检索 → LLM → 直接回复。提示词在 LLM 节点的 SYSTEM 框里，472 字，6 条规则。改完点"发布"生效，不用重启。数据库存两份：`draft` 草稿和已发布版本，线上跑的是已发布那份。

**代码这边（页面不调用）**：硬编码在 `app/services/rag_service.py`。

| 常量 | 行号 | 作用 |
|---|---|---|
| `_EXTRACTION_SYSTEM_PROMPT` | 120–178 | 抽取规则，8 条 |
| `_EXTRACTION_SCHEMA` | 81–117 | 要求模型输出的 JSON 结构 |
| `_ALIGN_SYSTEM_PROMPT` | 197–214 | 改写问题用的提示词 |
| `_ALIGN_SCHEMA` | 216–220 | 改写的输出结构 |

没有热加载、没有版本记录、没有 A/B。改一次要重启服务。

## 6. 输出能力

**引用溯源**：答案下面列来源，格式是文件名 + 章节名。页面上的引用由 Dify 生成，链路 B 由程序从 `sources` 字段渲染。

**格式化**：链路 B 走模板，不让模型自由写。真实输出形如：

```
一、规定明确的部分：
- 原文：「工牌遗失需到 406 室补办，工本费纸 5 元、绳 5 元、壳 10 元。」[文档1]
```

**拒答两套话术，不是同一句**：

| 链路 | 说什么 | 谁产生的 | 是否硬拦截 |
|---|---|---|---|
| A（页面） | 知识库中未收录相关规定，建议联系行政人事部门确认。 | qwen2.5 照提示词第 1 条复述 | 否 |
| B（直连） | 根据现有资料无法回答。 | 代码常量 `_NO_CONTEXT_ANSWER`（`rag_service.py:45`） | 是，模型不参与 |

页面那句不是硬拦截。检索服务挂掉、返回空片段时，模型也这么说。**"库里真没有"和"检索断了"在页面上长得一样**，只能翻后端日志分辨。

## 7. 权限与用户管理

只有接口级鉴权，没有用户体系。

- 鉴权方式：`Authorization: Bearer <token>` 精确匹配。token 和 user_id 的对应写在 `.env` 的 `RAG_MOCK_TOKENS`，格式 `token1:user_001,token2:user_002`
- 模拟实现：`auth_provider` 只有 `mock` 一个取值，只比字符串，不解析 JWT、不验签、不查过期。生产环境仍配 `mock` 会启动失败
- 没有角色字段。管理员是 `RAG_ADMIN_USER_IDS` 里列的 user_id，命中白名单才能删文档，`/auth/me` 返回 `is_admin` 给前端
- 没有登录、注册、密码、token 过期

免鉴权的只有 `/health`、`/health/*`、`/docs`、`/redoc`、`/openapi.json`、`/retrieval`（它自己校验 `DIFY_RETRIEVAL_API_KEY`）。其余全部要带 token，**`/metrics` 也要**。

## 8. 接口开放能力

13 个路径，按谁来调分三组，因为三方带的凭证不同：

- 页面调 `/api/v1/*`，带用户 token
- Dify 调 `/retrieval`，带独立密钥 `dify-rag-key`，请求体是 Dify 规定的格式
- 监控调 `/health/*` 和 `/metrics`

---

# 第四部分：组件清单与组件详细说明

## 1. 后端主服务（FastAPI）

- **输入**：HTTP 请求。**输出**：HTTP 响应
- **负责**：路由分发、参数校验、鉴权、检索编排、模型调用、结果渲染、落库
- **通信**：对上收 HTTP；对下用 `asyncpg` 连 PostgreSQL，用 `httpx` 调 Ollama
- **代码量**：`app/` 6982 行，`rag_service.py` 单文件 1829 行
- **启动**：`uvicorn app.main:app --host 0.0.0.0 --port 8000`

## 2. 大模型推理服务（Ollama）

- **输入**：prompt。**输出**：生成的文本或 JSON
- **版本**：本机安装，监听 11434，已装 `qwen2.5:7b`（约 4.7 GB，Q4 量化）
- **通信**：`POST http://127.0.0.1:11434/api/chat`
- **注意**：Dify 也直连它，两个调用方共用一个模型实例，同时请求会排队

## 3. 向量数据库（pgvector）

- **输入**：1024 维向量。**输出**：按余弦距离排序的片段
- **形态**：PostgreSQL 的扩展，挂在库里，不是独立服务（0.8.6）
- **负责**：`chunks.embedding vector(1024)` 这一列，用 `<=>` 算距离，HNSW 索引
- **为什么没用 Chroma**：装过、配置过，代码零引用。一套存储够用，两套要保证同步

## 4. 关系型数据库（PostgreSQL）

- **输入**：SQL。**输出**：行
- **版本**：本机 18.6。`docker-compose.yml` 里写的是 pg16 镜像，没启用
- **内容**：四张表 `chunks`、`chunks_split`、`sessions`、`messages`，见第六部分
- **通信**：`postgresql://rag_user@localhost:5432/rag_db`
- **建表脚本**：`scripts/init-db.sql`

## 5. 文本处理 & 切分组件

- **输入**：`DocumentBlock` 列表。**输出**：`chunk` 列表
- **位置**：`src/chunker.py`、`src/chunk_strategies.py`、`src/cleaner.py`
- **负责**：清洗（去零宽字符、统一全半角、合并多余空行）；按文档类型选切法
- **四种切法**：FAQ 型（一问一答一块）、制度条款型（按条款切）、流程型（按步骤切）、表单型（按表项切）

## 6. Embedding 向量化模型

- **输入**：文字。**输出**：1024 维向量
- **位置**：`D:\AI\Models\bge-m3`，进程内加载，不走网络
- **负责**：问题编码和文档块编码，两者必须用同一个模型，否则距离没意义
- **注意**：启动时一次性加载，占内存约 2 GB。目录不存在服务起不来

## 7. 文档解析组件

- **输入**：PDF / Word 文件。**输出**：带类型的文本块
- **位置**：`src/loaders/pdf_loader.py`、`src/loaders/word_loader.py`
- **PDF**：PyMuPDF 按页取文本，识别页眉页脚，产出 `heading` 和 `paragraph`，**不产表格块**
- **Word**：python-docx 按段落和表格取，多一类 `table`

## 8. 缓存组件

**没有。** 没有 Redis，没有进程内缓存。每次提问都重新编码、重新查库。

现在库里 191 块，检索本身只要 0.1 秒，瓶颈不在缓存。

## 9. 网关 / 鉴权组件

- **dev-proxy**（`RAG-frontend/dev-proxy.py`，5500）：托管静态页，把 `/dify/*` 转发到 Dify，转发时注入 API Key。开发用，不是生产网关
- **鉴权中间件**（`app/middleware/auth.py`）：校验 Bearer token，解析出 user_id 挂到请求上下文。免鉴权白名单在这个文件里

## 10. 前端交互组件

- **位置**：`D:\python-learn\RAG-frontend`，纯静态，无框架
- **文件**：`index.html` + `js/`（config、api、app、markdown、mock）+ `css/styles.css`
- **负责**：渲染聊天界面、保存会话号、发请求、按 Markdown 渲染答案、显示引用来源
- **通信**：`fetch` 打 5500 的 `/dify/v1/chat-messages`
- **视图**：5 个——聊天、会话、健康、指标、调试

## 11. 容器编排组件（Docker）

- **作用**：跑 Dify 全家桶
- **范围**：只有 Dify 在容器里，FastAPI、Ollama、PostgreSQL 都是本机进程
- **当前容器**：`docker-api-1`、`docker-worker-1`、`docker-db_postgres-1`、`docker-nginx-1`、`docker-ssrf_proxy-1`
- **已知异常**：`docker-agent_backend-1`、`docker-local_sandbox-1` 一直 Restarting，不影响主链路

## 12. 第三方依赖组件（Dify）

- **作用**：页面聊天的编排层和门面
- **版本**：Docker 自托管，本机 80 端口，应用`企业制度问答助手`（advanced-chat）
- **工作流**：4 个节点，用户输入 → 知识检索 → LLM → 直接回复
- **知识来源**：外部知识库，指向 `http://host.docker.internal:8000`，密钥 `dify-rag-key`
- **模型**：qwen2.5:7b，温度 0.2
- **取几条**：由知识库自己的设置决定（`datasets.retrieval_model.top_k = 8`），工作流节点上那个 top_k 输入框不生效
- **注意**：Dify 里有两个知识库都叫「公司制度库」，`top_k=5` 那个绑在闲置应用「规章制度ai」上，本应用绑的是 `990f9337…`（top_k=8）。改错不生效
- **坑**：检索失败时 Dify 不报错，返回空片段继续走，模型就答"未收录"

---

# 第五部分：关键流程详细拆解

## 1. 文档入库全流程

**上传 → 解析 → 清洗 → 分类 → 分块 → 向量化 → 入库**

| 步骤 | 输入 | 输出 | 关键动作 |
|---|---|---|---|
| 1. 落盘校验 | 上传的文件 | 临时文件 + 任务号 | 按 1 MB 一块读；超 20 MB 返回 413；先读前 1 KB 判文件头；扩展名只收 `.pdf`/`.docx` |
| 2. 解析 | 文件路径 | `DocumentBlock` 列表 | PDF 按页取文本，识别页眉页脚；Word 按段落和表格取 |
| 3. 清洗 | 块列表 | 干净块列表 | 去零宽字符、统一全半角、合并空行、去页眉页脚残留 |
| 4. 分类 | 干净块列表 | 文档类型 | 按特征打分，四类：FAQ / 制度 / 流程 / 表单。分不清标 `needs_review` |
| 5. 分块 | 块列表 + 类型 | `chunk` 列表 | 按类型选切法，每块带 `source_file`、`section`、`chunk_index` |
| 6. 向量化 | chunk 文本 | 1024 维向量 | bge-m3 批量编码 |
| 7. 入库 | chunk + 向量 | `chunks` 表 | 同一 `source_file` 先删后插 |

任一步抛异常，任务标 `failed`，记 `failed_stage` 和 `error_code`。已入库的块不回滚。

`chunks` 表只有主键一个唯一索引，`(source_file, chunk_index)` 没有唯一约束。重传如果不先清旧块，库里会有两份相同文本，检索时占满名额。

## 2. 异常请求处理流程

统一错误结构：

```json
{"error_code": "VALIDATION_ERROR", "message": "message 长度需在 1-2000 之间", "request_id": "..."}
```

| 场景 | HTTP | error_code |
|---|---|---|
| 参数不合法 | 400 | `VALIDATION_ERROR` |
| 没带 token / token 无效 | 401 | `UNAUTHORIZED` |
| 删文档但不是管理员 | 404 | `NOT_FOUND`（故意不暴露 403） |
| 资源不存在 | 404 | `NOT_FOUND` |
| 文件超 20 MB | 413 | `FILE_TOO_LARGE` |
| 并发名额抢不到 | 503 | `SERVICE_UNAVAILABLE` |
| 检索不可用 | 503 | `RETRIEVAL_UNAVAILABLE` |
| 生成失败 | 500 | `INTERNAL_ERROR` |

并发控制只有链路 B 有：同时处理上限 10（`_MAX_CONCURRENCY`），满了以后新请求最多等 0.1 秒（`_ACQUIRE_TIMEOUT_SECONDS`），等不到直接返回 503 并给 `Retry-After: 5`。

链路 A 没有限流。检索服务不可用时不报错，Dify 拿到空片段继续走，用户在页面上看到的是"知识库中未收录"。

---

# 第六部分：数据设计

## 1. 数据分类

| 类别 | 存在哪 | 内容 |
|---|---|---|
| 业务元数据 | `sessions` | 会话，按 user_id 归属 |
| 文档分片 | `chunks` 的文本列 | 原文、来源文件、章节、序号 |
| 向量数据 | `chunks.embedding` | 每块 1024 个浮点数 |
| 对话日志 | `messages` | 每轮的问题、答案、检索片段 |

`users` 和 `api_keys` 两张表已于 2026-10-30 删除。它们建于 10-25，当时方案是服务端自己发 API Key，后来改成身份由外部 token 决定，代码和建表脚本都清了，库里已建好的表没删。备份在 `D:\Temp\rag_backup\drop_users_api_keys_20261030.sql`。

## 2. 核心数据表说明

**chunks**（191 行）

字段：`id`(text, 主键)、`embedding`(vector 1024)、`text`、`source_file`、`heading_path`、`section`、`chunk_index`(int)、`token_count`(int)、`document_type`、`source_page`(int)、`question`、`stage`、`responsible_department`、`form_name`、`created_at`、`updated_at`。

- `chunk_index` 只在本文件内唯一，不同文件会重复，不能单独当 ID
- `document_type` 四个值：`faq`、`policy`、`process`、`form`
- `section` 空值存空字符串，不是 NULL

**sessions**（235 行）

字段：`session_id`(主键，`sess_` + 12 位十六进制)、`user_id`、`created_at`、`updated_at`、`metadata_json`、`deleted_at`。

没有 title 字段。会话标题是查询时现算的：`session_manager.py:110` 用子查询取该会话第一条用户消息，`_make_session_title()` 截到 40 字，不落库。

**messages**（530 行）

字段：`message_id`(主键)、`session_id`、`role`（`user`/`assistant`）、`content`、`timestamp`、`retrieved_chunks`(jsonb)。

字段名是 `retrieved_chunks`，不是 `sources`。接口响应里那个 `sources` 是包装后的名字。

**chunks_split**（412 行）：拆块实验用的沙盒表，主流程不读。

## 3. 数据流转规则

```
上传 → chunks 表（文本 + 向量）
提问 → 编码 → 查 chunks → 返回片段 → messages 表（问题 + 答案 + 检索片段）
删文档 → 按 source_file 删 chunks，不删 messages
删会话 → 给 sessions.deleted_at 打时间戳，行还在
```

---

# 第七部分：部署方案

## 1. 部署架构

单机，四个本机进程 + 一组容器，没有多副本和负载均衡。

```
本机 Windows 11
├── FastAPI（8000）       ← Dify 调它检索
├── dev-proxy（5500）     ← 浏览器打开这个
├── Ollama（11434）       ← Dify 和 FastAPI 都调它
├── PostgreSQL（5432）    ← 带 pgvector
└── Docker
    └── Dify 全家桶（80 端口对外）
```

进程之间全靠本机回环。Dify 在容器里，通过 `host.docker.internal` 访问宿主机 8000。

FastAPI 不进容器，因为 bge-m3 要从本地路径加载模型，进容器得挂卷、装 torch。

8000 是手敲 `uvicorn app.main:app --host 0.0.0.0 --port 8000` 起来的，没有 `--reload`，改代码不重启不生效；也没有开机自启，机器重启后要手动拉起，否则 Dify 会静默返回空片段。

## 2. 环境依赖清单

| 依赖 | 要求 | 当前 |
|---|---|---|
| 操作系统 | Windows 10/11 | Windows 11 |
| Python | 3.10+ | 3.11（`.venv`） |
| PostgreSQL | 14+，需装 pgvector | 18.6 + pgvector 0.8.6 |
| Ollama | 任意近版 | 本机安装 |
| 大模型 | qwen2.5:7b | 4.7 GB，Q4 |
| Embedding 模型 | bge-m3 | `D:\AI\Models\bge-m3` |
| Docker | 需 WSL2 后端 | Docker Desktop |
| 显卡 | 6 GB 显存以上 | 单卡 |

Python 包（`requirements.txt`，未锁版本）：fastapi、uvicorn、pydantic、pydantic-settings、asyncpg、httpx、PyMuPDF、python-docx、transformers、torch、numpy、pytest。

---

# 第八部分：接口说明

## 1. 接口总览

13 个路径、14 个方法。

| 方法 | 路径 | 作用 | 鉴权 |
|---|---|---|---|
| POST | `/api/v1/chat` | 问答（链路 B） | Bearer |
| POST | `/api/v1/retrieval/search` | 只检索不生成 | Bearer |
| GET | `/api/v1/conversations` | 列会话 | Bearer |
| POST | `/api/v1/conversations` | 建会话 | Bearer |
| GET | `/api/v1/conversations/{session_id}/messages` | 列消息 | Bearer |
| POST | `/api/v1/conversations/{session_id}/delete` | 删会话（软删） | Bearer |
| POST | `/api/v1/documents` | 上传文档 | Bearer |
| GET | `/api/v1/documents/{task_id}` | 查任务状态 | Bearer |
| DELETE | `/api/v1/documents/{source_file}` | 删文档 | Bearer + 管理员 |
| GET | `/api/v1/auth/me` | 查当前用户 | Bearer |
| POST | `/retrieval` | Dify 外部知识库协议 | 独立密钥 |
| GET | `/health/live` | 进程活着吗 | 免 |
| GET | `/health/ready` | 数据库连得上吗 | 免 |
| GET | `/metrics` | 运行指标 | Bearer |
| GET | `/docs`、`/openapi.json` | 接口文档 | 免 |

`/health` 这个路径不存在，只有 `/health/live` 和 `/health/ready`。浏览器访问 `/health` 看到 Not Found，不代表服务挂了。

## 2. 核心入参出参说明

**POST /api/v1/chat** 请求：

```json
{
  "message": "工牌丢了怎么补办",
  "session_id": "sess_a1b2c3d4e5f6"
}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `message` | 是 | 问题，1–2000 字。字段名是 message，不是 query |
| `session_id` | 否 | 不带就新建会话 |

`top_k` 已从请求体删除。传了不报错，会被静默丢掉。检索条数固定取 `max_sources = 5`。

响应：

```json
{
  "session_id": "sess_a1b2c3d4e5f6",
  "answer": "一、规定明确的部分：\n- 原文：「…」[文档1]",
  "sources": [
    {"source": {"source_file": "某某公司管理制度.pdf", "section": "工牌管理", "chunk_index": 12}, "score": 0.71}
  ],
  "message_count": 4,
  "persisted": true,
  "request_id": "req_xxx"
}
```

没有 `message_id`、`retrieval_ms`、`generation_ms` 这三个字段。

**POST /retrieval**（Dify 用）请求：

```json
{
  "knowledge_id": "default",
  "query": "私自更换门锁",
  "retrieval_setting": {"top_k": 8, "score_threshold": 0.0}
}
```

响应：

```json
{
  "records": [
    {"content": "…原文…", "score": 0.72, "title": "某某公司管理制度.pdf", "metadata": {"section": "…"}}
  ]
}
```

`score_threshold` 传 0 或 null 时不过滤。`query` 为空的返回 `{"records": []}`。`top_k` 限制在 1–20。

**GET /metrics** 返回 9 个字段：`in_flight`（此刻正在处理的请求数）、`peak_in_flight`、`total_requests`、`failed_requests`、`busy_rejections`、`avg_latency_ms`、`max_latency_ms`、`avg_retrieval_ms`、`avg_generation_ms`。计数在进程内存里，服务重启归零。

## 3. 调用示例

```bash
# 健康检查
curl --noproxy '*' http://127.0.0.1:8000/health/ready

# 问答（首轮不带 session_id）
curl --noproxy '*' -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Authorization: Bearer test-token-user-001" \
  -H "Content-Type: application/json" \
  -d '{"message":"工牌丢了怎么补办"}'

# 模拟 Dify 调检索（排查页面为什么不答）
curl --noproxy '*' -X POST http://127.0.0.1:8000/retrieval \
  -H "Authorization: Bearer dify-rag-key" \
  -H "Content-Type: application/json" \
  -d '{"knowledge_id":"default","query":"会议室怎么预订","retrieval_setting":{"top_k":2,"score_threshold":0}}'
```

Windows 上必须加 `--noproxy '*'`。本机设了 `http_proxy=127.0.0.1:64026`，不加会走代理，回环请求被转出去，报假的 502。

---

# 第九部分：附录

## 1. 术语表

| 术语 | 一句话解释 |
|---|---|
| RAG | 检索增强生成。先从文档里找相关段落，再让模型照着这几段回答 |
| Embedding | 把文字压成一串固定长度的数字，本项目 1024 个。意思越近，两串数字越近 |
| 向量库 | 专存这种数字串、能按接近程度查的数据库 |
| pgvector | PostgreSQL 的扩展，让数据库能存向量、算余弦距离 |
| 分块 | 把长文档切成小块，每块单独存、单独检索 |
| 召回 | 从库里找出候选段落这一步 |
| 重排 | 对召回结果再排序。本项目没有模型重排，只按章节加减分 |
| Ollama | 本地跑大模型的工具，对外提供 HTTP 推理接口 |
| Chroma | 一个独立的向量库。本项目装过、配置过，代码里没用 |
| bge-m3 | 把文字转向量的模型，本项目用它编码 |
| qwen2.5:7b | 通义千问 7B，本项目用它生成答案、抽条款 |
| Dify | 开源的 LLM 应用编排平台。页面聊天走它 |
| 温度 | 控制模型发挥程度的参数，越低越死板。Dify 那边定 0.2 |
| top_k | 取几条。Dify 取 8 条，链路 B 固定 5 条 |

## 2. 缩略语

| 缩写 | 全称 | 中文 |
|---|---|---|
| RAG | Retrieval-Augmented Generation | 检索增强生成 |
| LLM | Large Language Model | 大语言模型 |
| API | Application Programming Interface | 应用程序接口 |
| JSON | JavaScript Object Notation | 一种数据格式 |
| SSRF | Server-Side Request Forgery | 服务端请求伪造，Dify 有防护，默认不让访问内网地址 |
| HNSW | Hierarchical Navigable Small World | 一种近似最近邻索引算法 |

## 3. 参考资料

**代码位置**

- 后端：`D:\python-learn\RAG-learn`，入口 `app/main.py`
- 前端：`D:\python-learn\RAG-frontend`，入口 `index.html`
- 建表：`scripts/init-db.sql`
- 测试：`tests/`，155 个用例

**运行时地址**

- 接口文档：`http://127.0.0.1:8000/docs`
- 页面：`http://127.0.0.1:5500`
- Dify 控制台：`http://127.0.0.1`

**外部文档**

- pgvector：https://github.com/pgvector/pgvector
- Ollama：https://ollama.com/
- bge-m3：https://huggingface.co/BAAI/bge-m3
- Dify 外部知识库协议：https://docs.dify.ai/zh/self-host/use-dify/knowledge/connect-external-knowledge-base
- FastAPI：https://fastapi.tiangolo.com/

**配置备份位置**

- Dify 工作流和知识库：`D:\Temp\dify_backup\`
- 已删的两张表：`D:\Temp\rag_backup\`
