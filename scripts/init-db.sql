-- ==========================================
-- PostgreSQL + pgvector 数据库初始化脚本
-- 执行时机：数据目录为空时，由官方镜像 entrypoint 执行一次。
--          已有数据卷【不会】重跑本文件，存量库请用 scripts/migrations/ 下的迁移脚本。
--
-- 结构基准（三份真相，按优先级）：
--   1. sessions / messages  → src/session/models.py（ORM 定义，代码实际按它查询）
--   2. chunks               → 无 ORM 模型，基准是 vector_store.py 的 INSERT 语句
--                             + 现有生产库已跑通的 154 条数据
--   3. 索引与约束           → 只保留代码真正依赖的部分
-- ==========================================

-- 启用 pgvector 扩展（向量相似度检索）
CREATE EXTENSION IF NOT EXISTS vector;

-- ==========================================
-- 1. 向量块表 (chunks)
-- 存储文档切块及其向量表示
-- ==========================================
CREATE TABLE IF NOT EXISTS chunks (
    id                     TEXT           PRIMARY KEY,  -- 格式: source_file_chunk_index
    embedding              VECTOR(1024)   NOT NULL,     -- BGE-M3 输出维度，与向量库实际数据一致
    text                   TEXT           NOT NULL,     -- Chunk 正文内容
    source_file            TEXT           NOT NULL,     -- 原始文档文件名，定位切块归属
    heading_path           TEXT,                        -- 文档章节路径（"第一章 / 第一节"）
    chunk_index            INTEGER        NOT NULL,     -- 切块在文档中的序号
    token_count            INTEGER,                     -- Token 数量
    document_type          TEXT,                        -- 文档类型（制度/流程/FAQ等）
    source_page            INTEGER,                     -- 来源页码；允许 NULL，部分文档无页码
    section                TEXT,                        -- 所属章节
    question               TEXT,                        -- FAQ 问题（仅 FAQ 类型）
    stage                  TEXT,                        -- 流程阶段
    responsible_department TEXT,                        -- 责任部门
    form_name              TEXT,                        -- 关联表单名称
    created_at             TIMESTAMPTZ    DEFAULT now(),-- 入库时间
    updated_at             TIMESTAMPTZ    DEFAULT now() -- 最后更新时间
);

-- HNSW 索引：余弦相似度检索（适配 BGE-M3 的归一化向量）
-- m=16 / ef_construction=64 是 pgvector 默认值，显式写出便于日后调参
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- delete_by_source_file() 按此列过滤，数据量增长后避免全表扫描
CREATE INDEX IF NOT EXISTS idx_chunks_source_file ON chunks (source_file);

-- ==========================================
-- 2. 会话表 (sessions)
-- 对应 SessionModel；主键为应用侧生成的 sess_xxxxxxxxxxxx
--
-- 逻辑删除：删除会话只写 deleted_at，不删 messages。
--   deleted_at IS NULL  → 未删除（业务唯一可见状态）
--   deleted_at 非空     → 已删除，所有正常接口按「不存在」处理
-- 已删除会话的消息仍留在 messages 表里，将来若要做定期物理清理，
-- 应按「deleted_at 早于 N 天」为条件，而不是靠当前这种「删了就没了」的方式。
-- ==========================================
CREATE TABLE IF NOT EXISTS sessions (
    session_id    VARCHAR(64)  PRIMARY KEY,
    user_id       VARCHAR(128) NOT NULL,                -- 用户标识（支持多用户）
    created_at    TIMESTAMPTZ  DEFAULT now(),
    updated_at    TIMESTAMPTZ  DEFAULT now(),           -- 最后活跃时间，逻辑删除时不更新
    metadata_json JSONB,                                -- 扩展字段，无默认值，由应用写入
    deleted_at    TIMESTAMPTZ                           -- 逻辑删除时间；NULL = 未删除
);

CREATE INDEX IF NOT EXISTS ix_sessions_user_id ON sessions (user_id);

-- 列表查询专用：业务永远带 deleted_at IS NULL，已删除的行不进索引。
-- updated_at 不写 DESC —— B-tree 可反向扫描，ASC 索引同样服务 ORDER BY ... DESC。
CREATE INDEX IF NOT EXISTS idx_sessions_user_active
    ON sessions (user_id, updated_at)
    WHERE deleted_at IS NULL;

-- ==========================================
-- 3. 消息表 (messages)
-- 对应 MessageModel；一条消息属于一个会话，会话归属由应用层行锁保证
-- ==========================================
CREATE TABLE IF NOT EXISTS messages (
    message_id       VARCHAR(64) PRIMARY KEY,           -- 应用侧生成的 msg_xxxxxxxxxxxx
    session_id       VARCHAR(64) NOT NULL,              -- 与 sessions.session_id 同类型
    role             VARCHAR(32) NOT NULL,              -- user / assistant / system
    content          TEXT        NOT NULL,
    timestamp        TIMESTAMPTZ DEFAULT now(),         -- 消息时间，同一轮内助手消息 +1ms 保证排序
    retrieved_chunks JSONB                              -- 检索快照，仅 assistant 消息有值
);

CREATE INDEX IF NOT EXISTS ix_messages_session_id ON messages (session_id);

-- ==========================================
-- 4. 关于用户身份表：不建
--
-- 用户身份不再由本服务维护一本人名录。调用方在前端登录后拿到 token，
-- 请求时带上 Authorization: Bearer <token>，服务端验签后从 payload 取 user_id。
-- 因此这里既不需要 users 表，也不需要 api_keys 表：
--   - 人员增减、离职、换部门由公司登录系统负责，不会和本服务不同步；
--   - 没有「运维发 Key」这一层，也就不存在 Key 泄漏 / 吊销 / 重发的运维负担。
--
-- sessions.user_id 仍然保留，它存的是 token 里的用户标识，
-- 会话归属隔离靠它做，与身份从哪来无关。
-- ==========================================

-- ==========================================
-- 说明：以下旧对象已移除，理由是不在 ORM 定义中，
-- 保留会让「本脚本建的库」与「create_all 建的库」产生行为差异：
--   sessions.title                —— 代码从不读取，标题由首条 user 消息实时计算
--   sessions.metadata (旧名)      —— 实际列名是 metadata_json
--   messages.metadata (旧名)      —— 实际列名是 retrieved_chunks
--   messages.created_at (旧名)    —— 实际列名是 timestamp
--   messages 外键 + role CHECK    —— ORM 未声明；角色合法性由应用层校验
--   update_updated_at_column()    —— 应用在 3 处更新时均显式写时间戳
--   idx_chunks_created_at         —— 无任何查询按 created_at 过滤或排序
--   idx_sessions_updated_at       —— 同上
--   idx_messages_created_at       —— 同上
--   users / api_keys              —— 见上：身份改由前端 token 提供
-- ==========================================

COMMENT ON TABLE chunks   IS '向量知识库：文档切块及其 1024 维向量（BGE-M3）';
COMMENT ON TABLE sessions IS '会话管理：会话元数据';
COMMENT ON TABLE messages IS '消息记录：对话历史与检索快照';
