-- ==========================================
-- 迁移 003：sessions 增加逻辑删除标记
--
-- 适用对象：已经初始化过的数据库。
--          新环境不需要执行本脚本，init-db.sql 建出来的就是目标结构。
--
-- 背景：删除会话原本是物理 DELETE（先删 messages 再删 sessions），
--       用户点一下删除，对话内容就永久消失、无法追溯。
--       改成逻辑删除后只标记 deleted_at，消息全部保留在库里，
--       所有正常业务接口按 deleted_at IS NULL 过滤。
--
-- 与迁移 001 的部署顺序**相反**，务必注意：
--   001（改列类型）必须「先上线代码、再跑迁移」，否则旧代码写 str() 会 DataError；
--   003（加列）  必须「先跑迁移、再上线代码」，否则新代码查 deleted_at 会
--                UndefinedColumn —— 而这条查询在会话列表、消息历史、
--                聊天校验三条路径上都会走到，等于全站 500。
--   迁移完但代码未上线时是安全的：列全为 NULL，旧代码不认识它，行为零变化。
--
-- 本脚本在单个事务内执行，任一步失败会整体回滚。
-- 可重复执行（ADD COLUMN IF NOT EXISTS / CREATE INDEX IF NOT EXISTS）。
-- ==========================================

BEGIN;

-- ------------------------------------------
-- 第 1 步：加列
-- ------------------------------------------
-- 刻意不带 DEFAULT 和 NOT NULL：
--   1. 不带 DEFAULT → PostgreSQL 11+ 属于纯元数据操作，不重写表、不长时间持锁，
--      即使表很大也是毫秒级。带 DEFAULT 会触发整表重写（旧版本）或至少一次
--      全表扫描（新版本），在锁表期间所有会话查询都会被阻塞。
--   2. 允许 NULL → NULL 的语义就是「未删除」，存量数据无需回填。
--      若想用 NOT NULL，就必须先给 100 多万行填一个哨兵值（如 epoch），
--      而哨兵值会让「未删除」有两个表示（NULL 和 epoch），查询条件和索引
--      都要跟着复杂化。用 NULL 表示「没有删除时间」是最省事也最不容易写错的。
ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

COMMENT ON COLUMN sessions.deleted_at IS
    '逻辑删除时间；NULL 表示未删除。删除会话只写这一列，消息记录全部保留';

-- ------------------------------------------
-- 第 2 步：部分索引
-- ------------------------------------------
-- 为什么是部分索引（WHERE deleted_at IS NULL）而不是普通复合索引：
--   业务查询永远带 deleted_at IS NULL，已删除的行不会进入任何结果集。
--   把它们排除在索引之外，索引体积只随「活跃会话」增长，
--   已删除会话再多也不会拖慢列表查询。
--
-- 为什么 updated_at 不写 DESC：
--   B-tree 可以反向扫描，ASC 索引同样能服务 ORDER BY updated_at DESC。
--   显式写 DESC 只在「同一查询里升降序混用」时才有意义。
--
-- 注意：查询条件必须写成 deleted_at IS NULL（不能是 = NULL 或 IS NOT TRUE），
--       否则 PostgreSQL 的谓词蕴含推理匹配不上这个索引，会退化成全表扫描。
CREATE INDEX IF NOT EXISTS idx_sessions_user_active
    ON sessions (user_id, updated_at)
    WHERE deleted_at IS NULL;

-- ------------------------------------------
-- 第 3 步：后置核验
-- ------------------------------------------
DO $$
DECLARE
    v_has_column  boolean;
    v_has_index   boolean;
    v_deleted     bigint;
    v_total       bigint;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'sessions'
           AND column_name = 'deleted_at'
           AND data_type = 'timestamp with time zone'
    ) INTO v_has_column;

    IF NOT v_has_column THEN
        RAISE EXCEPTION 'deleted_at 列未创建成功';
    END IF;

    SELECT EXISTS (
        SELECT 1 FROM pg_indexes
         WHERE schemaname = current_schema()
           AND indexname = 'idx_sessions_user_active'
    ) INTO v_has_index;

    IF NOT v_has_index THEN
        RAISE EXCEPTION 'idx_sessions_user_active 索引未创建成功';
    END IF;

    SELECT COUNT(*) FILTER (WHERE deleted_at IS NOT NULL), COUNT(*)
      INTO v_deleted, v_total
      FROM sessions;

    -- 新加的列必然是 NULL。如果这里不是 0，说明脚本被改过或者库里有别的写入方，
    -- 应当人工确认后再继续，而不是默默接受。
    IF v_deleted > 0 THEN
        RAISE EXCEPTION '发现 % 行 deleted_at 非空（共 % 行），迁移结果不符合预期',
            v_deleted, v_total;
    END IF;

    RAISE NOTICE '迁移 003 完成：sessions 共 % 行，全部未删除（deleted_at 为 NULL）', v_total;
END $$;

COMMIT;
