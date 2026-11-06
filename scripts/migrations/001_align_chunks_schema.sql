-- ==========================================
-- 迁移 001：chunks 表结构与 init-db.sql 对齐
--
-- 适用对象：已经初始化过的数据库（Docker 数据卷或本地实例）。
--          新环境不需要执行本脚本，init-db.sql 建出来的就是目标结构。
--
-- 执行前请先做备份：
--   pg_dump -U rag_user -d rag_db -Fc -f rag_db_before_001.dump
--
-- 本脚本在单个事务内执行，任一步失败会整体回滚。
-- 可重复执行：类型变更与索引创建均带存在性判断。
-- ==========================================

BEGIN;

-- ------------------------------------------
-- 第 0 步：前置检查（只读，结果不合预期应中止）
-- ------------------------------------------
-- 期望：三列均无 NULL，source_page 无非数字内容。
--   SELECT COUNT(*) FILTER (WHERE embedding   IS NULL) AS null_embedding,
--          COUNT(*) FILTER (WHERE source_file IS NULL) AS null_source_file,
--          COUNT(*) FILTER (WHERE chunk_index IS NULL) AS null_chunk_index
--   FROM chunks;
--
--   SELECT COUNT(*) FROM chunks
--   WHERE source_page IS NOT NULL AND source_page <> ''
--     AND source_page !~ '^[0-9]+$';

DO $$
DECLARE
    v_null_embedding   bigint;
    v_null_source_file bigint;
    v_null_chunk_index bigint;
    v_bad_page         bigint;
    v_is_text          boolean;
BEGIN
    -- 若 source_page 已经是 integer，说明本脚本已执行过，跳过相关检查
    SELECT (data_type <> 'integer')
      INTO v_is_text
      FROM information_schema.columns
     WHERE table_schema = 'public' AND table_name = 'chunks'
       AND column_name = 'source_page';

    IF v_is_text IS NULL THEN
        RAISE EXCEPTION 'chunks.source_page 不存在，本脚本不适用';
    END IF;

    SELECT COUNT(*) FILTER (WHERE embedding IS NULL),
           COUNT(*) FILTER (WHERE source_file IS NULL),
           COUNT(*) FILTER (WHERE chunk_index IS NULL)
      INTO v_null_embedding, v_null_source_file, v_null_chunk_index
      FROM chunks;

    IF v_null_embedding > 0 OR v_null_source_file > 0 OR v_null_chunk_index > 0 THEN
        RAISE EXCEPTION
            '存在空值，不能收紧 NOT NULL：embedding=%, source_file=%, chunk_index=%',
            v_null_embedding, v_null_source_file, v_null_chunk_index;
    END IF;

    IF v_is_text THEN
        SELECT COUNT(*) INTO v_bad_page
          FROM chunks
         WHERE source_page IS NOT NULL AND source_page <> ''
           AND source_page !~ '^[0-9]+$';

        IF v_bad_page > 0 THEN
            RAISE EXCEPTION 'source_page 有 % 行无法转成整数，请先修正数据', v_bad_page;
        END IF;
    END IF;

    RAISE NOTICE '前置检查通过';
END $$;

-- ------------------------------------------
-- 第 1 步：source_page  TEXT -> INTEGER
-- 空字符串（历史上由向量库写入侧把 None 转成 "" 产生）转回真正的 NULL
-- ------------------------------------------
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'chunks'
           AND column_name = 'source_page' AND data_type <> 'integer'
    ) THEN
        ALTER TABLE chunks
            ALTER COLUMN source_page TYPE integer
            USING NULLIF(source_page, '')::integer;
        RAISE NOTICE 'source_page 已转换为 integer';
    ELSE
        RAISE NOTICE 'source_page 已是 integer，跳过';
    END IF;
END $$;

-- ------------------------------------------
-- 第 2 步：收紧三个基本字段的 NOT NULL
-- source_page 不在此列：部分文档本身没有页码，NULL 是合法值
-- ------------------------------------------
ALTER TABLE chunks ALTER COLUMN embedding   SET NOT NULL;
ALTER TABLE chunks ALTER COLUMN source_file SET NOT NULL;
ALTER TABLE chunks ALTER COLUMN chunk_index SET NOT NULL;

-- ------------------------------------------
-- 第 3 步：补 source_file 索引（delete_by_source_file 的过滤列）
-- ------------------------------------------
CREATE INDEX IF NOT EXISTS idx_chunks_source_file ON chunks (source_file);

-- ------------------------------------------
-- 第 4 步：后置核验（不通过则整个事务回滚）
-- ------------------------------------------
DO $$
DECLARE
    v_type    text;
    v_nullable text;
    v_nulls   bigint;
BEGIN
    SELECT data_type INTO v_type
      FROM information_schema.columns
     WHERE table_schema = 'public' AND table_name = 'chunks'
       AND column_name = 'source_page';

    IF v_type <> 'integer' THEN
        RAISE EXCEPTION 'source_page 类型仍为 %，应为 integer', v_type;
    END IF;

    SELECT is_nullable INTO v_nullable
      FROM information_schema.columns
     WHERE table_schema = 'public' AND table_name = 'chunks'
       AND column_name = 'source_page';

    IF v_nullable <> 'YES' THEN
        RAISE EXCEPTION 'source_page 不应被设为 NOT NULL（部分文档无页码）';
    END IF;

    SELECT COUNT(*) INTO v_nulls FROM chunks WHERE embedding IS NULL;
    IF v_nulls > 0 THEN
        RAISE EXCEPTION 'embedding 仍存在 % 行 NULL', v_nulls;
    END IF;

    RAISE NOTICE '后置核验通过';
END $$;

COMMIT;

-- ==========================================
-- 执行后可选验证：
--   SELECT COUNT(*) FILTER (WHERE source_page IS NULL) AS no_page,
--          COUNT(*) FILTER (WHERE source_page IS NOT NULL) AS has_page
--   FROM chunks;
--
-- 回滚方式：
--   本脚本在事务内执行，失败自动回滚；
--   若已 COMMIT 需要回退，用执行前的 dump 恢复，或手工：
--   ALTER TABLE chunks ALTER COLUMN source_page TYPE text USING source_page::text;
--   ALTER TABLE chunks ALTER COLUMN embedding DROP NOT NULL;  -- 等
-- ==========================================
