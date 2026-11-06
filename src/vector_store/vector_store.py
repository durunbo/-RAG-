"""
向量存储层: PostgreSQL + pgvector 实现。

保持与原有 Chroma 版本相同的接口契约,确保调用方无需修改。
保留所有业务逻辑(时间重排、中文数字转换等)。
"""

import asyncio
import json
import os
import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from src.chunk_strategies import EnhancedChunk
from src.embedding.embedding_model import EmbeddingModel, build_embedding_text


# ==========================================================
# 编码并发控制
# ==========================================================
# BGE-M3 编码是 CPU 密集操作，多路并发只会互相拖慢（抢同一个模型和 CPU 核）。
# 用信号量限制同时编码的数量，把并发请求转为短暂排队，整体吞吐更稳。
_EMBED_CONCURRENCY = 2
_embed_semaphore: Optional[asyncio.Semaphore] = None


def _get_embed_semaphore() -> asyncio.Semaphore:
    """惰性创建，确保绑定到当前运行的事件循环。"""
    global _embed_semaphore
    if _embed_semaphore is None:
        _embed_semaphore = asyncio.Semaphore(_EMBED_CONCURRENCY)
    return _embed_semaphore


logger = logging.getLogger(__name__)


# ==========================================================
# 配置类
# ==========================================================
@dataclass
class VectorStoreConfig:
    """向量库配置,单集合,PostgreSQL持久化存储"""
    
    # PostgreSQL 连接字符串
    # 格式: postgresql+asyncpg://user:password@host:port/dbname
    database_url: str = "postgresql+asyncpg://rag_user:rag_secret_2024@localhost:5432/rag_db"


# ==========================================================
# 内部工具函数
# ==========================================================
def _make_chunk_id(source_file: str, chunk_index: int) -> str:
    """
    生成全局唯一的向量ID
    ID规则集中管理,后续修改规则仅需改此处,不扩散到业务逻辑
    """
    return f"{source_file}_{chunk_index}"


def _to_pgvector(value) -> str:
    """
    向量 → pgvector字面量。唯一序列化出口。
    
    支持numpy array和Python list两种输入类型。
    使用紧凑格式(无空格),比json.dumps节省约30%空间。
    
    Args:
        value: List[float] 或 np.ndarray
    
    Returns:
        pgvector格式的字符串,如 "[0.1,0.2,0.3]"
    """
    if hasattr(value, "tolist"):     # 兼容numpy array
        value = value.tolist()
    return "[" + ",".join(repr(float(x)) for x in value) + "]"


# ==========================================================
# 异常类
# ==========================================================
class VectorStoreError(Exception):
    """向量存储操作异常。"""
    pass


# ==========================================================
# 向量库核心类
# ==========================================================
class VectorStore:
    """
    基于 PostgreSQL + pgvector 的向量存储。
    
    职责:
    1. 管理数据库连接池
    2. 提供向量入库接口(add_chunks)
    3. 提供相似度检索接口(search)
    4. 提供基础查询接口(count, delete_by_source_file, get_process_chunks)
    
    注意:
    - 所有方法均为异步,调用方需 await
    - embedding 维度由 EmbeddingModel 决定,当前为 1024
    - 保留原有 Chroma 版本的所有业务逻辑(时间重排等)
    """
    
    def __init__(self, embedding_model: EmbeddingModel, config: Optional[VectorStoreConfig] = None):
        """
        依赖注入 EmbeddingModel,全局共用一个实例,VectorStore 不创建模型
        :param embedding_model: 外部初始化好的 EmbeddingModel 实例
        :param config: 向量存储配置
        """
        self.embedding_model = embedding_model
        self.config = config or VectorStoreConfig()
        
        # 创建异步引擎(延迟初始化,首次使用时建立连接)
        self._engine: Optional[AsyncEngine] = None
    
    @property
    def engine(self) -> AsyncEngine:
        """懒加载数据库引擎。"""
        if self._engine is None:
            self._engine = create_async_engine(
                self.config.database_url,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30.0,
                echo=False,  # 生产环境关闭 SQL 日志
            )
            logger.info("PostgreSQL 连接池初始化完成")
        return self._engine
    
    async def initialize(self) -> None:
        """
        显式初始化数据库连接。
        
        可在应用启动时调用,提前发现连接问题。
        """
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                logger.info("PostgreSQL 连接测试成功")
        except Exception as exc:
            raise VectorStoreError(f"无法连接到 PostgreSQL: {exc}") from exc
    
    # ----------------------------------------
    # 内部工具：中文数字转阿拉伯数字
    # ----------------------------------------
    @staticmethod
    def _cn_num_to_int(cn: str) -> Optional[int]:
        """
        将常见中文数字转换为整数。

        支持：
        - 一、二、三……九
        - 十
        - 十一、十二……十九
        - 二十、三十……
        - 二十一、三十五……
        - 两

        该方法只负责中文数字转换，不处理“天”“周”“月”等时间单位。
        """
        if not cn:
            return None

        # 单个中文数字
        simple_map = {
            "零": 0,
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }

        if cn in simple_map:
            return simple_map[cn]

        # 处理"十X",例如：
        # 十一 -> 11
        # 十二 -> 12
        # 十九 -> 19
        if cn.startswith("十"):
            if len(cn) == 2:
                return 10

            unit = simple_map.get(cn[1])
            if unit is not None:
                return 10 + unit

        # 处理"X十",例如：
        # 二十 -> 20
        # 三十 -> 30
        if cn.endswith("十"):
            if len(cn) == 2:
                tens = simple_map.get(cn[0])
                if tens is not None:
                    return tens * 10

        # 处理"X十Y",例如：
        # 二十一 -> 21
        # 三十五 -> 35
        if len(cn) == 3 and cn[1] == "十":
            tens = simple_map.get(cn[0])
            ones = simple_map.get(cn[2])

            if tens is not None and ones is not None:
                return tens * 10 + ones

        return None

    # ----------------------------------------
    # 内部工具：从文本中提取时间节点（天数）
    # ----------------------------------------
    def _extract_time_days(self, text: str) -> Optional[int]:
        """
        从文本中识别时间节点，统一转换为天数。

        支持常见表达：
        - 第7天 / 第七天
        - 入职7天 / 入职七天
        - 7天 / 七天
        - 第2周 / 第二周
        - 入职2周 / 入职两周
        - 2周 / 两周
        - 30天 / 三十天
        - 一个月 / 1个月
        - 两个月 / 2个月
        - 半个月 / 15天
        - 半年 / 6个月 / 六个月

        识别不到返回 None。

        注意：
        这里的目的不是判断业务规则，而是把自然语言中的
        "时间节点"统一转换成可比较的天数。
        """

        if not text:
            return None

        text = text.strip()

        # ----------------------------------------
        # 模式1：X天
        # ----------------------------------------
        day_match = re.search(
            r"(?:第|入职\s*)?([零一二三四五六七八九十两\d]+)\s*天",
            text
        )

        if day_match:
            num_str = day_match.group(1)

            # 阿拉伯数字
            if num_str.isdigit():
                return int(num_str)

            # 中文数字
            num = self._cn_num_to_int(num_str)
            if num is not None:
                return num

        # ----------------------------------------
        # 模式2：X周
        # ----------------------------------------
        week_match = re.search(
            r"(?:第|入职\s*)?([零一二三四五六七八九十两\d]+)\s*(?:周|星期|礼拜)",
            text
        )

        if week_match:
            num_str = week_match.group(1)

            # 阿拉伯数字
            if num_str.isdigit():
                return int(num_str) * 7

            # 中文数字
            num = self._cn_num_to_int(num_str)
            if num is not None:
                return num * 7

        # ----------------------------------------
        # 模式3：X个月
        # ----------------------------------------
        month_match = re.search(
            r"([一二三四五六七八九十两\d]+)\s*个?月",
            text
        )

        if month_match:
            num_str = month_match.group(1)

            # 阿拉伯数字
            if num_str.isdigit():
                return int(num_str) * 30

            # 中文数字
            num = self._cn_num_to_int(num_str)
            if num is not None:
                return num * 30

        # ----------------------------------------
        # 模式4：半年
        # ----------------------------------------
        if re.search(r"半年|半年度", text):
            return 180

        # ----------------------------------------
        # 模式5：半个月
        # ----------------------------------------
        if re.search(r"半个月|半月", text):
            return 15

        return None

    # ----------------------------------------
    # 内部工具：从chunk metadata中提取时间节点（天数）
    # ----------------------------------------
    def _get_chunk_time_days(self, metadata: dict) -> Optional[int]:
        """优先从stage提取，其次从heading_path提取"""

        # 优先看stage字段
        stage = metadata.get("stage", "")
        if stage:
            days = self._extract_time_days(stage)
            if days is not None:
                return days

        # 再看heading_path
        heading = metadata.get("heading_path", "")
        if heading:
            days = self._extract_time_days(heading)
            if days is not None:
                return days

        return None

    # ----------------------------------------
    # 内部工具：时间节点轻量重排
    # ----------------------------------------
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        时间节点轻量重排。

        处理流程：
        1. 查询没有明确时间节点 → 完全保持原Dense排序
        2. 查询存在时间节点 → 对候选Chunk进行时间匹配判断
        3. 时间越接近，给予小幅加分
        4. 时间明显冲突，给予小幅减分
        5. 最终仍以综合分数排序

        这里使用"小幅调整"的原因：
        Dense相似度仍然是主要排序依据，
        时间信息只负责解决"同主题不同时间节点"的竞争问题。

        注意：
        该方法不会改变返回数据结构，只会调整已有 score。
        """

        if not results:
            return results

        # ----------------------------------------
        # 第一步：提取用户查询中的时间节点
        # ----------------------------------------
        query_days = self._extract_time_days(query)

        # 查询没有明确时间信息：
        # 不进行任何重排，保持原有Dense检索结果。
        if query_days is None:
            return results

        reranked = []

        for item in results:
            # 复制当前结果，避免直接修改原始结果对象
            new_item = item.copy()

            metadata = item.get("metadata") or {}

            # 从Chunk metadata中获取时间节点
            chunk_days = self._get_chunk_time_days(metadata)

            # ----------------------------------------
            # 候选Chunk没有时间信息
            # ----------------------------------------
            # 无法进行时间比较，因此保持原Dense分数。
            if chunk_days is None:
                reranked.append(new_item)
                continue

            # ----------------------------------------
            # 第二步：计算查询时间与Chunk时间的差距
            # ----------------------------------------
            diff = abs(query_days - chunk_days)

            original_score = item["score"]

            # ----------------------------------------
            # 时间完全一致
            # ----------------------------------------
            if diff == 0:
                new_item["score"] = round(original_score + 0.05, 4)

            # ----------------------------------------
            # 时间非常接近
            # ----------------------------------------
            elif diff <= 1:
                new_item["score"] = round(original_score + 0.04, 4)

            # ----------------------------------------
            # 时间差距较大
            # ----------------------------------------
            elif diff >= 14:
                new_item["score"] = round(original_score - 0.04, 4)

            # ----------------------------------------
            # 中间范围不调整
            # ----------------------------------------

            reranked.append(new_item)

        # 根据调整后的综合分数重新排序
        reranked.sort(key=lambda x: x["score"], reverse=True)

        return reranked

    # ----------------------------------------
    # 1. 批量添加Chunk
    # ----------------------------------------
    async def add_chunks(self, chunks: List[EnhancedChunk]) -> int:
        """
        批量添加Chunk到向量库
        
        Args:
            chunks: EnhancedChunk 列表
            
        Returns:
            成功插入的记录数
            
        Raises:
            VectorStoreError: 插入失败时抛出
        """
        if not chunks:
            logger.warning("尝试插入空列表")
            return 0

        # 第1步: 事务外批量编码(CPU密集,不占DB连接)
        logger.info(f"开始批量编码 {len(chunks)} 条chunk...")
        # 同步的 CPU 推理交给线程池，避免阻塞事件循环
        async with _get_embed_semaphore():
            embeddings = await asyncio.to_thread(
                self.embedding_model.encode_batch, chunks
            )
        logger.info("批量编码完成")

        # 第2步: 构造批量参数
        params_list = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = _make_chunk_id(chunk.source_file, chunk.chunk_index)
            heading_path_str = " / ".join(chunk.heading_path) if chunk.heading_path else ""
            embedding_text = build_embedding_text(chunk)
            
            params_list.append({
                "id": chunk_id,
                "embedding": _to_pgvector(embedding),
                "text": embedding_text,
                "source_file": chunk.source_file,
                "heading_path": heading_path_str,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "document_type": chunk.document_type,
                # 原样传递 int / None，与 EnhancedChunk.source_page 的语义一致。
                # 不要 str()、也不要把 None 转成 ""：
                # 列类型是 INTEGER，asyncpg 对整数参数做严格类型检查，
                # 传 "4" 或 "" 都会抛 DataError。
                "source_page": chunk.source_page,
                "section": chunk.section,
                "question": chunk.question,
                "stage": chunk.stage,
                "responsible_department": chunk.responsible_department,
                "form_name": chunk.form_name,
            })

        # 第3步: 批量写入数据库(IO密集)
        try:
            # 记录写入前总数
            before_count = await self.count()
            
            if not params_list:
                return 0
            
            async with self.engine.begin() as conn:
                insert_sql = text("""
                    INSERT INTO chunks (
                        id, embedding, text, source_file,
                        heading_path, chunk_index, token_count,
                        document_type, source_page, section,
                        question, stage, responsible_department, form_name
                    ) VALUES (
                        :id, :embedding, :text, :source_file,
                        :heading_path, :chunk_index, :token_count,
                        :document_type, :source_page, :section,
                        :question, :stage, :responsible_department, :form_name
                    )
                    ON CONFLICT (id) DO NOTHING
                """)
                
                # 逐条插入（asyncpg 不支持 executemany）
                for params in params_list:
                    await conn.execute(insert_sql, params)
            
            # 通过count差值计算实际插入数
            after_count = await self.count()
            inserted_count = after_count - before_count
            
            logger.info(f"批量写入完成: 提交 {len(params_list)} 条, 实际插入 {inserted_count} 条")
        
        except Exception as exc:
            raise VectorStoreError(f"插入向量块失败: {exc}") from exc
        
        return inserted_count

    # ----------------------------------------
    # 2. 按文件名删除所有对应向量
    # ----------------------------------------
    async def delete_by_source_file(self, source_file: str) -> int:
        """
        根据 source_file 元数据过滤删除,是文档更新覆盖的核心接口
        
        Args:
            source_file: 文档名称
            
        Returns:
            删除的记录数
        """
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(
                    text("DELETE FROM chunks WHERE source_file = :source_file"),
                    {"source_file": source_file}
                )
                deleted = result.rowcount
                logger.info(f"删除源文件 '{source_file}' 的 {deleted} 条记录")
                return deleted
        except Exception as exc:
            raise VectorStoreError(f"删除记录失败: {exc}") from exc

    # ----------------------------------------
    # 3. Dense 检索 + 时间轻量重排
    # ----------------------------------------
    async def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        纯Dense向量检索 + 时间节点轻量重排
        接口完全不变，内部新增重排逻辑
        :return: 按相似度降序排列，每条包含 score、text、完整metadata
        """

        # ========== 【低优先级】参数校验 ==========
        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，当前值：{top_k}")
        if not query.strip():
            return []

        # 直接调用独立查询向量化接口
        async with _get_embed_semaphore():
            query_vector = await asyncio.to_thread(
                self.embedding_model.encode_query, query
            )

        # --------------------------------------------------
        # Dense候选集扩大
        # --------------------------------------------------
        # 先获取总数
        collection_count = await self.count()

        # 至少尝试召回10条，同时保持为top_k的3倍
        candidate_k = min(
            max(top_k * 3, 10),
            collection_count
        )

        try:
            async with self.engine.connect() as conn:
                # pgvector 余弦相似度查询
                # <=> 是余弦距离,1 - <=> 得到相似度(0-1)
                result = await conn.execute(
                    text("""
                        SELECT 
                            text,
                            1 - (embedding <=> :query_vec) AS score,
                            jsonb_build_object(
                                'source_file', source_file,
                                'heading_path', heading_path,
                                'chunk_index', chunk_index,
                                'token_count', token_count,
                                'document_type', document_type,
                                'source_page', source_page,
                                'section', section,
                                'question', question,
                                'stage', stage,
                                'responsible_department', responsible_department,
                                'form_name', form_name
                            ) AS metadata
                        FROM chunks
                        ORDER BY embedding <=> :query_vec
                        LIMIT :candidate_k
                    """),
                    {
                        "query_vec": _to_pgvector(query_vector),
                        "candidate_k": candidate_k,
                    }
                )
                
                rows = result.fetchall()
                
                output = []
                for row in rows:
                    output.append({
                        "score": float(row.score),
                        "text": row.text,
                        "metadata": dict(row.metadata) if row.metadata else {},
                    })
        
        except Exception as exc:
            raise VectorStoreError(f"检索失败: {exc}") from exc

        # --------------------------------------------------
        # 时间节点轻量重排
        # --------------------------------------------------
        output = self._rerank_results(query, output)

        # --------------------------------------------------
        # 截取最终TopK
        # --------------------------------------------------
        return output[:top_k]

    # ----------------------------------------
    # 4. 统计向量总数
    # ----------------------------------------
    async def count(self) -> int:
        """返回当前集合向量总数"""
        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(text("SELECT COUNT(*) FROM chunks"))
                count = result.scalar()
                return int(count) if count else 0
        except Exception as exc:
            raise VectorStoreError(f"查询记录数失败: {exc}") from exc

    # ----------------------------------------
    # 5. 获取库配置元数据
    # ----------------------------------------
    async def get_metadata(self) -> Dict:
        """返回向量库基础配置与Embedding配置信息"""
        embedding_meta = self.embedding_model.get_metadata()
        total_count = await self.count()
        return {
            "database_url": self.config.database_url,
            "total_count": total_count,
            "embedding_config": embedding_meta
        }

    # ----------------------------------------
    # 6. 获取指定流程文档的全部Chunk
    # ----------------------------------------
    async def get_process_chunks(self, source_file: str) -> List[Dict[str, Any]]:
        """
        获取指定文档的所有流程类Chunk，按chunk_index升序排列
        仅做数据读取，不做业务逻辑、不排序相似度、不生成向量

        :param source_file: 文件名（与metadata中存储的source_file完全一致）
        :return: 按chunk_index升序排列的Chunk列表，每条包含id、text、metadata
        """
        # 1. 参数校验
        if not isinstance(source_file, str):
            raise ValueError(f"source_file必须是字符串类型，当前类型：{type(source_file)}")
        source_file = source_file.strip()
        if not source_file:
            raise ValueError("source_file不能为空字符串")

        try:
            async with self.engine.connect() as conn:
                # 2. 双条件过滤查询
                result = await conn.execute(
                    text("""
                        SELECT 
                            id,
                            text,
                            jsonb_build_object(
                                'source_file', source_file,
                                'heading_path', heading_path,
                                'chunk_index', chunk_index,
                                'token_count', token_count,
                                'document_type', document_type,
                                'source_page', source_page,
                                'section', section,
                                'question', question,
                                'stage', stage,
                                'responsible_department', responsible_department,
                                'form_name', form_name
                            ) AS metadata
                        FROM chunks
                        WHERE source_file = :source_file
                          AND document_type = 'process'
                        ORDER BY chunk_index ASC
                    """),
                    {"source_file": source_file}
                )
                
                rows = result.fetchall()
                
                # 3. 转换为统一返回结构
                items = []
                for row in rows:
                    items.append({
                        "id": row.id,
                        "text": row.text,
                        "metadata": dict(row.metadata) if row.metadata else {},
                    })
                
                return items
        
        except Exception as exc:
            raise VectorStoreError(f"查询流程Chunk失败: {exc}") from exc
    
    async def close(self) -> None:
        """关闭数据库连接池。"""
        if self._engine:
            await self._engine.dispose()
            logger.info("PostgreSQL 连接池已关闭")
