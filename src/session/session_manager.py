"""
会话管理器：提供会话和消息的 CRUD 操作。

复用 VectorStore 的 SQLAlchemy engine，统一连接池管理。
"""

import uuid
from typing import Optional, Sequence
from datetime import datetime, timedelta, timezone
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import select, delete, func, desc, update, exists
from sqlalchemy.dialects.postgresql import insert

from src.session.models import SessionModel, MessageModel


class DeleteOutcome(str, Enum):
    """删除会话的三种结果。

    为什么要区分「已删除过」和「不存在」：
    - 前端连点、网络重试都可能重复发同一次删除，这时应当幂等成功；
    - 别人的会话必须与「不存在」表现完全一致，否则 200/404 的差异
      就能被用来探测某个 session_id 是否存在过。
    两者混在一个 False 里，就必然牺牲其中一条。
    """

    DELETED = "deleted"
    ALREADY_DELETED = "already_deleted"
    NOT_FOUND = "not_found"


def _make_session_title(raw: Optional[str], max_len: int = 40) -> str:
    """
    把首条用户消息整理成历史列表用的标题。

    - 折叠所有空白（换行、连续空格），避免侧边栏条目被撑成多行
    - 超长截断并加省略号
    - 空内容返回空串，由前端决定占位文案
    """
    text = " ".join((raw or "").split())
    if not text:
        return ""
    return text if len(text) <= max_len else text[:max_len] + "…"


class SessionManager:
    """会话与消息生命周期管理器。"""

    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    # ── 会话管理 ──────────────────────────────────────────────

    async def create_session(
        self,
        user_id: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        创建新会话，并在同一次数据库操作里取回插入行的全部字段。

        用 INSERT ... RETURNING 而不是「插入后再 SELECT 回读」：
        回读发生在插入事务之外，一旦回读失败，调用方会误判为「创建失败」，
        但实际上会话行已经提交了，于是数据库里留下孤儿会话。
        RETURNING 让「创建成功」与「拿到字段」变成同一件事，没有中间态。

        Returns:
            {"session_id", "user_id", "created_at", "updated_at", "metadata"}
        """
        session_id = f"sess_{uuid.uuid4().hex[:12]}"

        async with self.engine.begin() as conn:
            result = await conn.execute(
                insert(SessionModel)
                .values(
                    session_id=session_id,
                    user_id=user_id,
                    metadata_json=metadata or {},
                )
                .returning(
                    SessionModel.session_id,
                    SessionModel.user_id,
                    SessionModel.created_at,
                    SessionModel.updated_at,
                    SessionModel.metadata_json,
                )
            )
            row = result.one()

        return {
            "session_id": row.session_id,
            "user_id": row.user_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "metadata": row.metadata_json or {},
        }

    async def list_sessions(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """列出用户的所有会话（按更新时间倒序），并附带会话标题。"""

        # 相关子查询：取该会话第一条用户消息，作为历史列表的标题来源。
        # 不新增数据库字段、不建视图，因此历史会话也能立即拥有标题，无需回填。
        title_subquery = (
            select(MessageModel.content)
            .where(
                MessageModel.session_id == SessionModel.session_id,
                MessageModel.role == "user",
            )
            .order_by(MessageModel.timestamp)
            .limit(1)
            .scalar_subquery()
            .label("title")
        )

        async with self.engine.connect() as conn:
            result = await conn.execute(
                select(
                    SessionModel.session_id,
                    SessionModel.user_id,
                    SessionModel.created_at,
                    SessionModel.updated_at,
                    SessionModel.metadata_json,
                    title_subquery,
                )
                .where(
                    SessionModel.user_id == user_id,
                    # 逻辑删除的会话不出现在历史列表里。
                    # 与下面的 EXISTS 是两条独立条件：前者决定「还不该看见」，
                    # 后者决定「看见也没内容」。
                    SessionModel.deleted_at.is_(None),
                    # 只返回有消息的会话：空会话（创建后未成功保存任何消息）
                    # 不进入历史列表，避免用户点进去看到空白
                    exists().where(
                        MessageModel.session_id == SessionModel.session_id
                    ),
                )
                .order_by(desc(SessionModel.updated_at))
                .limit(limit)
                .offset(offset)
            )
            rows = result.all()

        return [
            {
                "session_id": row.session_id,
                "user_id": row.user_id,
                "title": _make_session_title(row.title),
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "metadata": row.metadata_json or {},
            }
            for row in rows
        ]

    async def delete_session(self, session_id: str, user_id: str) -> DeleteOutcome:
        """
        逻辑删除会话：只写 deleted_at，**不删除任何消息**。

        删除后的语义（对调用方而言等同于「这个会话从未存在」）：
        - 列表不返回它
        - 消息历史 404
        - 聊天接口拒绝继续使用
        - 历史上下文构造读不到任何内容

        与物理删除的取舍：消息仍在库里，占用空间但可追溯；
        换来的是误删可从数据库层面恢复（把 deleted_at 置回 NULL 即可，
        但本服务刻意不开放恢复接口，避免「删了还能看见」的认知混乱）。

        必须传入 user_id：只有会话归属者才能删除，防止越权删除他人会话。

        Returns:
            DELETED         首次删除成功
            ALREADY_DELETED 本人之前删过（幂等：调用方重试应当视为成功）
            NOT_FOUND       不存在，或不属于该用户（两者不区分，避免探测存在性）
        """
        async with self.engine.begin() as conn:
            # 必须 FOR UPDATE，原因与物理删除时代相同：
            # 「读到会话还在」与「写入 deleted_at」之间，另一个请求可能正在
            # add_turn 写这一轮。加行锁后两者互斥 —— 要么删除等本轮写完，
            # 要么 add_turn 拿到锁时发现已删除而整体回滚。
            #
            # 只查归属、不查 deleted_at：这样「本人已删过」与「不是我的」
            # 能在同一把锁下区分开，不必额外再查一次。
            row = (
                await conn.execute(
                    select(SessionModel.deleted_at)
                    .where(
                        SessionModel.session_id == session_id,
                        SessionModel.user_id == user_id,
                    )
                    .with_for_update()
                )
            ).first()

            if row is None:
                return DeleteOutcome.NOT_FOUND

            if row.deleted_at is not None:
                # 幂等：重复点击、客户端重试都走这里。
                # 这里返回 NOT_FOUND 会让前端把一次成功的删除显示成失败。
                return DeleteOutcome.ALREADY_DELETED

            # 刻意不让 updated_at 变化：它是「最后活跃时间」，用于列表排序与展示，
            # 与「什么时候被删」是两个不同语义。删完再看 updated_at，
            # 应该是「最后一次对话发生在什么时候」，这个信息一旦被删除时间覆盖，
            # 就只能回头去 messages 表里翻。
            #
            # 必须显式把它写回自身：SessionModel.updated_at 声明了
            # onupdate=func.now()，SQLAlchemy 在**任何** UPDATE 语句里
            # 只要 values 中没有这一列，就会自动补上 updated_at = now()。
            # 也就是说，只写 deleted_at 是不够的，updated_at 会被连带刷掉。
            await conn.execute(
                update(SessionModel)
                .where(SessionModel.session_id == session_id)
                .values(
                    deleted_at=func.now(),
                    updated_at=SessionModel.updated_at,
                )
            )

        return DeleteOutcome.DELETED

    async def delete_session_if_empty(self, session_id: str, user_id: str) -> bool:
        """
        物理删除一个**没有任何消息**的会话，供内部回收使用。

        为什么不能复用 delete_session：那是逻辑删除，会把空会话变成
        「deleted_at 非空且没有消息」的幽灵行。这种行没有任何用户可见性，
        只有服务启动时的 purge_empty_sessions 会顺手清掉，
        于是一次对话失败就要在库里躺到下次重启。

        为什么是一条语句：原来是「先 get_message_count() 判断 0 条，
        再 delete_session()」，两步之间如果 add_turn 插进一条消息，
        就会把**有内容的会话**删掉。写成单条 DELETE ... NOT EXISTS，
        判定与删除在同一时刻完成，没有中间态。

        Returns:
            True 表示确实删掉了一行；False 表示会话不存在、不属于该用户、
            或已经有消息（此时不动它，交给正常流程）。
        """
        async with self.engine.begin() as conn:
            result = await conn.execute(
                delete(SessionModel).where(
                    SessionModel.session_id == session_id,
                    SessionModel.user_id == user_id,
                    ~exists().where(
                        MessageModel.session_id == SessionModel.session_id
                    ),
                )
            )
            return bool(result.rowcount)

    async def session_exists(self, session_id: str, user_id: str) -> bool:
        """
        检查会话是否存在、归属于指定用户、且未被逻辑删除。

        必须传入 user_id：归属条件直接下推到 SQL，
        避免"A 用户拿 B 用户的 session_id"造成越权读写。

        这个方法同时是逻辑删除的**主要收口点**：消息历史接口、聊天接口的
        session_id 校验、以及 get_history 内部，都先调它。
        在这里加上 deleted_at IS NULL，上面三条路径会同时把已删会话
        当作不存在处理 —— 少改一处、也少一处漏改的风险。

        Args:
            session_id: 会话 ID
            user_id: 调用方用户 ID

        Returns:
            True 如果存在、属于该用户、且未删除；False 否则
        """
        async with self.engine.connect() as conn:
            result = await conn.execute(
                select(SessionModel.session_id).where(
                    SessionModel.session_id == session_id,
                    SessionModel.user_id == user_id,
                    # 已删除 = 不存在。这是逻辑删除的核心约定：
                    # 对外所有正常接口都不该再看到它。
                    SessionModel.deleted_at.is_(None),
                )
            )
            return result.scalar() is not None

    async def get_session_info(self, session_id: str, user_id: str) -> Optional[dict]:
        """
        读取单个会话的基础信息（含归属校验）。

        与 list_sessions 的区别：只取一条、不做「空会话过滤」。
        用于「刚创建完立刻把会话信息返回给调用方」，
        这样不必为了拿 created_at / updated_at 再查一次会话列表。

        Returns:
            会话信息字典；会话不存在或不属于该用户时返回 None
        """
        async with self.engine.connect() as conn:
            result = await conn.execute(
                select(
                    SessionModel.session_id,
                    SessionModel.user_id,
                    SessionModel.created_at,
                    SessionModel.updated_at,
                    SessionModel.metadata_json,
                ).where(
                    SessionModel.session_id == session_id,
                    SessionModel.user_id == user_id,
                    # 当前无生产调用点，但条件必须写：将来一旦有人用它取
                    # 单个会话信息，默认就是安全的，不会读到已删除会话。
                    SessionModel.deleted_at.is_(None),
                )
            )
            row = result.first()

        if row is None:
            return None

        return {
            "session_id": row.session_id,
            "user_id": row.user_id,
            # 新建会话必然还没有消息，标题就是空串。
            # 需要真实标题的场景走 list_sessions（那里带首条消息的子查询）
            "title": "",
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "metadata": row.metadata_json or {},
        }

    async def purge_empty_sessions(self, older_than_minutes: int = 60) -> int:
        """
        兜底清理：删除创建超过 N 分钟且没有任何消息的空会话。

        用于防止 create_session 之后未成功保存消息而残留的空会话堆积。
        只删"已过宽限期"的空会话，不影响正在进行中的请求。

        Returns:
            实际删除的会话数量

        为什么这里保持物理删除、不跟着改逻辑删除：
        本方法清的是「没有任何消息的空壳」，它没有任何用户可见内容，
        也没有任何需要保留的历史，物理删掉最干净。
        如果改成逻辑删除，库里只会堆积一批「已删除且没有消息」的幽灵行，
        还得再写一个任务来清它们。
        已逻辑删除但**仍有消息**的会话不在本方法范围内（条件里有 NOT EXISTS），
        它们的物理清理是将来独立的定期任务，届时以 deleted_at 早于 N 天为条件。
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=max(1, int(older_than_minutes))
        )

        async with self.engine.begin() as conn:
            result = await conn.execute(
                delete(SessionModel).where(
                    SessionModel.created_at < cutoff,
                    ~exists().where(
                        MessageModel.session_id == SessionModel.session_id
                    ),
                )
            )
            return result.rowcount or 0

    # ── 消息管理 ──────────────────────────────────────────────

    async def add_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        retrieved_chunks: Optional[list[dict]] = None,
    ) -> str:
        """
        添加一条消息到会话。

        注意 user_id 是必传参数（原签名没有）：之前只校验 session_id 存在，
        任何人都能拿着别人的 session_id 往里写消息 —— 这是个归属缺口。
        目前生产代码只用 add_turn（那里校验了归属），本方法没有调用点，
        所以收紧密签名不会影响任何现有路径。

        Args:
            session_id: 会话 ID
            user_id: 归属用户，用于校验写入权限
            role: 'user' / 'assistant' / 'system'
            content: 消息内容
            retrieved_chunks: 检索到的文档片段（仅 assistant 角色需要）

        Returns:
            message_id

        Raises:
            ValueError: 会话不存在、已删除，或不属于该 user_id
            （三种情况用同一条消息，不区分——避免调用方据此探测他人会话）
        """
        message_id = f"msg_{uuid.uuid4().hex[:12]}"

        async with self.engine.begin() as conn:
            # 验证会话存在、归属正确、且未被逻辑删除
            result = await conn.execute(
                select(SessionModel.session_id).where(
                    SessionModel.session_id == session_id,
                    SessionModel.user_id == user_id,
                    SessionModel.deleted_at.is_(None),
                )
            )
            if not result.scalar():
                raise ValueError(
                    f"Session {session_id} 不存在、已删除或不属于 {user_id}"
                )

            # 插入消息
            await conn.execute(
                insert(MessageModel).values(
                    message_id=message_id,
                    session_id=session_id,
                    role=role,
                    content=content,
                    retrieved_chunks=retrieved_chunks,
                )
            )

            # 更新会话的 updated_at
            # 此处会话已在上方校验过必然存在，只需刷新活跃时间，不应使用 UPSERT，
            # 否则会因构造不完整行触发 sessions.user_id 的 NOT NULL 约束
            await conn.execute(
                update(SessionModel)
                .where(SessionModel.session_id == session_id)
                .values(updated_at=func.now())
            )

        return message_id

    async def add_turn(
        self,
        session_id: str,
        user_id: str,
        user_content: str,
        assistant_content: str,
        retrieved_chunks: Optional[list[dict]] = None,
    ) -> tuple[str, str, int]:
        """
        原子写入一轮对话：用户消息 + 助手消息（含检索快照）在同一事务内提交。

        「一轮」是一个不可分割的业务单位，所以这里不用两次 add_message 拼：
        - 两次事务之间失败会留下「有问无答」的半轮，下一轮读历史时
          这条孤零零的 user 消息会被当成上下文喂给模型，导致模型重复作答
        - 半轮还会让调用方的 persisted=False 语义失真——用户消息其实已经进库了

        会话行加 FOR UPDATE 的原因：单纯 EXISTS 检查挡不住并发删除，
        检查通过与会话被删之间没有原子性，随后插入的消息会变成
        「会话已不存在、消息还在」的孤儿数据。

        为什么显式写 timestamp：PostgreSQL 的 now() 取的是**事务开始时间**，
        同一事务内两条 INSERT 会拿到完全相同的值，get_history 的
        ORDER BY timestamp 就退化成不确定顺序（可能先答后问）。
        因此拿到行锁之后才取基准时间，助手消息用 base + 1ms 保证可排序。
        这是不改表结构的折中：1ms 是构造出的确定间隔，不依赖数据库时钟精度。
        若将来要求跨并发请求的严格全序，应给 messages 增加显式序号列。

        Args:
            session_id: 会话 ID
            user_id: 归属用户，用于校验写入权限（防止拿别人的 session_id 写消息）
            user_content: 用户提问
            assistant_content: 模型回答
            retrieved_chunks: 本轮引用的检索片段快照

        Returns:
            (user_message_id, assistant_message_id, 该会话消息总数)

        Raises:
            ValueError: 会话不存在、已被删除，或不属于该 user_id
        """
        user_message_id = f"msg_{uuid.uuid4().hex[:12]}"
        assistant_message_id = f"msg_{uuid.uuid4().hex[:12]}"

        async with self.engine.begin() as conn:
            locked = await conn.execute(
                select(SessionModel.session_id)
                .where(
                    SessionModel.session_id == session_id,
                    SessionModel.user_id == user_id,
                    # 已删除的会话不能再写入：否则「删完还能继续聊」，
                    # 等于删除只对列表生效、对数据没有任何约束力。
                    #
                    # 这一条同时兜住了「生成期间用户点了删除」的并发场景：
                    # 那一轮会在这里拿到 ValueError，chat.py 捕获后返回
                    # persisted=False（答案给了但没入库），不会留下半轮数据。
                    SessionModel.deleted_at.is_(None),
                )
                .with_for_update()
            )
            if locked.scalar() is None:
                raise ValueError(
                    f"Session {session_id} 不存在、已删除或不属于 {user_id}"
                )

            # 拿到锁之后再取时间：锁之前取的话，等待锁的时间会让这一轮
            # 的时间戳早于期间完成的其它轮次
            base = datetime.now(timezone.utc)

            await conn.execute(
                insert(MessageModel).values(
                    message_id=user_message_id,
                    session_id=session_id,
                    role="user",
                    content=user_content,
                    timestamp=base,
                )
            )
            await conn.execute(
                insert(MessageModel).values(
                    message_id=assistant_message_id,
                    session_id=session_id,
                    role="assistant",
                    content=assistant_content,
                    retrieved_chunks=retrieved_chunks,
                    timestamp=base + timedelta(milliseconds=1),
                )
            )

            # 会话活跃时间：会话行已由上面的 FOR UPDATE 锁住，必然存在
            await conn.execute(
                update(SessionModel)
                .where(SessionModel.session_id == session_id)
                .values(updated_at=func.now())
            )

            # 消息总数在同一事务里取：省一次数据库往返，
            # 且保证调用方拿到的计数与刚写入的这一轮严格一致
            counted = await conn.execute(
                select(func.count(MessageModel.message_id)).where(
                    MessageModel.session_id == session_id
                )
            )
            message_count = counted.scalar() or 0

        return user_message_id, assistant_message_id, message_count

    async def get_history(
        self,
        session_id: str,
        user_id: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        获取会话的最近 N 条消息（按时间正序）。

        必须传入 user_id：非归属者直接返回空列表，
        既不泄露会话是否存在，也不会读到他人对话内容。

        Returns:
            [{"role": "...", "content": "...", "timestamp": "..."}, ...]
        """
        # 归属与删除状态校验：会话不存在、已删除、或不属于该用户，一律返回空列表。
        # 这里是逻辑删除必须堵住的最后一道 —— RAG 的历史上下文正是通过
        # 本方法进入 prompt 的。漏了它，已删除会话的旧消息还会被喂给模型，
        # 表现为「界面上删了，回答里却还在引用那段旧对话」。
        if not await self.session_exists(session_id, user_id=user_id):
            return []

        async with self.engine.connect() as conn:
            result = await conn.execute(
                # 显式列选择：MessageModel 使用 legacy Column 定义，
                # select(Model) + scalars() 会退化为返回首列原始值（str），
                # 无法通过 row.message_id 访问
                select(
                    MessageModel.message_id,
                    MessageModel.role,
                    MessageModel.content,
                    MessageModel.timestamp,
                    MessageModel.retrieved_chunks,
                )
                .where(MessageModel.session_id == session_id)
                .order_by(desc(MessageModel.timestamp))
                .limit(limit)
            )
            rows = result.all()

        # 反转列表使其按时间正序（旧→新）
        rows = list(reversed(rows))

        return [
            {
                "message_id": row.message_id,
                "role": row.role,
                "content": row.content,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "retrieved_chunks": row.retrieved_chunks,
            }
            for row in rows
        ]

    async def get_message_count(self, session_id: str, user_id: str) -> int:
        """获取会话的消息总数（含归属与删除状态校验）。

        用 EXISTS 把归属条件压进同一条语句，而不是「先查归属再计数」：
        两次查询之间状态可能变化，也可能被调用方忘记校验。
        会话不存在、已删除、或不属于该用户，一律返回 0 ——
        调用方据此得到的唯一结论就是「这里没有可用的消息」。
        """
        async with self.engine.connect() as conn:
            result = await conn.execute(
                select(func.count(MessageModel.message_id)).where(
                    MessageModel.session_id == session_id,
                    exists().where(
                        SessionModel.session_id == session_id,
                        SessionModel.user_id == user_id,
                        SessionModel.deleted_at.is_(None),
                    ),
                )
            )
            return result.scalar() or 0
