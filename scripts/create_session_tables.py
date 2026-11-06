"""
创建会话与消息表的迁移脚本。

用法：
    python scripts/create_session_tables.py

依赖：
    - PostgreSQL 服务已启动
    - .env 中配置了 DATABASE_URL
"""

import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import get_settings
from src.session.models import Base


async def main():
    settings = get_settings()
    engine = create_async_engine(str(settings.database_url))

    try:
        async with engine.begin() as conn:
            # 创建所有表（如果已存在则跳过）
            await conn.run_sync(Base.metadata.create_all)
            print("[OK] sessions 和 messages 表创建成功")

            # 验证表是否存在
            result = await conn.execute(
                text(
                    """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                      AND table_name IN ('sessions', 'messages')
                    ORDER BY table_name
                    """
                )
            )
            tables = [row[0] for row in result.fetchall()]
            print(f"[INFO] 当前已创建的表: {tables}")

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
