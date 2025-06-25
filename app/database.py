from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Get the database URL as string and ensure it uses asyncpg
database_url = str(settings.DATABASE_URL).replace(
    "postgresql://", 
    "postgresql+asyncpg://"
)

engine = create_async_engine(
    database_url,  # Use the modified URL string
    echo=settings.DEBUG,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={
        "server_settings": {
            "application_name": settings.APP_NAME,
            "search_path": "llm,public"
        }
    }
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            await session.rollback()
            raise
        finally:
            await session.close()