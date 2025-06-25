from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Convert Pydantic PostgresDsn to string and ensure proper format
def get_database_url() -> str:
    url = str(settings.DATABASE_URL)
    # Ensure there's no double slash before the database name
    if '//' in url.split('@')[-1]:
        parts = url.split('@')
        parts[-1] = parts[-1].replace('//', '/')
        url = '@'.join(parts)
    return url

# Create engine with properly formatted URL string
engine = create_async_engine(
    get_database_url(),
    future=True,
    echo=True
)

AsyncSessionLocal = sessionmaker(
    engine, 
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as db:
        yield db