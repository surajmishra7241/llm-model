import asyncio
from sqlalchemy import text
from app.database import engine, Base
from app.models.db_models import DBAgent
import logging

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

async def init_db():
    async with engine.begin() as conn:
        # Create schema
        await conn.execute(text("CREATE SCHEMA IF NOT EXISTS llm"))
        # Set search path to include llm schema
        await conn.execute(text("SET search_path TO llm, public"))
        # Create tables
        await conn.run_sync(Base.metadata.create_all)
    print("Database initialized successfully!")

if __name__ == "__main__":
    asyncio.run(init_db())