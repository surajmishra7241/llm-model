import asyncio
import logging
from sqlalchemy import text
from app.database import engine, Base
from app.models.db_models import DBAgent, DBDocument, DBTrainingJob, DBConversation, DBMessage
from app.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_schema():
    """Create the llm schema if it doesn't exist"""
    try:
        async with engine.begin() as conn:
            # Create schema
            await conn.execute(text("CREATE SCHEMA IF NOT EXISTS llm"))
            logger.info("Schema 'llm' created or already exists")
    except Exception as e:
        logger.error(f"Error creating schema: {str(e)}")
        raise

async def create_tables():
    """Create all tables in the correct order"""
    try:
        async with engine.begin() as conn:
            # Set search path to include llm schema
            await conn.execute(text("SET search_path TO llm, public"))
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("All tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        raise

async def verify_tables():
    """Verify that all required tables exist"""
    required_tables = [
        'agents', 'documents', 'training_jobs', 'conversations', 'messages'
    ]
    
    try:
        async with engine.begin() as conn:
            for table in required_tables:
                result = await conn.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'llm' 
                        AND table_name = '{table}'
                    )
                """))
                exists = result.scalar()
                if exists:
                    logger.info(f"✅ Table 'llm.{table}' exists")
                else:
                    logger.error(f"❌ Table 'llm.{table}' does not exist")
    except Exception as e:
        logger.error(f"Error verifying tables: {str(e)}")
        raise

async def init_db():
    """Initialize the database with schema and tables"""
    try:
        logger.info("Starting database initialization...")
        
        # Step 1: Create schema
        await create_schema()
        
        # Step 2: Create tables
        await create_tables()
        
        # Step 3: Verify tables
        await verify_tables()
        
        logger.info("✅ Database initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        raise
    finally:
        # Close the engine
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(init_db())
