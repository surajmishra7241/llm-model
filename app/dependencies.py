from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils.auth import verify_token, get_current_user
from app.database import AsyncSessionLocal
import logging

logger = logging.getLogger(__name__)

async def get_db() -> AsyncSession:
    """Database session dependency"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Add this missing function
def get_db_session():
    """Returns the database session dependency"""
    return Depends(get_db)

# Authentication dependencies
def verify_token_dep():
    return Depends(verify_token)

def get_current_user_dep():
    return Depends(get_current_user)
