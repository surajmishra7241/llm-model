from fastapi import Depends
from app.utils.auth import verify_token, get_current_user
from app.utils.helpers import get_db
from sqlalchemy.orm import Session

# Database dependency
def get_db_session():
    return Depends(get_db)

# Authentication dependencies
def verify_token_dep():
    return Depends(verify_token)

def get_current_user_dep():
    return Depends(get_current_user)