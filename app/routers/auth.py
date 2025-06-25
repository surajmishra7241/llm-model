from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from app.utils.auth import create_access_token
from app.config import settings

router = APIRouter(tags=["Authentication"])

# Mock user database - replace with real checks in production
fake_users_db = {
    "nodejs_service": {
        "username": "nodejs_service",
        "password": "nodejs_service_password",
        "id": "nodejs_service",  # Add this line
        "sub": "nodejs_service"   # Add this line for JWT compatibility
    }
}

@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}