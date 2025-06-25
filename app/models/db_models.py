from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from app.database import Base
from datetime import datetime
import uuid

class DBAgent(Base):
    __tablename__ = "agents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(String, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    model = Column(String, nullable=False)
    system_prompt = Column(String, nullable=False)
    is_public = Column(Boolean, default=False, index=True)
    tools = Column(ARRAY(String))
    # Rename 'metadata' to something else like 'agent_metadata'
    agent_metadata = Column(JSONB, name="metadata")  # The 'name' parameter keeps the column name as 'metadata' in DB
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        {'schema': 'llm'},
    )