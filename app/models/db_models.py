# app/models/db_models.py
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Integer, or_
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, UUID
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime
import uuid

class DBAgent(Base):
    __tablename__ = "agents"
    __table_args__ = {'schema': 'llm'}  # Add schema here
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(String, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    model = Column(String, nullable=False)
    system_prompt = Column(String, nullable=False)
    is_public = Column(Boolean, default=False, index=True)
    tools = Column(ARRAY(String))
    agent_metadata = Column(JSONB)  
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    documents = relationship("DBDocument", back_populates="agent", cascade="all, delete-orphan")
    training_jobs = relationship("DBTrainingJob", back_populates="agent", cascade="all, delete-orphan")
    conversations = relationship("DBConversation", back_populates="agent", cascade="all, delete-orphan")

class DBDocument(Base):
    __tablename__ = "documents"
    __table_args__ = {'schema': 'llm'}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey('llm.agents.id'), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    filename = Column(String, nullable=False)
    content_type = Column(String)
    size = Column(Integer)
    qdrant_id = Column(String)
    document_metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    agent = relationship("DBAgent", back_populates="documents")

class DBTrainingJob(Base):
    __tablename__ = "training_jobs"
    __table_args__ = {'schema': 'llm'}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey('llm.agents.id'), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False)
    config = Column(JSONB)
    result = Column(JSONB)
    error_message = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    agent = relationship("DBAgent", back_populates="training_jobs")

class DBConversation(Base):
    __tablename__ = "conversations"
    __table_args__ = {'schema': 'llm'}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey('llm.agents.id'), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    memory_id = Column(String)
    context = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    agent = relationship("DBAgent", back_populates="conversations")
    messages = relationship("DBMessage", back_populates="conversation", cascade="all, delete-orphan")

class DBMessage(Base):
    __tablename__ = "messages"
    __table_args__ = {'schema': 'llm'}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey('llm.conversations.id'), nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    message_metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("DBConversation", back_populates="messages")