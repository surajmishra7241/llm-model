# app/services/voice_session_manager.py - Enhanced with better error handling
import time
import uuid
import logging
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from fastapi import WebSocket

logger = logging.getLogger(__name__)

@dataclass
class VoiceSession:
    session_id: str
    user_id: str
    agent_id: str
    websocket: WebSocket
    agent: Any
    created_at: float
    last_activity: float
    conversation_history: list = field(default_factory=list)
    is_active: bool = True
    error_count: int = 0
    processing_count: int = 0
    
    def add_to_history(self, user_message: str, agent_response: str):
        """Add conversation to history with automatic cleanup"""
        self.conversation_history.append({
            "timestamp": time.time(),
            "user": user_message,
            "agent": agent_response,
            "session_id": self.session_id
        })
        
        # Keep only last 20 exchanges
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        self.last_activity = time.time()
    
    def get_recent_history(self, count: int = 4) -> list:
        """Get recent conversation history formatted for LLM"""
        recent = self.conversation_history[-count:] if count > 0 else []
        formatted = []
        
        for item in recent:
            formatted.extend([
                {"role": "user", "content": item["user"]},
                {"role": "assistant", "content": item["agent"]}
            ])
        
        return formatted
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation"""
        if not self.conversation_history:
            return "New conversation"
        
        # Simple summary - last few exchanges
        recent = self.conversation_history[-3:]
        summary_parts = []
        
        for item in recent:
            user_preview = item['user'][:30] + "..." if len(item['user']) > 30 else item['user']
            agent_preview = item['agent'][:30] + "..." if len(item['agent']) > 30 else item['agent']
            summary_parts.append(f"User: {user_preview} | Agent: {agent_preview}")
        
        return " | ".join(summary_parts)
    
    def is_first_message(self) -> bool:
        """Check if this is the first message in the session"""
        return len(self.conversation_history) == 0
    
    def increment_error(self):
        """Increment error count"""
        self.error_count += 1
        self.last_activity = time.time()
    
    def increment_processing(self):
        """Increment processing count"""
        self.processing_count += 1
        self.last_activity = time.time()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        current_time = time.time()
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "duration": round(current_time - self.created_at, 2),
            "last_activity": round(current_time - self.last_activity, 2),
            "message_count": len(self.conversation_history),
            "error_count": self.error_count,
            "processing_count": self.processing_count,
            "is_active": self.is_active
        }

class VoiceSessionManager:
    def __init__(self):
        self.sessions: Dict[str, VoiceSession] = {}
        self.user_sessions: Dict[str, set] = {}
        self.session_timeout = 1800  # 30 minutes
        self.max_sessions_per_user = 5
        self.cleanup_task = None
        self.total_sessions_created = 0
        
        logger.info("Enhanced Voice Session Manager initialized")

    async def create_session(
        self, 
        connection_id: str, 
        user_id: str, 
        agent_id: str, 
        websocket: WebSocket, 
        agent: Any
    ) -> VoiceSession:
        """Create a new voice session with enhanced validation"""
        try:
            # Check user session limits
            if user_id in self.user_sessions:
                if len(self.user_sessions[user_id]) >= self.max_sessions_per_user:
                    # Clean up old sessions for this user
                    await self._cleanup_user_sessions(user_id)
                    
                    if len(self.user_sessions[user_id]) >= self.max_sessions_per_user:
                        raise Exception(f"User {user_id} has too many active sessions")
            
            current_time = time.time()
            
            session = VoiceSession(
                session_id=connection_id,
                user_id=user_id,
                agent_id=agent_id,
                websocket=websocket,
                agent=agent,
                created_at=current_time,
                last_activity=current_time
            )
            
            # Store session
            self.sessions[connection_id] = session
            
            # Track user sessions
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(connection_id)
            
            # Start cleanup task if needed
            if not self.cleanup_task:
                self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            self.total_sessions_created += 1
            
            logger.info(f"Voice session created: {connection_id} for user {user_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create voice session: {str(e)}")
            raise

    async def get_session(self, session_id: str) -> Optional[VoiceSession]:
        """Get session with activity update"""
        session = self.sessions.get(session_id)
        if session and session.is_active:
            session.last_activity = time.time()
            return session
        return None

    async def remove_session(self, session_id: str):
        """Remove session with proper cleanup"""
        try:
            if session_id not in self.sessions:
                logger.warning(f"Session not found for removal: {session_id}")
                return
            
            session = self.sessions[session_id]
            
            # Mark as inactive
            session.is_active = False
            
            # Remove from user tracking
            if session.user_id in self.user_sessions:
                self.user_sessions[session.user_id].discard(session_id)
                if not self.user_sessions[session.user_id]:
                    del self.user_sessions[session.user_id]
            
            # Cleanup session resources
            await session.cleanup() if hasattr(session, 'cleanup') else None
            
            # Remove from sessions
            del self.sessions[session_id]
            
            logger.info(f"Voice session removed: {session_id}")
            
        except Exception as e:
            logger.error(f"Error removing session {session_id}: {str(e)}")

    async def _cleanup_user_sessions(self, user_id: str):
        """Clean up old sessions for a specific user"""
        if user_id not in self.user_sessions:
            return
        
        user_session_ids = list(self.user_sessions[user_id])
        current_time = time.time()
        
        # Sort by last activity (oldest first)
        sessions_with_activity = []
        for session_id in user_session_ids:
            session = self.sessions.get(session_id)
            if session:
                sessions_with_activity.append((session_id, session.last_activity))
        
        sessions_with_activity.sort(key=lambda x: x[1])
        
        # Remove oldest sessions if over limit
        sessions_to_remove = len(sessions_with_activity) - self.max_sessions_per_user + 1
        
        for i in range(min(sessions_to_remove, len(sessions_with_activity))):
            session_id = sessions_with_activity[i][0]
            await self.remove_session(session_id)
            logger.info(f"Removed old session for user {user_id}: {session_id}")

    async def _cleanup_stale_sessions(self):
        """Clean up stale sessions"""
        try:
            current_time = time.time()
            stale_sessions = []
            
            for session_id, session in self.sessions.items():
                # Check if session is stale
                if (current_time - session.last_activity) > self.session_timeout:
                    stale_sessions.append(session_id)
                    continue
                
                # Check if session has too many errors
                if session.error_count > 10:
                    stale_sessions.append(session_id)
                    continue
                
                # Check if WebSocket is still alive
                try:
                    await session.websocket.ping()
                except Exception:
                    stale_sessions.append(session_id)
            
            # Remove stale sessions
            for session_id in stale_sessions:
                await self.remove_session(session_id)
            
            if stale_sessions:
                logger.info(f"Cleaned up {len(stale_sessions)} stale sessions")
                
        except Exception as e:
            logger.error(f"Session cleanup error: {str(e)}")

    async def _periodic_cleanup(self):
        """Periodic cleanup task"""
        try:
            while True:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                if not self.sessions:
                    # No sessions, stop cleanup task
                    self.cleanup_task = None
                    break
                
                await self._cleanup_stale_sessions()
                
        except asyncio.CancelledError:
            logger.info("Session cleanup task cancelled")
        except Exception as e:
            logger.error(f"Periodic session cleanup error: {str(e)}")

    def get_total_sessions(self) -> int:
        """Get total number of sessions ever created"""
        return self.total_sessions_created

    def get_active_session_count(self) -> int:
        """Get number of active sessions"""
        return len([s for s in self.sessions.values() if s.is_active])

    async def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        try:
            current_time = time.time()
            active_sessions = 0
            total_messages = 0
            total_errors = 0
            avg_duration = 0
            
            for session in self.sessions.values():
                if session.is_active:
                    active_sessions += 1
                    avg_duration += (current_time - session.created_at)
                total_messages += len(session.conversation_history)
                total_errors += session.error_count
            
            if active_sessions > 0:
                avg_duration = avg_duration / active_sessions
            
            return {
                "total_sessions": len(self.sessions),
                "active_sessions": active_sessions,
                "unique_users": len(self.user_sessions),
                "total_sessions_created": self.total_sessions_created,
                "total_messages": total_messages,
                "total_errors": total_errors,
                "average_duration": round(avg_duration, 2),
                "session_timeout": self.session_timeout,
                "max_sessions_per_user": self.max_sessions_per_user
            }
            
        except Exception as e:
            logger.error(f"Error getting session stats: {str(e)}")
            return {}

    async def broadcast_to_user_sessions(self, user_id: str, message: Dict[str, Any]) -> int:
        """Broadcast message to all sessions for a user"""
        if user_id not in self.user_sessions:
            return 0
        
        successful_sends = 0
        session_ids = list(self.user_sessions[user_id])
        
        for session_id in session_ids:
            session = await self.get_session(session_id)
            if session and session.websocket:
                try:
                    import json
                    await session.websocket.send_text(json.dumps(message))
                    successful_sends += 1
                except Exception as e:
                    logger.error(f"Failed to send to session {session_id}: {str(e)}")
                    session.increment_error()
        
        return successful_sends

    async def cleanup(self):
        """Cleanup all sessions and resources"""
        try:
            # Cancel cleanup task
            if self.cleanup_task and not self.cleanup_task.done():
                self.cleanup_task.cancel()
            
            # Remove all sessions
            session_ids = list(self.sessions.keys())
            for session_id in session_ids:
                await self.remove_session(session_id)
            
            # Clear data structures
            self.sessions.clear()
            self.user_sessions.clear()
            
            logger.info("Voice Session Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Voice Session Manager cleanup error: {str(e)}")
