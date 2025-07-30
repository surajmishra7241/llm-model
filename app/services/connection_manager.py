# app/services/connection_manager.py - Enhanced with error recovery
import asyncio
import time
import logging
from typing import Dict, Any, Optional
from fastapi import WebSocket
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Connection:
    connection_id: str
    websocket: WebSocket
    user_id: Optional[str]
    agent_id: Optional[str]
    created_at: float
    last_activity: float
    is_active: bool = True
    retry_count: int = 0
    error_count: int = 0

class ConnectionManager:
    def __init__(self):
        self.connections: Dict[str, Connection] = {}
        self.user_connections: Dict[str, set] = {}
        self.cleanup_task = None
        self.max_connections = 100
        self.connection_timeout = 300  # 5 minutes
        self.max_retries = 3
        
        logger.info("Enhanced Connection Manager initialized")

    async def add_connection(
        self, 
        connection_id: str, 
        websocket: WebSocket,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ):
        """Add a new WebSocket connection with enhanced tracking"""
        try:
            # Check connection limits
            if len(self.connections) >= self.max_connections:
                logger.warning(f"Max connections reached: {len(self.connections)}")
                await self._cleanup_stale_connections()
                
                if len(self.connections) >= self.max_connections:
                    raise Exception("Server at maximum capacity")
            
            current_time = time.time()
            
            connection = Connection(
                connection_id=connection_id,
                websocket=websocket,
                user_id=user_id,
                agent_id=agent_id,
                created_at=current_time,
                last_activity=current_time
            )
            
            self.connections[connection_id] = connection
            
            # Track user connections
            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(connection_id)
            
            # Start cleanup task if not running
            if not self.cleanup_task:
                self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            logger.info(f"Connection added: {connection_id} (active: {len(self.connections)})")
            
        except Exception as e:
            logger.error(f"Failed to add connection {connection_id}: {str(e)}")
            raise

    async def remove_connection(self, connection_id: str):
        """Remove a WebSocket connection with proper cleanup"""
        try:
            if connection_id not in self.connections:
                logger.warning(f"Connection not found for removal: {connection_id}")
                return
            
            connection = self.connections[connection_id]
            
            # Mark as inactive
            connection.is_active = False
            
            # Remove from user tracking
            if connection.user_id and connection.user_id in self.user_connections:
                self.user_connections[connection.user_id].discard(connection_id)
                if not self.user_connections[connection.user_id]:
                    del self.user_connections[connection.user_id]
            
            # Close WebSocket if still open
            try:
                if connection.websocket:
                    await connection.websocket.close(code=1000, reason="Connection removed")
            except Exception as ws_error:
                logger.debug(f"WebSocket already closed for {connection_id}: {str(ws_error)}")
            
            # Remove from connections
            del self.connections[connection_id]
            
            logger.info(f"Connection removed: {connection_id} (active: {len(self.connections)})")
            
        except Exception as e:
            logger.error(f"Error removing connection {connection_id}: {str(e)}")

    async def get_connection(self, connection_id: str) -> Optional[Connection]:
        """Get connection with activity update"""
        connection = self.connections.get(connection_id)
        if connection and connection.is_active:
            connection.last_activity = time.time()
            return connection
        return None

    async def send_to_connection(
        self, 
        connection_id: str, 
        message: Dict[str, Any],
        retry_on_error: bool = True
    ) -> bool:
        """Send message to specific connection with error handling"""
        try:
            connection = await self.get_connection(connection_id)
            if not connection:
                logger.warning(f"Connection not found: {connection_id}")
                return False
            
            # Try to send message
            try:
                import json
                await connection.websocket.send_text(json.dumps(message))
                
                # Reset error count on successful send
                connection.error_count = 0
                return True
                
            except Exception as send_error:
                logger.error(f"Send failed for {connection_id}: {str(send_error)}")
                connection.error_count += 1
                
                # Remove connection if too many errors
                if connection.error_count >= 3:
                    logger.warning(f"Too many errors for {connection_id}, removing")
                    await self.remove_connection(connection_id)
                
                return False
                
        except Exception as e:
            logger.error(f"Error sending to connection {connection_id}: {str(e)}")
            return False

    async def broadcast_to_user(self, user_id: str, message: Dict[str, Any]) -> int:
        """Broadcast message to all user connections"""
        if user_id not in self.user_connections:
            return 0
        
        successful_sends = 0
        connection_ids = list(self.user_connections[user_id])
        
        for connection_id in connection_ids:
            success = await self.send_to_connection(connection_id, message)
            if success:
                successful_sends += 1
        
        return successful_sends

    async def _cleanup_stale_connections(self):
        """Clean up stale or inactive connections"""
        try:
            current_time = time.time()
            stale_connections = []
            
            for connection_id, connection in self.connections.items():
                # Check if connection is stale
                if (current_time - connection.last_activity) > self.connection_timeout:
                    stale_connections.append(connection_id)
                    continue
                
                # Check if WebSocket is still alive
                try:
                    await connection.websocket.ping()
                except Exception:
                    stale_connections.append(connection_id)
            
            # Remove stale connections
            for connection_id in stale_connections:
                await self.remove_connection(connection_id)
            
            if stale_connections:
                logger.info(f"Cleaned up {len(stale_connections)} stale connections")
                
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")

    async def _periodic_cleanup(self):
        """Periodic cleanup task"""
        try:
            while True:
                await asyncio.sleep(60)  # Run every minute
                
                if not self.connections:
                    # No connections, stop cleanup task
                    self.cleanup_task = None
                    break
                
                await self._cleanup_stale_connections()
                
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
        except Exception as e:
            logger.error(f"Periodic cleanup error: {str(e)}")

    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return len([c for c in self.connections.values() if c.is_active])

    def get_user_connection_count(self, user_id: str) -> int:
        """Get number of connections for a specific user"""
        return len(self.user_connections.get(user_id, set()))

    async def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        try:
            current_time = time.time()
            active_count = 0
            total_errors = 0
            avg_duration = 0
            
            for connection in self.connections.values():
                if connection.is_active:
                    active_count += 1
                    avg_duration += (current_time - connection.created_at)
                total_errors += connection.error_count
            
            if active_count > 0:
                avg_duration = avg_duration / active_count
            
            return {
                "total_connections": len(self.connections),
                "active_connections": active_count,
                "unique_users": len(self.user_connections),
                "total_errors": total_errors,
                "average_duration": round(avg_duration, 2),
                "max_connections": self.max_connections,
                "connection_timeout": self.connection_timeout
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}

    async def cleanup(self):
        """Cleanup all connections and resources"""
        try:
            # Cancel cleanup task
            if self.cleanup_task and not self.cleanup_task.done():
                self.cleanup_task.cancel()
            
            # Close all connections
            connection_ids = list(self.connections.keys())
            for connection_id in connection_ids:
                await self.remove_connection(connection_id)
            
            # Clear data structures
            self.connections.clear()
            self.user_connections.clear()
            
            logger.info("Connection Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Connection Manager cleanup error: {str(e)}")
