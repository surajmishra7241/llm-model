# app/services/mcp_client.py
import httpx
import logging
from app.config import settings
from typing import Dict, Any
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self):
        self.base_url = settings.MCP_SERVER_URL
        self.timeout = 30
    
    async def dispatch_task(self, agent_id: str, task_type: str, payload: Dict[str, Any]) -> str:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/tasks",
                    json={
                        "agent_id": agent_id,
                        "task_type": task_type,
                        "payload": payload
                    }
                )
                response.raise_for_status()
                return response.json()["task_id"]
        except httpx.HTTPStatusError as e:
            logger.error(f"MCP server returned error: {e.response.text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="MCP service error"
            )
        except Exception as e:
            logger.error(f"Failed to dispatch MCP task: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MCP service unavailable"
            )
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/tasks/{task_id}")
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"MCP server returned error: {e.response.text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="MCP service error"
            )
        except Exception as e:
            logger.error(f"Failed to get MCP task status: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MCP service unavailable"
            )