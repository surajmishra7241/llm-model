from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Path
from fastapi.responses import StreamingResponse
from app.services.voice_service import VoiceService
from app.services.voice_agent_service import VoiceAgentService
from app.services.agent_service import AgentService
from app.dependencies import get_current_user
import io

router = APIRouter()

@router.post("/voice-chat/{agent_id}")
async def voice_chat_with_agent(
    agent_id: int = Path(..., title="The ID of the agent to chat with"),
    audio_file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    voice_service: VoiceService = Depends(),
    voice_agent_service: VoiceAgentService = Depends(),
    agent_service: AgentService = Depends(),
):
    """
    Handles a full voice conversation with an agent.
    1. Converts user's speech to text.
    2. Gets a conversational response from the agent's brain.
    3. Converts the response back to speech and streams it.
    """
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    try:
        # Get user ID from the token
        user_id = user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Could not validate credentials")

        # 1. Get agent details
        agent = await agent_service.get_agent(agent_id, user_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # 2. Speech-to-Text
        user_text = await voice_service.speech_to_text(audio_file)
        if not user_text:
            # Return empty audio if transcription fails
            return StreamingResponse(io.BytesIO(), media_type="audio/wav")

        # 3. Get response from agent's brain
        agent_response_text = await voice_agent_service.voice_chat(user_id, user_text, agent)

        # 4. Text-to-Speech
        audio_data = await voice_service.text_to_speech(agent_response_text)
        
        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        # Handle other exceptions
        raise HTTPException(status_code=500, detail=str(e))
