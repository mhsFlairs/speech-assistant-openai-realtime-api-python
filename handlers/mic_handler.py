"""
Browser microphone WebSocket handler.
"""
import asyncio
import base64
from dataclasses import dataclass
from typing import Optional
from fastapi import APIRouter, WebSocket
from fastapi.websockets import WebSocketDisconnect

from config import AUDIO_FORMAT_PCM16, RAG_ENABLED
from clients import OpenAIRealtimeClient, QdrantRAGClient

router = APIRouter()


@dataclass
class MicStreamState:
    """State tracking for microphone stream interruption handling."""
    last_assistant_item: Optional[str] = None
    response_start_timestamp: Optional[float] = None
    total_bytes_sent: int = 0
    is_responding: bool = False

    def reset(self):
        """Reset state after interruption or response completion."""
        self.last_assistant_item = None
        self.response_start_timestamp = None
        self.total_bytes_sent = 0
        self.is_responding = False


@router.websocket("/mic-stream")
async def handle_mic_stream(websocket: WebSocket):
    """Handle WebSocket connections between browser microphone and OpenAI."""
    print("Microphone client connected")
    await websocket.accept()

    state = MicStreamState()

    # Initialize Qdrant RAG client if enabled
    qdrant_client = None
    if RAG_ENABLED:
        try:
            qdrant_client = QdrantRAGClient()
            print("Qdrant RAG client initialized")
        except Exception as e:
            print(f"Failed to initialize Qdrant RAG client: {e}")

    async with OpenAIRealtimeClient(audio_format=AUDIO_FORMAT_PCM16, qdrant_client=qdrant_client) as openai_client:

        async def receive_from_client():
            """Receive audio data from the browser client and send it to OpenAI."""
            try:
                async for message in websocket.iter_text():
                    import json
                    data = json.loads(message)
                    if data.get('type') == 'audio' and openai_client.is_open:
                        await openai_client.send_audio(data['data'])
            except WebSocketDisconnect:
                print("Microphone client disconnected.")
                await openai_client.close()

        async def send_to_client():
            """Receive events from OpenAI and send audio back to the browser client."""
            try:
                async for event in openai_client.iter_events():
                    event_type = event.get('type')

                    # Handle audio delta events
                    if event_type == 'response.output_audio.delta' and 'delta' in event:
                        item_id = event.get('item_id')
                        
                        # Track new response starting
                        if item_id and item_id != state.last_assistant_item:
                            print(f"New response started: {item_id}")
                            state.last_assistant_item = item_id
                            state.total_bytes_sent = 0
                            state.is_responding = True
                            
                            # Notify browser that new response started
                            await websocket.send_json({
                                "type": "response_start",
                                "item_id": item_id
                            })
                        
                        # Calculate bytes from base64 delta
                        audio_bytes = len(base64.b64decode(event['delta']))
                        state.total_bytes_sent += audio_bytes
                        
                        # Send audio to browser
                        await websocket.send_json({
                            "type": "audio",
                            "data": event['delta'],
                            "item_id": item_id
                        })

                    # Handle user speech interruption
                    elif event_type == 'input_audio_buffer.speech_started':
                        print("Speech started detected - handling interruption")
                        
                        if state.last_assistant_item and state.is_responding:
                            # Calculate how much audio was sent (in milliseconds)
                            # At 24kHz PCM16: 48 bytes per millisecond
                            audio_end_ms = state.total_bytes_sent // 48
                            
                            print(f"Interrupting response {state.last_assistant_item} at {audio_end_ms}ms")
                            
                            # Truncate the assistant's response at OpenAI
                            await openai_client.send_truncate(state.last_assistant_item, audio_end_ms)
                            
                            # Tell browser to stop playing audio
                            await websocket.send_json({
                                "type": "stop_audio",
                                "audio_end_ms": audio_end_ms
                            })
                            
                            # Reset state to prevent duplicate truncates
                            state.reset()

                    # Handle response completion
                    elif event_type == 'response.done':
                        print("Response completed")
                        state.is_responding = False

            except Exception as e:
                print(f"Error in send_to_client: {e}")

        await asyncio.gather(receive_from_client(), send_to_client())