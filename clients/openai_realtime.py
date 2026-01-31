"""
OpenAI Realtime API client with async context manager support.
"""
import json
import websockets
from websockets import ClientConnection
from typing import AsyncIterator, Optional

from config import (
    OPENAI_API_KEY,
    OPENAI_REALTIME_URL,
    OPENAI_MODEL,
    TEMPERATURE,
    SYSTEM_MESSAGE,
    VOICE,
    LOG_EVENT_TYPES,
    VAD_THRESHOLD,
    VAD_PREFIX_PADDING_MS,
    VAD_SILENCE_DURATION_MS,
)


class OpenAIRealtimeClient:
    """
    Async context manager for OpenAI Realtime API WebSocket connections.

    Usage:
        async with OpenAIRealtimeClient(audio_format="audio/pcm16") as client:
            await client.send_audio(audio_data)
            async for event in client.iter_events():
                handle_event(event)
    """

    def __init__(self, audio_format: str = "audio/pcmu"):
        """
        Initialize the client.

        Args:
            audio_format: Audio format to use ("audio/pcmu" for Twilio, "audio/pcm16" for browser)
        """
        self.audio_format = audio_format
        self._ws: Optional[ClientConnection] = None
        self._url = f"{OPENAI_REALTIME_URL}?model={OPENAI_MODEL}&temperature={TEMPERATURE}"

    async def __aenter__(self) -> "OpenAIRealtimeClient":
        """Connect to OpenAI Realtime API and initialize session."""
        self._ws = await websockets.connect(
            self._url,
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
        await self._initialize_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the WebSocket connection."""
        if self._ws and self._ws.state.name == 'OPEN':
            await self._ws.close()

    @property
    def is_open(self) -> bool:
        """Check if the WebSocket connection is open."""
        return self._ws is not None and self._ws.state.name == 'OPEN'

    async def _initialize_session(self):
        """Configure the OpenAI Realtime session."""
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
            
        session_update = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": OPENAI_MODEL,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": self.audio_format},
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": VAD_THRESHOLD,
                            "prefix_padding_ms": VAD_PREFIX_PADDING_MS,
                            "silence_duration_ms": VAD_SILENCE_DURATION_MS
                        }
                    },
                    "output": {
                        "format": {"type": self.audio_format},
                        "voice": VOICE
                    }
                },
                "instructions": SYSTEM_MESSAGE,
            }
        }
        print('Sending session update:', json.dumps(session_update))
        await self._ws.send(json.dumps(session_update))

    async def send_audio(self, audio_data: str):
        """
        Send audio data to OpenAI.

        Args:
            audio_data: Base64-encoded audio data
        """
        if not self.is_open or self._ws is None:
            return

        audio_append = {
            "type": "input_audio_buffer.append",
            "audio": audio_data
        }
        await self._ws.send(json.dumps(audio_append))

    async def send_truncate(self, item_id: str, audio_end_ms: int):
        """
        Send a truncate event to interrupt the current response.

        Args:
            item_id: The ID of the assistant item to truncate
            audio_end_ms: The timestamp in milliseconds where to truncate
        """
        if not self.is_open or self._ws is None:
            return

        truncate_event = {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": audio_end_ms
        }
        await self._ws.send(json.dumps(truncate_event))

    async def send_initial_greeting(self, greeting_text: Optional[str] = None):
        """
        Send an initial conversation item to make the AI speak first.

        Args:
            greeting_text: Optional custom greeting text
        """
        if not self.is_open or self._ws is None:
            return

        if greeting_text is None:
            greeting_text = (
                "Greet the user with 'Hello there! I am an AI voice assistant powered by "
                "Twilio and the OpenAI Realtime API. You can ask me for facts, jokes, or "
                "anything you can imagine. How can I help you?'"
            )

        initial_conversation_item = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": greeting_text
                    }
                ]
            }
        }
        await self._ws.send(json.dumps(initial_conversation_item))
        await self._ws.send(json.dumps({"type": "response.create"}))

    async def iter_events(self) -> AsyncIterator[dict]:
        """
        Iterate over events received from OpenAI.

        Yields:
            Parsed JSON event dictionaries
        """
        if self._ws is None:
            return
            
        async for message in self._ws:
            event = json.loads(message)

            # Log configured event types
            if event.get('type') in LOG_EVENT_TYPES:
                print(f"Received event: {event['type']}", event)

            yield event

    async def close(self):
        """Manually close the WebSocket connection."""
        if self._ws and self._ws.state.name == 'OPEN':
            await self._ws.close()