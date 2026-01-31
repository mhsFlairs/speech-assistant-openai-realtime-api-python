"""
Twilio Media Stream WebSocket handler.
"""
import asyncio
import base64
from fastapi import APIRouter, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect
from typing import Optional

from config import SHOW_TIMING_MATH, AUDIO_FORMAT_TWILIO
from clients import OpenAIRealtimeClient

router = APIRouter()


@router.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    response.say(
        "Please wait while we connect your call to the A. I. voice assistant, powered by Twilio and the Open A I Realtime API",
        voice="Google.en-US-Chirp3-HD-Aoede"
    )
    response.pause(length=1)
    response.say(
        "O.K. you can start talking!",
        voice="Google.en-US-Chirp3-HD-Aoede"
    )
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")


@router.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Twilio client connected")
    await websocket.accept()

    async with OpenAIRealtimeClient(audio_format=AUDIO_FORMAT_TWILIO) as openai_client:
        # Connection specific state
        state = TwilioStreamState()

        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to OpenAI."""
            try:
                async for message in websocket.iter_text():
                    await state.handle_twilio_message(message, openai_client)
            except WebSocketDisconnect:
                print("Twilio client disconnected.")
                await openai_client.close()

        async def send_to_twilio():
            """Receive events from OpenAI and send audio back to Twilio."""
            try:
                async for event in openai_client.iter_events():
                    await state.handle_openai_event(event, websocket, openai_client)
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        await asyncio.gather(receive_from_twilio(), send_to_twilio())


class TwilioStreamState:
    """Manages state for a Twilio Media Stream connection."""

    def __init__(self):
        self.stream_sid: Optional[str] = None
        self.latest_media_timestamp: int = 0
        self.last_assistant_item: Optional[str] = None
        self.mark_queue: list = []
        self.response_start_timestamp_twilio: Optional[int] = None

    async def handle_twilio_message(self, message: str, openai_client: OpenAIRealtimeClient):
        """Process incoming Twilio Media Stream messages."""
        import json
        data = json.loads(message)

        if data['event'] == 'media' and openai_client.is_open:
            self.latest_media_timestamp = int(data['media']['timestamp'])
            await openai_client.send_audio(data['media']['payload'])

        elif data['event'] == 'start':
            self.stream_sid = data['start']['streamSid']
            print(f"Incoming stream has started {self.stream_sid}")
            self.response_start_timestamp_twilio = None
            self.latest_media_timestamp = 0
            self.last_assistant_item = None

        elif data['event'] == 'mark':
            if self.mark_queue:
                self.mark_queue.pop(0)

    async def handle_openai_event(self, event: dict, websocket: WebSocket, openai_client: OpenAIRealtimeClient):
        """Process OpenAI events and send responses to Twilio."""
        if event.get('type') == 'response.output_audio.delta' and 'delta' in event:
            # Forward audio to Twilio
            audio_payload = base64.b64encode(base64.b64decode(event['delta'])).decode('utf-8')
            audio_delta = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {
                    "payload": audio_payload
                }
            }
            await websocket.send_json(audio_delta)

            # Track response timing for interruption handling
            if event.get("item_id") and event["item_id"] != self.last_assistant_item:
                self.response_start_timestamp_twilio = self.latest_media_timestamp
                self.last_assistant_item = event["item_id"]
                if SHOW_TIMING_MATH:
                    print(f"Setting start timestamp for new response: {self.response_start_timestamp_twilio}ms")

            await self._send_mark(websocket)

        # Handle interruption
        if event.get('type') == 'input_audio_buffer.speech_started':
            print("Speech started detected.")
            if self.last_assistant_item:
                print(f"Interrupting response with id: {self.last_assistant_item}")
                await self._handle_interruption(websocket, openai_client)

    async def _send_mark(self, websocket: WebSocket):
        """Send a mark event to Twilio for timing synchronization."""
        if self.stream_sid:
            mark_event = {
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {"name": "responsePart"}
            }
            await websocket.send_json(mark_event)
            self.mark_queue.append('responsePart')

    async def _handle_interruption(self, websocket: WebSocket, openai_client: OpenAIRealtimeClient):
        """Handle interruption when the caller's speech starts."""
        print("Handling speech started event.")

        if self.mark_queue and self.response_start_timestamp_twilio is not None:
            elapsed_time = self.latest_media_timestamp - self.response_start_timestamp_twilio
            if SHOW_TIMING_MATH:
                print(f"Calculating elapsed time for truncation: {self.latest_media_timestamp} - {self.response_start_timestamp_twilio} = {elapsed_time}ms")

            if self.last_assistant_item:
                if SHOW_TIMING_MATH:
                    print(f"Truncating item with ID: {self.last_assistant_item}, Truncated at: {elapsed_time}ms")
                await openai_client.send_truncate(self.last_assistant_item, elapsed_time)

            # Send clear event to Twilio
            await websocket.send_json({
                "event": "clear",
                "streamSid": self.stream_sid
            })

            self.mark_queue.clear()
            self.last_assistant_item = None
            self.response_start_timestamp_twilio = None