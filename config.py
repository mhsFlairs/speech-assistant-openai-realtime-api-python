"""
Configuration settings for the Speech Assistant application.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"
OPENAI_MODEL = "gpt-realtime"

# Server Configuration
PORT = int(os.getenv('PORT', 5050))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.8))

# Voice Assistant Configuration
SYSTEM_MESSAGE = (
    "You are a helpful and bubbly AI assistant who loves to chat about "
    "anything the user is interested in and is prepared to offer them facts. "
    "You have a penchant for dad jokes, owl jokes, and rickrolling – subtly. "
    "Always stay positive, but work in a joke when appropriate.\n\n"
    "IMPORTANT: You're in a real-time voice conversation. The user can interrupt you at any time. "
    "When interrupted, immediately stop your current thought and respond to what they just said. "
    "If they say 'Actually', 'Wait', 'Hold on', 'No', or similar phrases, stop and ask what they need. "
    "Be very responsive to corrections, clarifications, and topic changes. Never continue your previous "
    "response after being interrupted - always address the interruption directly."
)
VOICE = 'alloy'

# Audio Formats
AUDIO_FORMAT_TWILIO = "audio/pcmu"  # μ-law 8kHz for Twilio
AUDIO_FORMAT_PCM16 = "audio/pcm16"   # PCM16 24kHz for browser mic

# Voice Activity Detection (VAD) Configuration
VAD_THRESHOLD = float(os.getenv('VAD_THRESHOLD', 0.5))  # 0.0-1.0, sensitivity for speech detection
VAD_PREFIX_PADDING_MS = int(os.getenv('VAD_PREFIX_PADDING_MS', 300))  # Audio to include before speech
VAD_SILENCE_DURATION_MS = int(os.getenv('VAD_SILENCE_DURATION_MS', 200))  # Silence before end-of-turn (lower = more responsive)

# Logging Configuration
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created', 'session.updated'
]
SHOW_TIMING_MATH = False

# Validation
if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')