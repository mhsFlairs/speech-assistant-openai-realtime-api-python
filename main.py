"""
Speech Assistant Server - FastAPI entry point.

Supports two modes:
- Twilio Media Stream: Phone calls via /incoming-call and /media-stream
- Browser Microphone: Direct testing via /mic-stream
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import PORT
from handlers import twilio_router, mic_router

app = FastAPI(title="Speech Assistant Server")

# Register route handlers
app.include_router(twilio_router)
app.include_router(mic_router)


@app.get("/", response_class=HTMLResponse)
async def index_page():
    """Landing page with links to both modes."""
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Speech Assistant Server</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    background-color: white;
                    border-radius: 8px;
                    padding: 30px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                    border-bottom: 3px solid #0066cc;
                    padding-bottom: 10px;
                }
                h2 {
                    color: #0066cc;
                    margin-top: 30px;
                }
                .mode {
                    background-color: #f9f9f9;
                    border-left: 4px solid #0066cc;
                    padding: 15px;
                    margin: 15px 0;
                }
                a {
                    color: #0066cc;
                    text-decoration: none;
                    font-weight: bold;
                }
                a:hover {
                    text-decoration: underline;
                }
                code {
                    background-color: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎙️ Speech Assistant Server</h1>
                <p>Server is running! Choose your mode:</p>

                <h2>🌐 Browser Microphone Mode</h2>
                <div class="mode">
                    <p>Test locally using your computer's microphone directly in your browser.</p>
                    <p><a href="/static/mic_client.html">→ Open Microphone Client</a></p>
                    <p><small>Uses WebSocket connection with PCM16 @ 24kHz audio format</small></p>
                </div>

                <h2>📞 Twilio Phone Mode</h2>
                <div class="mode">
                    <p>Connect via Twilio phone calls using the Media Stream API.</p>
                    <p><strong>Endpoints:</strong></p>
                    <ul>
                        <li><code>POST /incoming-call</code> - TwiML webhook for incoming calls</li>
                        <li><code>wss://your-domain/media-stream</code> - WebSocket for Twilio Media Streams</li>
                    </ul>
                    <p><small>Uses PCMU (μ-law) @ 8kHz audio format</small></p>
                </div>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# Mount static files for browser client
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    # Directory doesn't exist yet
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
