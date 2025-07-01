from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from auth_middleware import verify_token
from hf_proxy import HFProxy
import uvicorn

app = FastAPI(title="🇦🇫 ZamAI API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hf_proxy = HFProxy()

class GenerationRequest(BaseModel):
    prompt: str
    model: str = "tasal9/ZamAI-Mistral-7B-Pashto"
    max_tokens: int = 150

@app.post("/api/generate")
async def generate_text(
    request: GenerationRequest,
    authorization: str = Header(None)
):
    """Generate text using HF models"""
    if not verify_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    try:
        response = hf_proxy.generate(
            model=request.model,
            prompt=request.prompt,
            max_tokens=request.max_tokens
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            "tasal9/ZamAI-Mistral-7B-Pashto",
            "tasal9/ZamAI-LIama3-Pashto",
            "tasal9/pashto-tutor-bot"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
