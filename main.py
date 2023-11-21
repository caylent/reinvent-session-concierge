import json
import logging
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from retriever import question_stream, DateTimeEncoder

log_level = logging.getLevelName(os.getenv("LOG_LEVEL", "DEBUG").upper())
logger = logging.getLogger("question-service")
logger.setLevel(log_level)
handler = logging.StreamHandler()
handler.setLevel(log_level)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class QuestionRequest(BaseModel):
    prompt: str


app = FastAPI(title="re:Invent Question API")

# Configure CORS for the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/question")
async def process_question(request: QuestionRequest):
    async def json_stream():
        for chunk in question_stream(
            request.prompt, retry_on_exception=True, retry_on_no_results=True
        ):
            yield json.dumps(chunk, cls=DateTimeEncoder) + "\n"

    return StreamingResponse(json_stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
