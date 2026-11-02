# CloudSec Knowledge Graph Assistant: FastAPI service (Module 10 In-Class Activity).
# Optional API alongside the Streamlit app.
# Run: pip install fastapi uvicorn ; uvicorn cloudsec_assistant_service:app --reload
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import kg_rag                               # the engine: load data, build indexes, ask()

app = FastAPI(title="CloudSec Knowledge Graph Assistant", version="1.0.0")
ENGINE = kg_rag.Engine()                    # loads the knowledge base and builds the indexes once

class AskRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok", "runbooks": ENGINE.n_docs}

@app.post("/ask", response_model=kg_rag.AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    return ENGINE.ask(req.question)
