from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.policy_qa.pipeline import RAGPipeline

app = FastAPI(title="Policy Document QA & Extraction Engine", version="1.0.0")
_pipeline = None

class QueryRequest(BaseModel):
    question: str

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline.from_disk()
    return _pipeline

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = get_pipeline().answer(request.question)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
