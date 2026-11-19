import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import init_db, get_db, SupportTicket
from rag_engine import TwinRepRAG, MANUALS_DIR

load_dotenv()

LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8001/v1")
LLM_SERVER_ROOT = os.getenv("LLM_SERVER_ROOT", "http://localhost:8001")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instruct")

rag = TwinRepRAG()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="TwinRep AI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    customer_name: str = "Valued Partner"
    question: str

class EditTicketRequest(BaseModel):
    edited_response: str

class PersonaScaleRequest(BaseModel):
    scale: float  # 0.0 = Generic Base LLM, 1.0 = Full Cloned Manager Persona

@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest, db: AsyncSession = Depends(get_db)):
    # 1. Retrieve top matching technical manual & historical QA chunks
    matches = rag.retrieve(payload.question, k=4)
    context_block = "\n\n".join(
        f"[Source: {m.get('source', 'Verified KB')}]\n{m['content']}" for m in matches
    ) or "No matching manual excerpt found."

    system_prompt = (
        "You are the Senior After-Sales Service Manager. Answer the customer's technical "
        "question in your natural, direct, and experienced tone. Base your technical facts "
        "strictly on the VERIFIED KNOWLEDGE BASE below. If the exact procedure is not in the "
        "knowledge base, clearly state that you are escalating the ticket to Level-2 Engineering "
        "rather than guessing.\n\n"
        f"VERIFIED KNOWLEDGE BASE:\n{context_block}"
    )

    # 2. Query local llama.cpp server (with LoRA persona active)
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            resp = await client.post(
                f"{LLM_API_BASE}/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": payload.question},
                    ],
                    "temperature": 0.25,
                    "max_tokens": 512,
                },
            )
            resp.raise_for_status()
            ai_draft = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM Server error: {e}")

    # 3. Log ticket in MySQL for Manager Review Loop
    ticket = SupportTicket(
        customer_name=payload.customer_name,
        question=payload.question,
        retrieved_sources=json.dumps(matches),
        ai_draft=ai_draft,
        status="pending_review",
    )
    db.add(ticket)
    await db.commit()
    await db.refresh(ticket)

    return {
        "ticket_id": ticket.id,
        "ai_draft": ai_draft,
        "sources": matches,
        "status": ticket.status,
    }

@app.post("/api/upload")
async def upload_manual(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, and .md files are supported.")
    dest = MANUALS_DIR / file.filename
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    chunks_added = rag.ingest_file(dest)
    return {"filename": file.filename, "chunks_indexed": chunks_added}

@app.post("/api/ingest-folder")
async def ingest_folder():
    total = rag.ingest_existing_manuals()
    return {"total_chunks_indexed": total}

@app.get("/api/tickets")
async def list_tickets(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(SupportTicket).order_by(desc(SupportTicket.id)))
    tickets = result.scalars().all()
    return [
        {
            "id": t.id,
            "customer_name": t.customer_name,
            "question": t.question,
            "sources": json.loads(t.retrieved_sources or "[]"),
            "ai_draft": t.ai_draft,
            "final_response": t.final_response,
            "status": t.status,
            "created_at": t.created_at.isoformat(),
        }
        for t in tickets
    ]

@app.post("/api/tickets/{ticket_id}/approve")
async def approve_ticket(ticket_id: int, db: AsyncSession = Depends(get_db)):
    ticket = await db.get(SupportTicket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    ticket.status = "approved"
    ticket.final_response = ticket.ai_draft
    await db.commit()

    # Index verified Q&A into FAISS
    rag.add_verified_resolution(ticket.id, ticket.question, ticket.final_response)
    return {"status": "approved", "ticket_id": ticket.id}

@app.post("/api/tickets/{ticket_id}/edit")
async def edit_and_teach_ticket(ticket_id: int, payload: EditTicketRequest, db: AsyncSession = Depends(get_db)):
    ticket = await db.get(SupportTicket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    ticket.status = "edited"
    ticket.final_response = payload.edited_response.strip()
    await db.commit()

    # Self-Updating Pipeline: immediately vectorize manager's correction
    rag.add_verified_resolution(ticket.id, ticket.question, ticket.final_response)
    return {"status": "edited_and_indexed", "ticket_id": ticket.id}

@app.post("/api/persona/scale")
async def set_persona_scale(payload: PersonaScaleRequest):
    """Hot-swaps LoRA adapter weight on llama-server between 0.0 (Generic) and 1.0 (Cloned Manager)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(f"{LLM_SERVER_ROOT}/lora-adapters", json=[{"id": 0, "scale": payload.scale}])
            return {"scale": payload.scale, "llama_server_status": r.status_code}
        except Exception as e:
            return {"scale": payload.scale, "warning": f"Could not toggle LoRA dynamically: {e}"}

# Serve Frontend HTML/CSS/JS at root URL
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
