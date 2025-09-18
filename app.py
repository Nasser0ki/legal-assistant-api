import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from openai import OpenAI

# ========= Env =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION     = os.getenv("COLLECTION", "Legal-Docs")
OWNER_DEFAULT  = os.getenv("OWNER_DEFAULT", "user_test_001")
GEN_MODEL      = os.getenv("GEN_MODEL", "gpt-4o-mini")  # رخيص وممتاز
EMB_MODEL      = os.getenv("EMB_MODEL", "text-embedding-3-small")  # 1536

if not (OPENAI_API_KEY and QDRANT_URL and QDRANT_API_KEY):
    raise RuntimeError("Missing env vars: OPENAI_API_KEY / QDRANT_URL / QDRANT_API_KEY")

# ========= Clients =========
client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ========= FastAPI =========
app = FastAPI(title="Legal Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # افتحها الآن للتجارب، لاحقًا حدّد دومين Bubble
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Schemas =========
class ChatIn(BaseModel):
    query: str
    owner: Optional[str] = None

class Citation(BaseModel):
    score: float
    text: str
    doc_id: Optional[str] = None
    law_name: Optional[str] = None

class ChatOut(BaseModel):
    answer: str
    citations: List[Citation]

# ========= Helpers =========
def embed_text(text: str) -> List[float]:
    resp = client.embeddings.create(model=EMB_MODEL, input=text)
    return resp.data[0].embedding

def search_qdrant(query: str, owner: str, k: int = 5) -> List[Citation]:
    vec = embed_text(query)
    flt = models.Filter(
        must=[models.FieldCondition(key="owner", match=models.MatchValue(value=owner))]
    )
    res = qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=k,
        filter=flt,
        with_payload=True,
        with_vectors=False
    )
    cites: List[Citation] = []
    for p in res.points:
        payload = p.payload or {}
        cites.append(Citation(
            score=float(p.score),
            text=(payload.get("text") or "")[:1200],
            doc_id=payload.get("doc_id"),
            law_name=payload.get("law_name")
        ))
    return cites

def generate_answer(question: str, citations: List[Citation]) -> str:
    # نبني سياق مختصر للـ GPT
    ctx_lines = []
    for i, c in enumerate(citations, 1):
        line = f"[{i}] ({c.law_name or 'غير محدد'}) {c.text}"
        ctx_lines.append(line)
    context = "\n\n".join(ctx_lines)

    prompt = f"""
أجب بالعربية الفصحى باختصار ودقة. اعتمد فقط على المقتطفات التالية من الأنظمة السعودية،
واذكر أرقام المراجع بهذا الشكل [1][2] داخل الجواب حيث يلزم.

السؤال:
{question}

المقتطفات:
{context}

إن لم تجد إجابة صريحة، قل: "المقتطفات المتاحة لا تكفي لإجابة دقيقة" واقترح ما يجب البحث عنه.
""".strip()

    chat = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": "أنت مساعد قانوني متخصص في الأنظمة السعودية."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return chat.choices[0].message.content.strip()

# ========= Routes =========
@app.get("/health")
def health():
    return {"status": "ok", "collection": COLLECTION}

@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn):
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    owner = body.owner or OWNER_DEFAULT
    citations = search_qdrant(body.query, owner, k=5)
    answer = generate_answer(body.query, citations)
    return ChatOut(answer=answer, citations=citations)
