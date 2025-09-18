import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# يقرأ متغيرات البيئة من Render (أو .env محلياً)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION     = os.getenv("COLLECTION", "Legal-Docs")

if not (OPENAI_API_KEY and QDRANT_URL and QDRANT_API_KEY):
    raise RuntimeError("Missing env vars")

# عملاء OpenAI + Qdrant
oa  = OpenAI(api_key=OPENAI_API_KEY)
qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def embed(text: str):
    # إرجاع متجه 1536 (نفس اللي استخدمناه محلياً)
    resp = oa.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

app = FastAPI(title="legal-assistant-api")

# سماح CORS لأي عميل (Bubble بيستفيد)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# نمط الطلب
class ChatReq(BaseModel):
    query: str
    owner: str = "user_test_001"
    top_k: int = 5

@app.get("/")
def root():
    return {"status": "running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatReq):
    # 1) نحسب embedding للسؤال
    q_vec = embed(req.query)

    # 2) نفلتر على المالك (owner)
    flt = qmodels.Filter(
        must=[qmodels.FieldCondition(
            key="owner",
            match=qmodels.MatchValue(value=req.owner)
        )]
    )

    # 3) بحث في Qdrant
    res = qdr.search(
        collection_name=COLLECTION,
        query_vector=q_vec,
        limit=req.top_k,
        query_filter=flt
    )

    # 4) نرجّع أفضل المقاطع
    contexts = []
    for hit in res:
        # بعض إصدارات العميل ترجع tuple، لذا نعالج الحالتين
        payload = getattr(hit, "payload", {}) or hit[0].payload
        score   = getattr(hit, "score", None) or hit[1]
        contexts.append({
            "text": payload.get("text",""),
            "law_name": payload.get("law_name"),
            "doc_id": payload.get("doc_id"),
            "score": score
        })

    return {"ok": True, "contexts": contexts}
