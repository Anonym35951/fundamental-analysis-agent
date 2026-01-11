from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.analyze import router as analyze_router
from api.routes.full_analysis import router as full_analysis_router

app = FastAPI(title="AIAgent API", version="0.1")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://fundamental-analysis-agent-6.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Wichtig für Render: schneller Root-Endpoint
@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(analyze_router)
app.include_router(full_analysis_router)