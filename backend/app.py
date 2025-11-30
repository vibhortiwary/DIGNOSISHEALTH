from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.auth import auth_router
from backend.predict_routes import predict_router
from backend.history_routes import history_router
from backend.database import init_db

app = FastAPI(title="AI Diagnostics Backend")

# CORS


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:3002",
        "http://localhost:3002",
        "http://127.0.0.1:5500",
        "http://localhost:5500",


    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],
)



# Static mounts
app.mount("/reports", StaticFiles(directory="backend/reports"), name="reports")
app.mount("/gradcam", StaticFiles(directory="backend/gradcam"), name="gradcam")

# 🔥 FIX: Serve frontend at /app instead of /
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

# Routers
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(predict_router, prefix="/predict", tags=["predict"])
app.include_router(history_router, prefix="/history", tags=["history"])

# DB init
init_db()

@app.get("/api")
def root():
    return {"message": "AI Diagnostics Backend Online"}
