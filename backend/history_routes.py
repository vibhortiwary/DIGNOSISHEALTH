# history_routes.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import os   # <-- FIXED: Required for file deletion

from backend.database import SessionLocal, History, User
from backend.auth import get_current_user

history_router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@history_router.get("/list")
def get_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    records: List[History] = (
        db.query(History)
        .filter(History.user_id == current_user.id)
        .order_by(History.created_at.desc())
        .all()
    )

    return [
        {
            "id": r.id,
            "disease": r.disease,
            "input_data": r.input_data,
            "prediction": r.prediction,
            "probability": r.probability,
            "report_url": r.report_path,
            "gradcam_url": r.gradcam_path,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]


@history_router.get("/{record_id}")
def get_history_item(
    record_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rec = (
        db.query(History)
        .filter(History.id == record_id, History.user_id == current_user.id)
        .first()
    )
    if not rec:
        raise HTTPException(status_code=404, detail="Record not found")

    return {
        "id": rec.id,
        "disease": rec.disease,
        "input_data": rec.input_data,
        "prediction": rec.prediction,
        "probability": rec.probability,
        "report_url": rec.report_path,
        "gradcam_url": rec.gradcam_path,
        "created_at": rec.created_at.isoformat() if rec.created_at else None,
    }


@history_router.delete("/{record_id}")
def delete_history_item(
    record_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rec = (
        db.query(History)
        .filter(History.id == record_id, History.user_id == current_user.id)
        .first()
    )

    if not rec:
        raise HTTPException(status_code=404, detail="Record not found")

    # Attempt file deletion (safe)
    try:
        if rec.report_path:
            path = "." + rec.report_path
            if os.path.exists(path):
                os.remove(path)

        if rec.gradcam_path:
            path = "." + rec.gradcam_path
            if os.path.exists(path):
                os.remove(path)

    except Exception as e:
        print("File delete error:", e)

    # Delete DB record
    db.delete(rec)
    db.commit()

    return {"message": "Deleted successfully!"}
