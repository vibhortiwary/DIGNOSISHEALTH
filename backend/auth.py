from fastapi import APIRouter, HTTPException, Depends, Header
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
import jwt, requests

from sqlalchemy.orm import Session
from backend.database import SessionLocal, User

auth_router = APIRouter()

pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


# ⚠ change to env var in production
SECRET = "SUPER_SECRET_KEY_CHANGE_THIS"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_token(user: User) -> str:
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)


def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = authorization.split(" ", 1)[1]
    data = decode_token(token)
    email = data.get("email")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user



@auth_router.post("/signup")
def signup(data: dict, db: Session = Depends(get_db)):
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        raise HTTPException(400, "Missing email or password")

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(400, "User already exists")

    user = User(
        email=email,
        password=pwd.hash(password),
        provider="local",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {"message": "Account created!"}


@auth_router.post("/login")
def login(data: dict, db: Session = Depends(get_db)):
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        raise HTTPException(400, "Missing email or password")

    user = db.query(User).filter(User.email == email).first()
    if not user or not user.password:
        raise HTTPException(400, "User not found")

    if not pwd.verify(password, user.password):
        raise HTTPException(400, "Incorrect password")

    token = create_token(user)
    return {"access_token": token, "token_type": "bearer"}


@auth_router.post("/google")
def google_login(data: dict, db: Session = Depends(get_db)):
    id_token = data.get("id_token")
    if not id_token:
        raise HTTPException(400, "Missing id_token")

    url = f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
    google_data = requests.get(url).json()

    if "email" not in google_data:
        raise HTTPException(400, "Invalid Google token")

    email = google_data["email"]

    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(
            email=email,
            password=None,
            provider="google",
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    token = create_token(user)
    return {"access_token": token, "token_type": "bearer"}

@auth_router.post("/forgot")
def forgot(data: dict):
    # real app: send email
    return {"message": "If this email exists, a reset link has been sent."}


@auth_router.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "email": current_user.email}
