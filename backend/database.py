from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# SQLite DB in backend folder
DATABASE_URL = "sqlite:///./backend/diagnostics.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=True)          # null = google users
    provider = Column(String, default="local")        # local / google


class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)             # FK -> users.id (logical)
    disease = Column(String, nullable=False)
    input_data = Column(Text)                         # JSON string
    prediction = Column(String)
    probability = Column(String)
    report_path = Column(String)                      # /reports/...
    gradcam_path = Column(String)                     # /gradcam/...
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
