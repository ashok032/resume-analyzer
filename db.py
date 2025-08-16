# db.py
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("DATABASE_URL not found in .env file")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# -------------------- TABLES --------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)

class Application(Base):
    __tablename__ = "applications"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    candidate_name = Column(String)
    candidate_email = Column(String)
    phone = Column(String)
    skills = Column(Text)
    job_role = Column(String)
    match_score = Column(Float)
    phase = Column(String, default="Not Selected")
    uploaded_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# -------------------- FUNCTIONS --------------------
def register_user(username, password, role):
    from hashlib import sha256
    hashed_pw = sha256(password.encode()).hexdigest()
    session = SessionLocal()
    exists = session.query(User).filter_by(username=username).first()
    if exists:
        session.close()
        return False
    user = User(username=username, password_hash=hashed_pw, role=role)
    session.add(user)
    session.commit()
    session.close()
    return True

def login_user(username, password):
    from hashlib import sha256
    hashed_pw = sha256(password.encode()).hexdigest()
    session = SessionLocal()
    user = session.query(User).filter_by(username=username, password_hash=hashed_pw).first()
    if user:
        session.close()
        return True, user.role, user.id
    session.close()
    return False, None, None

def create_application(user_id, candidate_name, candidate_email, phone, skills, job_role, match_score):
    session = SessionLocal()
    app = Application(
        user_id=user_id,
        candidate_name=candidate_name,
        candidate_email=candidate_email,
        phone=phone,
        skills=",".join(skills) if isinstance(skills, list) else skills,
        job_role=job_role,
        match_score=match_score,
        phase="Round 1" if match_score > 70 else "Rejected"
    )
    session.add(app)
    session.commit()
    app_id = app.id
    session.close()
    return app_id

def list_user_applications(user_id):
    session = SessionLocal()
    apps = session.query(Application).filter_by(user_id=user_id).all()
    session.close()
    return [
        {
            "id": a.id,
            "candidate_name": a.candidate_name,
            "candidate_email": a.candidate_email,
            "job_role": a.job_role,
            "match_score": a.match_score,
            "phase": a.phase,
            "uploaded_at": a.uploaded_at.strftime("%Y-%m-%d %H:%M")
        } for a in apps
    ]

def list_all_applications_for_hr():
    session = SessionLocal()
    apps = session.query(Application).all()
    session.close()
    return [
        {
            "id": a.id,
            "candidate_name": a.candidate_name,
            "candidate_email": a.candidate_email,
            "job_role": a.job_role,
            "match_score": a.match_score,
            "phase": a.phase
        } for a in apps
    ]

def update_phase(application_id, new_phase):
    session = SessionLocal()
    app = session.query(Application).filter_by(id=application_id).first()
    if app:
        app.phase = new_phase
        session.commit()
    session.close()

def get_user_id_by_email(email):
    session = SessionLocal()
    user = session.query(User).filter_by(username=email).first()
    session.close()
    if user:
        return user.id
    return None
