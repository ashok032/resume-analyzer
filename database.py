import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# This line reads the secret connection key we got from Neon.
# The cloud server will provide this key to our app later.
DATABASE_URL = os.environ.get("DATABASE_URL")

# Set up the main connection engine to the database
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# This class defines the structure of the "users" table in our database.
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String)

# This class defines the structure of the "candidate_progress" table.
class CandidateProgress(Base):
    __tablename__ = "candidate_progress"
    id = Column(Integer, primary_key=True, index=True)
    logged_in_username = Column(String)
    email = Column(String, index=True)
    name = Column(String)
    role = Column(String)
    company = Column(String)
    match_score = Column(Float)
    current_phase = Column(String)
    status = Column(String)

# This function creates the tables in our Neon database if they don't already exist.
# It runs the first time our app starts.
def init_db():
    Base.metadata.create_all(bind=engine)