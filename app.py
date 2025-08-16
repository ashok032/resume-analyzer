# app.py
import streamlit as st
import pandas as pd
import os
import hashlib
import pdfplumber
import docx
import re
import spacy
import nltk
from sentence_transformers import SentenceTransformer, util
from supabase import create_client, Client

# ---------------------------
# Supabase setup
# ---------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# Load NLP Models
# ---------------------------
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
nltk.download('punkt')

# ---------------------------
# Utility Functions
# ---------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------------------
# Supabase Auth Functions
# ---------------------------
def register_user(email, password, role):
    existing = supabase.table("users").select("*").eq("email", email).execute()
    if existing.data:
        return False, "Email already registered."
    
    hashed_password = hash_password(password)
    supabase.table("users").insert({
        "email": email,
        "password_hash": hashed_password,
        "role": role
    }).execute()
    
    return True, "Registration successful!"

def login_user(email, password):
    hashed_password = hash_password(password)
    response = supabase.table("users").select("*").eq("email", email).execute()
    user_data = response.data
    if not user_data:
        return False, "Email not found."
    user = user_data[0]
    if user["password_hash"] != hashed_password:
        return False, "Incorrect password."
    return True, user

# ---------------------------
# Resume Parsing
# ---------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_name_email(file_text):
    # Name using spaCy
    doc = nlp(file_text)
    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break
    # Email regex
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', file_text)
    email = email_match.group(0) if email_match else None
    return name, email

# ---------------------------
# Skill Matching
# ---------------------------
def extract_keywords(text):
    words = nltk.word_tokenize(text.lower())
    words = [w for w in words if w.isalpha()]
    return list(set(words))

def match_resume_to_job(resume_keywords, job_keywords):
    if not job_keywords:
        return [], [], 0
    resume_emb = model.encode(resume_keywords, convert_to_tensor=True)
    job_emb = model.encode(job_keywords, convert_to_tensor=True)
    cosine_scores = util.cos_sim(resume_emb, job_emb)
    matched = []
    missing = []
    for i, jk in enumerate(job_keywords):
        if max(cosine_scores[:, i]).item() > 0.7:
            matched.append(jk)
        else:
            missing.append(jk)
    score = int((len(matched)/len(job_keywords))*100) if job_keywords else 0
    return matched, missing, score

# ---------------------------
# Supabase Application Functions
# ---------------------------
def submit_application(user_id, job_role, resume_url, match_score):
    supabase.table("applications").insert({
        "user_id": user_id,
        "job_role": job_role,
        "resume_url": resume_url,
        "match_score": match_score,
        "phase": "Round 1"
    }).execute()

def get_user_applications(user_id):
    response = supabase.table("applications").select("*").eq("user_id", user_id).execute()
    return response.data

def get_all_applications():
    response = supabase.table("applications").select("*").execute()
    return response.data

def update_application_phase(application_id, new_phase):
    supabase.table("applications").update({"phase": new_phase}).eq("id", application_id).execute()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Sidebar for login/register
st.sidebar.title("Resume Analyzer")
tabs = st.sidebar.radio("Navigation", ["Login", "Register"])

if tabs == "Register":
    st.header("Register")
    reg_email = st.text_input("Email")
    reg_password = st.text_input("Password", type="password")
    reg_role = st.selectbox("Role", ["User", "HR"])
    if st.button("Register"):
        success, msg = register_user(reg_email, reg_password, reg_role)
        st.success(msg) if success else st.error(msg)

elif tabs == "Login":
    st.header("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        success, result = login_user(email, password)
        if success:
            st.session_state["user"] = result
            st.success(f"Logged in as {result['email']} ({result['role']})")
        else:
            st.error(result)

# ---------------------------
# Main App after login
# ---------------------------
if "user" in st.session_state:
    user = st.session_state["user"]
    st.sidebar.write(f"Logged in as: {user['email']}")
    if st.sidebar.button("Logout"):
        del st.session_state["user"]
        st.experimental_rerun()
    
    if user["role"] == "User":
        st.header("User Dashboard")
        job_role = st.selectbox("Select Job Role", ["Data Analyst", "Software Engineer", "ML Engineer"])
        uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
        if uploaded_file:
            file_text = ""
            if uploaded_file.type == "application/pdf":
                file_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                file_text = extract_text_from_docx(uploaded_file)
            name, email_in_resume = extract_name_email(file_text)
            st.write(f"Detected Name: {name}, Email: {email_in_resume}")
            
            resume_keywords = extract_keywords(file_text)
            # Define job skills
            job_skills_dict = {
                "Data Analyst": ["python", "sql", "excel", "tableau", "statistics"],
                "Software Engineer": ["python", "java", "git", "oop", "rest api"],
                "ML Engineer": ["python", "tensorflow", "pytorch", "ml", "data preprocessing"]
            }
            jd_skills = job_skills_dict.get(job_role, [])
            matched, missing, score = match_resume_to_job(resume_keywords, jd_skills)
            st.write(f"Match Score: {score}%")
            st.write(f"Matched Skills: {matched}")
            st.write(f"Missing Skills: {missing}")
            
            if st.button("Submit Application"):
                # Upload file to local folder or Supabase storage (here simplified as local path)
                save_path = f"resumes/{uploaded_file.name}"
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                submit_application(user["id"], job_role, save_path, score)
                st.success("Application submitted successfully!")
        
        st.subheader("Your Applications")
        apps = get_user_applications(user["id"])
        if apps:
            for app in apps:
                st.write(f"Job: {app['job_role']}, Score: {app['match_score']}%, Phase: {app['phase']}")
        else:
            st.info("No applications submitted yet.")
    
    elif user["role"] == "HR":
        st.header("HR Dashboard")
        apps = get_all_applications()
        if apps:
            for app in apps:
                st.write(f"Candidate ID: {app['user_id']}, Job: {app['job_role']}, Score: {app['match_score']}%, Phase: {app['phase']}")
                next_phase = st.selectbox(f"Update Phase for Candidate {app['user_id']}", ["Round 1", "Round 2", "Final", "Rejected"], key=app['id'])
                if st.button(f"Update Phase {app['user_id']}", key=f"btn_{app['id']}"):
                    update_application_phase(app['id'], next_phase)
                    st.success(f"Phase updated to {next_phase}")
        else:
            st.info("No applications submitted yet.")
