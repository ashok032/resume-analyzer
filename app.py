import streamlit as st
import pandas as pd
import hashlib
import os
import pdfplumber
import docx
import re
import spacy
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer, util

# ================== CONFIG ================== #
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Supabase connection
supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# Gmail SMTP details
SMTP_EMAIL = st.secrets["SMTP_EMAIL"]
SMTP_PASSWORD = st.secrets["SMTP_PASSWORD"]

# ================== HELPERS ================== #
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def send_email(to_email, subject, body):
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = SMTP_EMAIL
        msg["To"] = to_email

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, [to_email], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False

def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

def extract_name_email_phone(text):
    name = "Unknown"
    email = None
    phone = None

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if email_match:
        email = email_match.group(0)

    phone_match = re.search(r"\+?\d[\d -]{8,}\d", text)
    if phone_match:
        phone = phone_match.group(0)

    return name, email, phone

def skill_match(resume_text, jd_skills):
    resume_doc = nlp(resume_text.lower())
    resume_tokens = set([token.text for token in resume_doc if token.is_alpha])

    matched = []
    missing = []
    score = 0

    for skill in jd_skills:
        skill_embedding = model.encode(skill, convert_to_tensor=True)
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        similarity = util.cos_sim(skill_embedding, resume_embedding).item()

        if similarity > 0.5:
            matched.append(skill)
        else:
            missing.append(skill)

    if jd_skills:
        score = int((len(matched) / len(jd_skills)) * 100)

    return matched, missing, score

def update_application_phase(app_id, candidate_email, current_round, result):
    if result == "Pass":
        if current_round == "Round 1":
            next_round = "Round 2"
            send_email(candidate_email, "Interview Update", "Congrats! You passed Round 1. Round 2 scheduled.")
        elif current_round == "Round 2":
            next_round = "Final Round"
            send_email(candidate_email, "Interview Update", "Congrats! You passed Round 2. Final round scheduled.")
        elif current_round == "Final Round":
            next_round = "Selected"
            send_email(candidate_email, "Offer", "Congratulations! You have been selected.")
        supabase.table("applications").update({"phase": next_round}).eq("id", app_id).execute()
    else:
        supabase.table("applications").update({"phase": "Rejected"}).eq("id", app_id).execute()
        send_email(candidate_email, "Application Update", "We regret to inform you that you were not selected.")

# ================== STREAMLIT UI ================== #
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("📄 AI Resume Analyzer")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register":
    st.subheader("Create a New Account")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    role = st.selectbox("Role", ["User", "HR"])
    if st.button("Register"):
        hashed_pw = hash_password(password)
        supabase.table("users").insert({"email": email, "password": hashed_pw, "role": role}).execute()
        st.success("Account created! Please login.")

elif choice == "Login":
    st.subheader("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        hashed_pw = hash_password(password)
        result = supabase.table("users").select("*").eq("email", email).eq("password", hashed_pw).execute()
        if result.data:
            role = result.data[0]["role"]
            st.session_state["email"] = email
            st.session_state["role"] = role
            st.success(f"Logged in as {role}")
        else:
            st.error("Invalid login credentials")

if "email" in st.session_state:
    st.sidebar.write(f"👤 {st.session_state['email']}")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    if st.session_state["role"] == "User":
        st.subheader("Upload Resume")
        uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
        job_role = st.text_input("Job Role Applying For")
        if uploaded_file and st.button("Submit"):
            text = extract_text_from_file(uploaded_file)
            name, candidate_email, phone = extract_name_email_phone(text)
            jd_skills = ["python", "machine learning", "nlp", "sql"]  # Example skills
            matched, missing, score = skill_match(text, jd_skills)

            phase = "Round 1" if score > 70 else "Rejected"
            supabase.table("applications").insert({
                "name": name,
                "email": candidate_email,
                "job_role": job_role,
                "score": score,
                "phase": phase
            }).execute()

            if phase == "Round 1":
                send_email(candidate_email, "Interview Scheduled", "Your Round 1 interview has been scheduled.")
            else:
                send_email(candidate_email, "Application Update", "We regret to inform you that you were not selected.")

            st.success(f"Application submitted! Current phase: {phase}")

    elif st.session_state["role"] == "HR":
        st.subheader("HR Dashboard")
        apps = supabase.table("applications").select("*").execute()
        df = pd.DataFrame(apps.data)
        st.dataframe(df)

        app_id = st.selectbox("Select Application ID to Update", df["id"] if not df.empty else [])
        if app_id:
            current_phase = df[df["id"] == app_id]["phase"].values[0]
            candidate_email = df[df["id"] == app_id]["email"].values[0]
            st.write(f"Current Phase: {current_phase}")
            result = st.radio("Mark Result", ["Pass", "Fail"])
            if st.button("Update Phase"):
                update_application_phase(app_id, candidate_email, current_phase, result)
                st.success("Application updated!")
                st.rerun()
