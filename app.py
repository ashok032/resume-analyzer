import streamlit as st
import pandas as pd
import hashlib
import os
import pdfplumber
import docx
import re
import spacy
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from job_descriptions import job_descriptions
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
from dotenv import load_dotenv

# ------------------- LOAD ENV VARIABLES -------------------
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

# ------------------- LOAD MODELS -------------------
nlp = spacy.load("en_core_web_sm")   # ✅ fixed model load
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------- HELPER FUNCTIONS -------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_resume_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    else:
        return ""

def extract_name_email_phone(text):
    doc = nlp(text)
    name = None
    email = None
    phone = None

    # Extract Name (first PERSON entity or heuristic)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            break

    if not name:
        first_line = text.strip().split("\n")[0]
        if first_line.isupper() or len(first_line.split()) <= 4:
            name = first_line.strip()

    # Extract Email
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    if email_match:
        email = email_match.group(0)

    # Extract Phone
    phone_match = re.search(r"\+?\d[\d -]{8,}\d", text)
    if phone_match:
        phone = phone_match.group(0)

    return name, email, phone

def match_resume_to_job(resume_text, jd_skills):
    resume_doc = nlp(resume_text.lower())
    resume_tokens = list(set([token.lemma_ for token in resume_doc if not token.is_stop and token.is_alpha]))

    jd_embeddings = model.encode(jd_skills, convert_to_tensor=True)
    resume_embeddings = model.encode(resume_tokens, convert_to_tensor=True)

    cosine_scores = util.cos_sim(resume_embeddings, jd_embeddings)
    matched = []
    missing = []

    for i, skill in enumerate(jd_skills):
        scores = cosine_scores[:, i]
        best_score = float(scores.max())
        if best_score > 0.6:
            matched.append(skill)
        else:
            missing.append(skill)

    score = round((len(matched) / len(jd_skills)) * 100, 2) if jd_skills else 0
    return matched, missing, score

def send_email(to_email, subject, body):
    try:
        sender_email = os.getenv("EMAIL_USER")
        sender_pass = os.getenv("EMAIL_PASS")
        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_pass)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print("Email send failed:", e)

def save_application(email, role, name, score, phase="Applied"):
    supabase.table("applications").upsert({
        "email": email,
        "role": role,
        "name": name,
        "score": score,
        "phase": phase,
        "updated_at": datetime.utcnow().isoformat()
    }).execute()

def get_applications(email=None):
    if email:
        return supabase.table("applications").select("*").eq("email", email).execute().data
    return supabase.table("applications").select("*").execute().data

def update_phase(email, role, new_phase):
    supabase.table("applications").update({"phase": new_phase}).eq("email", email).eq("role", role).execute()

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "email" not in st.session_state:
    st.session_state.email = None

# Sidebar for login info
with st.sidebar:
    if st.session_state.logged_in:
        st.write(f"👤 Logged in as: {st.session_state.email}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.session_state.email = None
            st.rerun()

# Login & Registration
if not st.session_state.logged_in:
    tabs = st.tabs(["Login", "Register"])

    with tabs[0]:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            res = supabase.table("users").select("*").eq("email", email).execute()
            if res.data and res.data[0]["password"] == hash_password(password):
                st.session_state.logged_in = True
                st.session_state.role = res.data[0]["role"]
                st.session_state.email = email
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tabs[1]:
        st.subheader("Register")
        email = st.text_input("Email", key="reg_email")
        password = st.text_input("Password", type="password", key="reg_pass")
        role = st.selectbox("Role", ["User", "HR"], key="reg_role")
        if st.button("Register"):
            res = supabase.table("users").select("*").eq("email", email).execute()
            if res.data:
                st.error("Email already registered.")
            else:
                supabase.table("users").insert({
                    "email": email,
                    "password": hash_password(password),
                    "role": role
                }).execute()
                st.success("Registration successful! Please login.")

else:
    # --------- USER VIEW ---------
    if st.session_state.role == "User":
        st.header("📄 Candidate Dashboard")

        uploaded_file = st.file_uploader("Upload your Resume (PDF/DOCX)", type=["pdf", "docx"])
        job_role = st.selectbox("Select Job Role", list(job_descriptions.keys()))

        if uploaded_file and job_role:
            resume_text = extract_resume_text(uploaded_file)
            name, email, phone = extract_name_email_phone(resume_text)
            jd_skills = job_descriptions[job_role]["skills"]
            matched, missing, score = match_resume_to_job(resume_text, jd_skills)

            st.subheader("Analysis Result")
            st.write(f"**Name:** {name}")
            st.write(f"**Email:** {email}")
            st.write(f"**Phone:** {phone}")
            st.write(f"**Match Score:** {score}%")
            st.success(f"✅ Matched Skills: {', '.join(matched)}")
            st.error(f"❌ Missing Skills: {', '.join(missing)}")

            save_application(email, job_role, name, score)

        st.subheader("📊 Your Applications")
        apps = get_applications(st.session_state.email)
        for app in apps:
            st.write(f"**Role:** {app['role']} | **Score:** {app['score']}% | **Phase:** {app['phase']}")

    # --------- HR VIEW ---------
    elif st.session_state.role == "HR":
        st.header("🧑‍💼 HR Dashboard")

        apps = get_applications()
        df = pd.DataFrame(apps)
        st.dataframe(df)

        for app in apps:
            st.write(f"### {app['name']} - {app['role']}")
            st.write(f"📧 {app['email']} | Score: {app['score']}% | Phase: {app['phase']}")

            if app["phase"] == "Applied" and app["score"] > 70:
                if st.button(f"Schedule Round 1 for {app['email']}", key=f"r1_{app['email']}"):
                    update_phase(app["email"], app["role"], "Round 1")
                    send_email(app["email"], "Interview Scheduled - Round 1", "Congratulations! You have been shortlisted for Round 1 interview.")
                    st.success("Round 1 scheduled!")

            elif app["phase"] in ["Round 1", "Round 2"]:
                decision = st.radio(f"Decision for {app['email']} ({app['phase']})", ["Pending", "Pass", "Fail"], key=f"dec_{app['email']}")
                if decision == "Pass":
                    next_phase = "Round 2" if app["phase"] == "Round 1" else "Final"
                    update_phase(app["email"], app["role"], next_phase)
                    send_email(app["email"], f"Interview Scheduled - {next_phase}", f"Congratulations! You have been shortlisted for {next_phase}.")
                    st.success(f"Moved to {next_phase}")
                elif decision == "Fail":
                    update_phase(app["email"], app["role"], "Rejected")
                    send_email(app["email"], "Application Update", "We regret to inform you that your application has been rejected.")
                    st.error("Candidate Rejected")

            elif app["phase"] == "Final":
                decision = st.radio(f"Final Decision for {app['email']}", ["Pending", "Offer", "Reject"], key=f"final_{app['email']}")
                if decision == "Offer":
                    update_phase(app["email"], app["role"], "Selected")
                    send_email(app["email"], "Congratulations!", "You have been selected for the role!")
                    st.success("Offer Sent")
                elif decision == "Reject":
                    update_phase(app["email"], app["role"], "Rejected")
                    send_email(app["email"], "Application Update", "We regret to inform you that your application has been rejected.")
                    st.error("Candidate Rejected")
