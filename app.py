import streamlit as st
import pandas as pd
import hashlib
import os
import pdfplumber
import docx
import re
import spacy
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from job_descriptions import job_descriptions
from sentence_transformers import SentenceTransformer, util
from supabase import create_client, Client

# Load secrets
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
EMAIL_USER = st.secrets["EMAIL_USER"]
EMAIL_PASS = st.secrets["EMAIL_PASS"]

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------- HELPER FUNCTIONS ----------------------

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def send_email(to_email, subject, body):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        st.error("Unsupported file format.")
    return text

def extract_name_email_phone(text):
    doc = nlp(text)
    name = None
    email = None
    phone = None

    # Extract name (first PERSON entity or ALL CAPS heuristic)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break
    if not name:
        for line in text.split("\n")[:5]:
            if line.isupper() and len(line.split()) <= 3:
                name = line.strip()
                break

    # Extract email
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    if email_match:
        email = email_match.group(0)

    # Extract phone
    phone_match = re.search(r"\+?\d[\d -]{8,}\d", text)
    if phone_match:
        phone = phone_match.group(0)

    return name, email, phone

def extract_skills(text):
    words = set([token.text.lower() for token in nlp(text) if not token.is_stop and token.is_alpha])
    return words

def match_resume_to_job(resume_skills, job_skills):
    embeddings_resume = model.encode(list(resume_skills), convert_to_tensor=True)
    embeddings_job = model.encode(job_skills, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings_resume, embeddings_job)

    matched = []
    for i, score in enumerate(cosine_scores):
        if float(score.max()) > 0.6:
            matched.append(job_skills[i])

    missing = list(set(job_skills) - set(matched))
    score = int((len(matched) / len(job_skills)) * 100) if job_skills else 0
    return matched, missing, score

# ---------------------- STREAMLIT APP ----------------------

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "email" not in st.session_state:
    st.session_state.email = None

st.title("📄 AI-Powered Resume Analyzer")

if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["🔑 Login", "🆕 Register"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            hashed_pw = hash_password(password)
            res = supabase.table("users").select("*").eq("email", email).eq("password", hashed_pw).execute()
            if res.data:
                st.session_state.logged_in = True
                st.session_state.role = res.data[0]["role"]
                st.session_state.email = email
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        email = st.text_input("Register Email")
        password = st.text_input("Register Password", type="password")
        role = st.selectbox("Role", ["User", "HR"])
        if st.button("Register"):
            hashed_pw = hash_password(password)
            res = supabase.table("users").insert({"email": email, "password": hashed_pw, "role": role}).execute()
            if res.data:
                st.success("Registration successful! Please log in.")
            else:
                st.error("Error registering user")

else:
    with st.sidebar:
        st.write(f"👤 Logged in as: {st.session_state.email}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.session_state.email = None
            st.rerun()

    if st.session_state.role == "User":
        st.header("Upload Resume & Track Application")
        uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
        job_role = st.selectbox("Select Job Role", list(job_descriptions.keys()))

        if uploaded_file and job_role:
            text = extract_text_from_file(uploaded_file)
            name, email, phone = extract_name_email_phone(text)
            resume_skills = extract_skills(text)
            jd_skills = job_descriptions[job_role]["skills"]

            matched, missing, score = match_resume_to_job(resume_skills, jd_skills)

            st.subheader("📊 Analysis Result")
            st.write(f"**Name:** {name}")
            st.write(f"**Email:** {email}")
            st.write(f"**Phone:** {phone}")
            st.write(f"**Match Score:** {score}%")
            st.write(f"✅ Matched Skills: {', '.join(matched)}")
            st.write(f"❌ Missing Skills: {', '.join(missing)}")

            # Save application to Supabase
            supabase.table("applications").upsert({
                "user_email": st.session_state.email,
                "candidate_name": name,
                "resume_email": email,
                "phone": phone,
                "job_role": job_role,
                "score": score,
                "phase": "Round 1" if score > 70 else "Rejected",
                "updated_at": datetime.now().isoformat()
            }).execute()

    elif st.session_state.role == "HR":
        st.header("📋 Manage Applications")
        res = supabase.table("applications").select("*").execute()
        if res.data:
            df = pd.DataFrame(res.data)
            st.dataframe(df[["candidate_name", "resume_email", "job_role", "score", "phase"]])

            selected_email = st.selectbox("Select Candidate by Email", df["resume_email"].unique())
            action = st.selectbox("Action", ["None", "Mark Pass", "Mark Fail"])

            if st.button("Update Result"):
                candidate = next((row for row in df.to_dict("records") if row["resume_email"] == selected_email), None)
                if candidate:
                    new_phase = candidate["phase"]
                    if action == "Mark Pass":
                        if new_phase == "Round 1":
                            new_phase = "Round 2"
                            send_email(candidate["resume_email"], "Interview Round 2 Scheduled", "Congrats! You are selected for Round 2.")
                        elif new_phase == "Round 2":
                            new_phase = "Final Round"
                            send_email(candidate["resume_email"], "Final Round Scheduled", "Congrats! You are selected for Final Round.")
                        elif new_phase == "Final Round":
                            new_phase = "Selected"
                            send_email(candidate["resume_email"], "Offer Letter", "Congratulations! You are selected.")
                    elif action == "Mark Fail":
                        new_phase = "Rejected"
                        send_email(candidate["resume_email"], "Application Update", "Sorry, you were not selected.")

                    supabase.table("applications").update({"phase": new_phase, "updated_at": datetime.now().isoformat()}).eq("resume_email", selected_email).execute()
                    st.success("Application updated!")
                    st.rerun()

        else:
            st.info("No applications found yet.")
