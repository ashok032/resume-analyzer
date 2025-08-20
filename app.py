import streamlit as st
import pandas as pd
import hashlib
import pdfplumber
import docx
import re
import spacy
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from supabase import create_client
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords

# Load spaCy & NLTK resources
nlp = spacy.load("en_core_web_sm")
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Connect to Supabase
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# Email Config
EMAIL_USER = st.secrets["EMAIL_USER"]
EMAIL_PASS = st.secrets["EMAIL_PASS"]

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


# ----------------- UTILITIES -----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def send_email(to_email, subject, body):
    msg = MIMEText(body, "plain")
    msg["From"] = EMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = subject

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
    except Exception as e:
        st.error(f"❌ Email sending failed: {e}")


def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_resume_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    else:
        return ""


def extract_contact_info(text):
    email = re.search(r"[\w\.-]+@[\w\.-]+", text)
    phone = re.search(r"\+?\d[\d \-]{8,}\d", text)
    return email.group(0) if email else None, phone.group(0) if phone else None


def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return text.split("\n")[0].strip()


def extract_skills(text):
    stop_words = set(stopwords.words("english"))
    words = nltk.word_tokenize(text)
    skills = [w.lower() for w in words if w.isalpha() and w.lower() not in stop_words]
    return list(set(skills))


def match_resume_to_job(resume_skills, job_skills):
    if not resume_skills or not job_skills:
        return 0, [], job_skills

    resume_emb = model.encode(resume_skills, convert_to_tensor=True)
    job_emb = model.encode(job_skills, convert_to_tensor=True)
    cos_sim = util.cos_sim(resume_emb, job_emb)

    matched = []
    for i, js in enumerate(job_skills):
        if cos_sim[:, i].max().item() > 0.6:
            matched.append(js)

    score = round((len(matched) / len(job_skills)) * 100, 2)
    missing = list(set(job_skills) - set(matched))
    return score, matched, missing


def update_phase(application, result):
    user_email = application["user_email"]
    phase = application["phase"]

    if result == "fail":
        send_email(user_email, "Application Update", f"❌ You were rejected in {phase}.")
        supabase.table("applications").update({"phase": "Rejected"}).eq("id", application["id"]).execute()
    else:
        next_phase = {"Round 1": "Round 2", "Round 2": "Final", "Final": "Selected"}
        new_phase = next_phase.get(phase, "Completed")

        supabase.table("applications").update({"phase": new_phase}).eq("id", application["id"]).execute()
        if new_phase == "Selected":
            send_email(user_email, "Congratulations!", "🎉 You have been selected!")
        else:
            send_email(user_email, "Next Interview Scheduled", f"📅 You have been promoted to {new_phase}.")


# ----------------- AUTH -----------------
def login(email, password):
    hashed = hash_password(password)
    user = supabase.table("users").select("*").eq("email", email).eq("password_hash", hashed).execute()
    return user.data[0] if user.data else None


def register(email, password, role):
    exists = supabase.table("users").select("*").eq("email", email).execute()
    if exists.data:
        return False, "User already exists"
    supabase.table("users").insert({"email": email, "password_hash": hash_password(password), "role": role}).execute()
    return True, "User registered successfully"


# ----------------- STREAMLIT APP -----------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer")

if "user" not in st.session_state:
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login(email, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        email = st.text_input("New Email")
        password = st.text_input("New Password", type="password")
        role = st.selectbox("Role", ["User", "HR"])
        if st.button("Register"):
            ok, msg = register(email, password, role)
            st.success(msg) if ok else st.error(msg)

else:
    user = st.session_state.user
    st.sidebar.write(f"👤 Logged in as {user['email']} ({user['role']})")
    if st.sidebar.button("Logout"):
        st.session_state.pop("user")
        st.rerun()

    if user["role"] == "User":
        st.header("Upload Resume & Get Feedback")
        uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
        if uploaded_file:
            text = extract_resume_text(uploaded_file)
            name = extract_name(text)
            email, phone = extract_contact_info(text)
            resume_skills = extract_skills(text)

            jobs = supabase.table("job_descriptions").select("*").execute().data
            job = st.selectbox("Select Job Role", [j["role"] for j in jobs])
            job_data = next(j for j in jobs if j["role"] == job)

            score, matched, missing = match_resume_to_job(resume_skills, job_data["skills"])

            st.write(f"**Match Score:** {score}%")
            st.write(f"✅ Matched Skills: {matched}")
            st.write(f"❌ Missing Skills: {missing}")

            if st.button("Submit Application"):
                phase = "Round 1" if score > 70 else "Rejected"
                supabase.table("applications").insert({
                    "user_id": user["id"],
                    "job_role": job,
                    "resume_url": uploaded_file.name,
                    "match_score": score,
                    "phase": phase,
                    "submission_date": datetime.now().isoformat(),
                }).execute()
                if phase == "Round 1":
                    send_email(user["email"], "Interview Scheduled", "✅ Your Round 1 interview has been scheduled.")
                else:
                    send_email(user["email"], "Application Result", "❌ You were not selected.")

    elif user["role"] == "HR":
        st.header("HR Dashboard")
        apps = supabase.table("applications").select("*").execute().data
        df = pd.DataFrame(apps)
        st.dataframe(df)

        app_id = st.selectbox("Select Application ID", [a["id"] for a in apps]) if apps else None
        if app_id:
            application = next(a for a in apps if a["id"] == app_id)
            st.write(f"Applicant Email: {application['resume_url']}")
            st.write(f"Phase: {application['phase']}")
            result = st.radio("Mark Result", ["Pass", "Fail"])
            if st.button("Update Result"):
                update_phase(application, "pass" if result == "Pass" else "fail")
                st.success("Result updated!")
