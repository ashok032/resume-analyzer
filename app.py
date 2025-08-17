import streamlit as st
import pandas as pd
import hashlib
import os
import pdfplumber
import docx
import re
import spacy
import nltk
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
from job_descriptions import job_descriptions
from supabase import create_client, Client

# ----------------- SUPABASE CONFIG -----------------
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)

# ----------------- PASSWORD HASHING -----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ----------------- LOAD SPACY MODEL -----------------
try:
    import en_core_web_sm
    nlp = en_core_web_sm.load()
except:
    nlp = spacy.load("en_core_web_sm")

# ----------------- LOAD NLTK -----------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ----------------- EMBEDDING MODEL -----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------- RESUME PARSER -----------------
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join([page.extract_text() or "" for page in pdf.pages])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def parse_resume(file):
    if file.name.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        text = extract_text_from_docx(file)
    else:
        return None, None, None, ""

    doc = nlp(text)

    # Extract name (first PERSON entity or first line fallback)
    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break
    if not name:
        name = text.split("\n")[0].strip()

    # Extract email
    email = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    email = email.group(0) if email else None

    # Extract phone
    phone = re.search(r"(\+?\d{1,3}[-.\s]?)?\d{10}", text)
    phone = phone.group(0) if phone else None

    return name, email, phone, text

# ----------------- SKILL EXTRACTION -----------------
def extract_keywords(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return list(set([word for word in tokens if word.isalpha() and word not in stop_words]))

def match_resume_to_job(resume_keywords, jd_skills):
    resume_embeddings = embedder.encode(resume_keywords, convert_to_tensor=True)
    jd_embeddings = embedder.encode(jd_skills, convert_to_tensor=True)

    matches, missing = [], []
    for skill, jd_emb in zip(jd_skills, jd_embeddings):
        cos_sim = util.cos_sim(jd_emb, resume_embeddings).max().item()
        if cos_sim > 0.6:
            matches.append(skill)
        else:
            missing.append(skill)

    score = round((len(matches) / len(jd_skills)) * 100, 2) if jd_skills else 0
    return matches, missing, score

# ----------------- APPLICATION PHASE -----------------
def get_next_phase(current_phase, result):
    if result == "Fail":
        return "Rejected"
    if current_phase == "Applied":
        return "Round 1"
    if current_phase == "Round 1":
        return "Round 2"
    if current_phase == "Round 2":
        return "Final"
    if current_phase == "Final":
        return "Selected"
    return current_phase

# ----------------- AUTHENTICATION -----------------
USERS_FILE = "users.csv"
if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["email", "password", "role"]).to_csv(USERS_FILE, index=False)

def load_users():
    return pd.read_csv(USERS_FILE)

def save_user(email, password, role):
    users = load_users()
    if email in users["email"].values:
        return False
    new_user = pd.DataFrame([[email, hash_password(password), role]], columns=["email", "password", "role"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USERS_FILE, index=False)
    return True

def authenticate(email, password):
    users = load_users()
    user = users[(users["email"] == email) & (users["password"] == hash_password(password))]
    if not user.empty:
        return user.iloc[0]["role"]
    return None

# ----------------- STREAMLIT APP -----------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.email = None

if not st.session_state.logged_in:
    st.title("🔐 Resume Analyzer Login")

    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            role = authenticate(email, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.role = role
                st.session_state.email = email
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        email = st.text_input("New Email")
        password = st.text_input("New Password", type="password")
        role = st.selectbox("Role", ["User", "HR"])
        if st.button("Register"):
            if save_user(email, password, role):
                st.success("User registered! Please login.")
            else:
                st.error("Email already exists.")

else:
    st.sidebar.write(f"👤 Logged in as {st.session_state.email} ({st.session_state.role})")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("📄 AI Resume Analyzer")

    # ----------------- HR VIEW -----------------
    if st.session_state.role == "HR":
        st.subheader("👔 HR Dashboard")

        job_role = st.selectbox("Select Job Role", list(job_descriptions.keys()))
        uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

        if uploaded_file:
            name, email, phone, text = parse_resume(uploaded_file)
            resume_keywords = extract_keywords(text)
            jd_skills = job_descriptions[job_role]["skills"]
            matched, missing, score = match_resume_to_job(resume_keywords, jd_skills)

            st.write(f"**Candidate:** {name}")
            st.write(f"**Email:** {email}")
            st.write(f"**Phone:** {phone}")
            st.write(f"**Match Score:** {score}%")

            # Save to Supabase
            data = {
                "name": name,
                "email": email,
                "job_role": job_role,
                "score": score,
                "phase": "Round 1" if score > 70 else "Rejected",
                "submitted_at": datetime.now().isoformat()
            }
            supabase.table("applications").insert(data).execute()
            st.success(f"Application saved! Phase: {data['phase']}")

        st.subheader("📋 Manage Applications")
        apps = supabase.table("applications").select("*").execute().data
        if apps:
            df = pd.DataFrame(apps)
            st.dataframe(df)

            for idx, row in df.iterrows():
                st.write(f"{row['name']} ({row['job_role']}) - {row['phase']}")
                if row["phase"] not in ["Rejected", "Selected"]:
                    result = st.radio(f"Result for {row['name']} in {row['phase']}", ["Pending", "Pass", "Fail"], key=f"{idx}")
                    if st.button(f"Update {row['name']}", key=f"btn_{idx}"):
                        new_phase = get_next_phase(row["phase"], result)
                        supabase.table("applications").update({"phase": new_phase}).eq("id", row["id"]).execute()
                        st.success(f"Updated {row['name']} → {new_phase}")
                        st.rerun()

    # ----------------- USER VIEW -----------------
    elif st.session_state.role == "User":
        st.subheader("🙋 Candidate Dashboard")
        email = st.session_state.email
        apps = supabase.table("applications").select("*").eq("email", email).execute().data
        if apps:
            df = pd.DataFrame(apps)
            st.dataframe(df[["job_role", "score", "phase", "submitted_at"]])
        else:
            st.info("No applications found. Please ask HR to upload your resume.")
