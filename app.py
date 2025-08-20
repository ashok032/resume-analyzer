import io
import json
import re
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np

from supabase import create_client, Client

# ---- UI BASE ----
st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")

# -----------------------------
# Secrets (already provided by you in Streamlit)
# -----------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
EMAIL_USER = st.secrets["EMAIL_USER"]
EMAIL_PASS = st.secrets["EMAIL_PASS"]

# -----------------------------
# Supabase Client
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

sb = get_supabase()

# -----------------------------
# Lightweight Skill Library
# -----------------------------
# Lowercase for matching; include common tech & soft skills
SKILL_LIBRARY = sorted(set([
    # languages
    "python","java","javascript","typescript","c","c++","c#","go","golang","ruby","php","scala","kotlin","r","sql",
    # web / backend
    "node","node.js","express","django","flask","fastapi","spring","spring boot","graphql","rest","grpc",
    # data / ml
    "pandas","numpy","scikit-learn","sklearn","tensorflow","keras","pytorch","spacy","nltk","xgboost","lightgbm",
    "mlflow","airflow","dbt","spark","hadoop","databricks","snowflake","bigquery","redshift","data engineering",
    "feature engineering","computer vision","nlp","llm","generative ai","deep learning","machine learning",
    # cloud / devops
    "aws","azure","gcp","docker","kubernetes","terraform","ansible","jenkins","github actions","gitlab ci",
    "linux","bash","shell","prometheus","grafana","elastic","elk","datadog",
    # web ui
    "react","next.js","vue","svelte","html","css","sass","tailwind","redux","vite",
    # databases & messaging
    "postgres","mysql","mongodb","redis","dynamodb","cassandra","kafka","rabbitmq",
    # testing
    "pytest","unittest","cypress","playwright","selenium","junit",
    # analytics / bi
    "power bi","tableau","looker",
    # security
    "oauth","oidc","jwt","sso",
    # soft skills
    "leadership","mentoring","communication","stakeholder management","agile","scrum","kanban"
]))

# Pre-compile for fast contains checks
SKILL_PATTERNS = [(s, re.compile(rf"\b{re.escape(s)}\b")) for s in SKILL_LIBRARY]

# -----------------------------
# Resume Reading
# -----------------------------
def read_pdf(file) -> str:
    import pdfplumber
    text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def read_docx(file) -> str:
    from docx import Document
    d = Document(file)
    return "\n".join(p.text for p in d.paragraphs)

def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    f = io.BytesIO(data)
    if name.endswith(".pdf"):
        return read_pdf(f)
    elif name.endswith(".docx"):
        return read_docx(f)
    else:
        # fallback: treat as text
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def normalize_text(s: str) -> str:
    s = s.replace("C++", "c++").replace("C#", "c#").replace("Node.js","node.js").replace("Next.js","next.js")
    return re.sub(r"\s+", " ", s).strip().lower()

# -----------------------------
# Skill Extraction / Matching
# -----------------------------
def extract_skills(text: str) -> List[str]:
    text_l = normalize_text(text)
    found = []
    for skill, pat in SKILL_PATTERNS:
        if pat.search(text_l):
            found.append(skill)
    return sorted(set(found))

def job_required_skills(job_text: str) -> List[str]:
    return extract_skills(job_text)

# -----------------------------
# Semantic Similarity (lazy import + fallback)
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    try:
        # Lazy import so the app still runs even if sentence-transformers fails
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model
    except Exception:
        return None

EMBEDDER = get_embedder()

@st.cache_resource(show_spinner=False)
def get_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=20000)

def semantic_score(text_a: str, text_b: str) -> float:
    a = normalize_text(text_a)
    b = normalize_text(text_b)

    if EMBEDDER is not None:
        try:
            from sentence_transformers import util
            va = EMBEDDER.encode([a], convert_to_tensor=True, normalize_embeddings=True)
            vb = EMBEDDER.encode([b], convert_to_tensor=True, normalize_embeddings=True)
            cos = float(util.cos_sim(va, vb)[0][0].cpu().item())
            return max(0.0, min(1.0, cos))
        except Exception:
            pass

    # TF-IDF fallback
    vec = get_tfidf()
    mats = vec.fit_transform([a, b])
    va, vb = mats[0], mats[1]
    denom = (np.linalg.norm(va.toarray()) * np.linalg.norm(vb.toarray()))
    if denom == 0:
        return 0.0
    sim = float((va @ vb.T).toarray()[0][0] / denom)
    return max(0.0, min(1.0, sim))

def match_resume_to_job(resume_text: str, job_text: str) -> Dict:
    resume_sk = extract_skills(resume_text)
    job_sk = job_required_skills(job_text)

    # Coverage of job-required skills
    required = set(job_sk)
    have = sorted(set(resume_sk) & required)
    gaps = sorted(required - set(resume_sk))
    coverage = (len(have) / len(required)) if required else 0.0

    sem = semantic_score(resume_text, job_text)

    # Blend: tweak weights as needed
    final = 0.6 * sem + 0.4 * coverage
    return {
        "semantic": round(sem * 100, 2),
        "coverage": round(coverage * 100, 2),
        "final_score": round(final * 100, 2),
        "matched_skills": have,
        "gaps": gaps,
        "resume_skills": sorted(set(resume_sk)),
        "job_skills": sorted(set(job_sk)),
    }

# -----------------------------
# Email
# -----------------------------
def send_email(to_email: str, subject: str, body: str) -> Tuple[bool, str]:
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["From"] = formataddr(("Resume Analyzer", EMAIL_USER))
        msg["To"] = to_email
        msg["Subject"] = subject

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.sendmail(EMAIL_USER, [to_email], msg.as_string())
        return True, "sent"
    except Exception as e:
        return False, str(e)

# -----------------------------
# Supabase Helpers
# -----------------------------
def list_jobs() -> List[Dict]:
    resp = sb.table("jobs").select("*").order("created_at", desc=True).execute()
    return resp.data or []

def create_job(title: str, description: str) -> Optional[Dict]:
    resp = sb.table("jobs").insert({"title": title, "description": description}).execute()
    if resp.data:
        return resp.data[0]
    return None

def save_application(payload: Dict) -> Optional[Dict]:
    resp = sb.table("applications").insert(payload).execute()
    if resp.data:
        return resp.data[0]
    return None

def list_applications(job_id: Optional[int] = None) -> List[Dict]:
    q = sb.table("applications").select("*").order("created_at", desc=True)
    if job_id:
        q = q.eq("job_id", job_id)
    return (q.execute().data) or []

def update_application(app_id: int, updates: Dict) -> None:
    sb.table("applications").update(updates).eq("id", app_id).execute()

# -----------------------------
# UI Components
# -----------------------------
def ui_candidate():
    st.header("Apply for a Job")
    with st.expander("Select a Job (or paste JD)"):
        jobs = list_jobs()
        job_titles = ["-- Paste custom Job Description --"] + [f"{j['id']} · {j['title']}" for j in jobs]
        sel = st.selectbox("Job", job_titles)
        jd_text = ""
        job_id = None
        if sel == job_titles[0]:
            jd_text = st.text_area("Job Description", height=200, placeholder="Paste the JD here…")
        else:
            idx = job_titles.index(sel) - 1
            job = jobs[idx]
            job_id = job["id"]
            jd_text = job["description"]

    st.subheader("Your Details")
    col1, col2 = st.columns(2)
    name = col1.text_input("Full Name *")
    email = col2.text_input("Email *")
    phone = col1.text_input("Phone")

    st.subheader("Upload Resume")
    up = st.file_uploader("PDF or DOCX", type=["pdf", "docx"])

    analyze = st.button("Analyze & Submit", type="primary", use_container_width=True)

    if analyze:
        if not name or not email:
            st.error("Name and Email are required.")
            return
        if not up:
            st.error("Please upload your resume.")
            return
        if not jd_text.strip():
            st.error("Please provide a job description (select a job or paste JD).")
            return

        with st.spinner("Reading resume…"):
            resume_text = extract_text(up)

        with st.spinner("Matching…"):
            result = match_resume_to_job(resume_text, jd_text)

        st.success(f"Match Score: {result['final_score']}%")
        colA, colB, colC = st.columns(3)
        colA.metric("Semantic Similarity", f"{result['semantic']}%")
        colB.metric("Skill Coverage", f"{result['coverage']}%")
        colC.metric("Skills Found", f"{len(result['resume_skills'])}")

        st.write("**Matched skills**:", ", ".join(result["matched_skills"]) or "—")
        st.write("**Gaps**:", ", ".join(result["gaps"]) or "—")

        # Save to Supabase
        payload = {
            "name": name,
            "email": email,
            "phone": phone,
            "job_id": job_id,
            "job_title": None if job_id is None else next((j["title"] for j in jobs if j["id"] == job_id), None),
            "resume_text": resume_text[:20000],  # keep row size reasonable
            "job_text": jd_text[:20000],
            "score": result["final_score"],
            "semantic": result["semantic"],
            "coverage": result["coverage"],
            "matched_skills": result["matched_skills"],
            "gaps": result["gaps"],
            "status": "Applied",
            "round": 0,
            "notes": "",
        }
        try:
            saved = save_application(payload)
            st.info("Application saved.")
        except Exception as e:
            st.warning(f"Could not save application: {e}")
            saved = None

        # Email candidate (you) + HR (sender) as a simple example
        subject = f"Application received: {payload.get('job_title') or 'Custom JD'}"
        body = f"""Hi {name},

Thanks for applying. Here is your instant match summary:

- Overall match: {result['final_score']}%
- Semantic similarity: {result['semantic']}%
- Skill coverage: {result['coverage']}%
- Matched skills: {', '.join(result['matched_skills']) or '—'}
- Gaps: {', '.join(result['gaps']) or '—'}

We'll get back to you shortly.
"""
        ok1, msg1 = send_email(email, subject, body)
        ok2, msg2 = send_email(EMAIL_USER, f"[HR Copy] {subject}", f"{body}\n\nCandidate email: {email}\nPhone: {phone}")
        if ok1:
            st.success("Confirmation email sent to candidate.")
        else:
            st.warning(f"Could not email candidate: {msg1}")
        if ok2:
            st.success("Notification sent to HR inbox.")
        else:
            st.warning(f"Could not email HR: {msg2}")

def ui_hr():
    st.header("HR / Recruiter Console")

    tabs = st.tabs(["📌 Post a Job", "📥 Applications"])
    # --- Post a Job
    with tabs[0]:
        st.subheader("Create new Job")
        jt = st.text_input("Job Title *")
        jd = st.text_area("Job Description *", height=220)
        if st.button("Create Job", type="primary"):
            if not jt.strip() or not jd.strip():
                st.error("Title and Description are required.")
            else:
                try:
                    job = create_job(jt.strip(), jd.strip())
                    if job:
                        st.success(f"Job #{job['id']} created: {job['title']}")
                except Exception as e:
                    st.error(f"Failed to create job: {e}")

        st.markdown("---")
        st.subheader("Existing Jobs")
        jobs = list_jobs()
        if jobs:
            df = pd.DataFrame([{
                "id": j["id"], "title": j["title"], "created_at": j.get("created_at")
            } for j in jobs])
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No jobs yet.")

    # --- Applications
    with tabs[1]:
        jobs = list_jobs()
        job_filter = st.selectbox(
            "Filter by Job",
            ["All"] + [f"{j['id']} · {j['title']}" for j in jobs]
        )

        selected_job_id = None
        if job_filter != "All":
            idx = [f"{j['id']} · {j['title']}" for j in jobs].index(job_filter)
            selected_job_id = jobs[idx]["id"]

        apps = list_applications(selected_job_id)
        if not apps:
            st.info("No applications found.")
            return

        df = pd.DataFrame([{
            "id": a["id"],
            "created_at": a.get("created_at"),
            "name": a["name"],
            "email": a["email"],
            "phone": a.get("phone"),
            "job_id": a.get("job_id"),
            "job_title": a.get("job_title") or ("Custom JD" if a.get("job_id") is None else ""),
            "score": a.get("score"),
            "semantic": a.get("semantic"),
            "coverage": a.get("coverage"),
            "status": a.get("status"),
            "round": a.get("round", 0),
        } for a in apps])
        st.dataframe(df.sort_values("score", ascending=False), hide_index=True, use_container_width=True)

        st.markdown("#### Review / Update")
        app_ids = [str(a["id"]) for a in apps]
        pick = st.selectbox("Select application", app_ids)
        chosen = next(a for a in apps if str(a["id"]) == pick)

        st.write(f"**Candidate:** {chosen['name']}  |  **Email:** {chosen['email']}")
        st.write(f"**Match:** {chosen.get('score', 0)}%  |  Semantic {chosen.get('semantic', 0)}%  |  Coverage {chosen.get('coverage', 0)}%")
        with st.expander("Matched skills / Gaps"):
            st.write("**Matched:**", ", ".join(chosen.get("matched_skills") or []) or "—")
            st.write("**Gaps:**", ", ".join(chosen.get("gaps") or []) or "—")

        new_status = st.selectbox("Status", ["Applied","Screening","Shortlisted","Interviewing","Offered","Rejected"], index=[
            "Applied","Screening","Shortlisted","Interviewing","Offered","Rejected"
        ].index(chosen.get("status","Applied")))
        new_round = st.number_input("Interview Round", min_value=0, max_value=10, step=1, value=int(chosen.get("round",0)))
        new_notes = st.text_area("Notes", value=chosen.get("notes",""), height=160)

        colu1, colu2, colu3 = st.columns([1,1,1])
        if colu1.button("Save Update", type="primary"):
            try:
                update_application(chosen["id"], {"status": new_status, "round": int(new_round), "notes": new_notes})
                st.success("Application updated.")
            except Exception as e:
                st.error(f"Update failed: {e}")

        if colu2.button("Email Candidate"):
            subj = f"Update on your application (Status: {new_status})"
            body = f"""Hi {chosen['name']},

We wanted to share an update on your application.

Current status: {new_status}
Round: {new_round if new_status.lower()!='applied' else '—'}

Notes:
{new_notes or '—'}

Best,
HR Team
"""
            ok, msg = send_email(chosen["email"], subj, body)
            st.success("Email sent.") if ok else st.error(f"Email failed: {msg}")

        if colu3.button("Email Myself (HR) Summary"):
            subj = f"[HR] {chosen['name']} · Score {chosen.get('score',0)}%"
            body = f"""Candidate: {chosen['name']} <{chosen['email']}>
Phone: {chosen.get('phone') or '—'}
Job: {chosen.get('job_title') or chosen.get('job_id') or 'Custom JD'}
Score: {chosen.get('score',0)}% (Semantic {chosen.get('semantic',0)}%, Coverage {chosen.get('coverage',0)}%)

Matched: {', '.join(chosen.get('matched_skills') or []) or '—'}
Gaps: {', '.join(chosen.get('gaps') or []) or '—'}

Notes:
{new_notes or '—'}
"""
            ok, msg = send_email(EMAIL_USER, subj, body)
            st.success("Email sent to HR inbox.") if ok else st.error(f"Email failed: {msg}")

# -----------------------------
# Top-level Nav
# -----------------------------
st.title("📄 Resume Analyzer")
st.caption("Instant matching. Simple hiring workflow. Powered by Supabase + Streamlit.")

mode = st.sidebar.radio("Mode", ["Candidate", "HR"])

if mode == "Candidate":
    ui_candidate()
else:
    ui_hr()
