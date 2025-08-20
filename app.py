# app.py
import io
import uuid
from datetime import datetime

import streamlit as st
from supabase import create_client, Client

# ---- CONFIG ----
st.set_page_config(page_title="Resume Analyzer", page_icon="🧾", layout="wide")

REQUIRED_SECRETS = ["SUPABASE_URL", "SUPABASE_ANON_KEY"]
missing = [k for k in REQUIRED_SECRETS if k not in st.secrets]
if missing:
    st.error(
        f"Missing Streamlit secrets: {missing}. "
        "In Streamlit Cloud, go to App → Settings → Secrets and add:\n\n"
        "SUPABASE_URL = 'https://xxxx.supabase.co'\n"
        "SUPABASE_ANON_KEY = 'eyJ...'"
    )
    st.stop()

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

def get_client() -> Client:
    # One global client (no DB driver)
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

sb = get_client()

# ---- SIMPLE PIPELINE / CONSTANTS ----
PIPELINE = ["Applied", "Screening", "Manager Review", "Interview", "Offer", "Hired", "Rejected"]
ADVANCE_TARGET = {i: PIPELINE[i+1] for i in range(len(PIPELINE)-2)}  # everything before Rejected
ADVANCE_TARGET["Offer"] = "Hired"

# ---- UTIL: TEXT EXTRACTION (pure Python deps) ----
def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception:
            return ""
    elif name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""
    elif name.endswith(".txt"):
        return data.decode(errors="ignore")
    else:
        return ""

def simple_match_score(resume_text: str, job_text: str) -> float:
    # Very naive: keyword overlap ratio
    r = set(w.lower() for w in resume_text.split())
    j = set(w.lower() for w in job_text.split())
    if not r or not j:
        return 0.0
    overlap = len(r & j)
    return round(100.0 * overlap / max(1, len(j)), 2)

# ---- AUTH & SESSION ----
if "auth" not in st.session_state:
    st.session_state.auth = None   # dict: {user_id, email, role, access_token}

def sign_in(email: str, password: str):
    try:
        resp = sb.auth.sign_in_with_password({"email": email, "password": password})
        user = resp.user
        token = resp.session.access_token if resp.session else None
        if not user:
            return "Invalid credentials."
        # fetch role from users table
        prof = (
            sb.table("users")
            .select("*")
            .eq("id", user.id)
            .maybe_single()
            .execute()
        )
        # if profile missing, create default candidate profile
        if prof.data is None:
            sb.table("users").insert(
                {"id": user.id, "email": email, "full_name": "", "role": "candidate"}
            ).execute()
            role = "candidate"
        else:
            role = prof.data.get("role", "candidate")
        st.session_state.auth = {
            "user_id": user.id,
            "email": email,
            "role": role,
            "access_token": token,
        }
        return None
    except Exception as e:
        return f"Sign-in error: {e}"

def sign_up(email: str, password: str):
    try:
        resp = sb.auth.sign_up({"email": email, "password": password})
        if not resp.user:
            return "Sign-up failed."
        # create profile row
        sb.table("users").insert(
            {"id": resp.user.id, "email": email, "full_name": "", "role": "candidate"}
        ).execute()
        return None
    except Exception as e:
        return f"Sign-up error: {e}"

def sign_out():
    try:
        sb.auth.sign_out()
    except Exception:
        pass
    st.session_state.auth = None

# ---- UI: AUTH ----
def auth_view():
    st.title("🧾 Resume Analyzer")
    tabs = st.tabs(["Sign In", "Sign Up", "About"])
    with tabs[0]:
        with st.form("signin"):
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign In")
        if submit:
            err = sign_in(email, pw)
            if err:
                st.error(err)
            else:
                st.success("Signed in.")
                st.rerun()
    with tabs[1]:
        with st.form("signup"):
            email = st.text_input("Email ", key="su_email")
            pw = st.text_input("Password ", type="password", key="su_pw")
            submit = st.form_submit_button("Create Account")
        if submit:
            err = sign_up(email, pw)
            if err:
                st.error(err)
            else:
                st.success("Account created. Please sign in.")
    with tabs[2]:
        st.markdown(
            """
**How it works**
- Candidates upload a resume and (optionally) paste a job description.
- We compute a quick keyword overlap score and save an application row.
- HR can review applications and mark **Pass** (auto-advance stage) or **Fail** (Reject).
- Candidates can see their current **phase** at any time.

**Tables expected**
- `users`: `id uuid (PK)`, `email text`, `full_name text`, `role text`
- `applications`: `id uuid (PK)`, `user_id uuid`, `resume_text text`,
  `job_id text`, `job_title text`, `job_description text`,
  `match_score float8`, `phase text`, `created_at timestamptz`
"""
        )

# ---- CANDIDATE VIEW ----
def candidate_view():
    st.sidebar.success(f"Logged in as {st.session_state.auth['email']} (candidate)")
    st.sidebar.button("Sign out", on_click=sign_out)

    st.header("📤 Submit your resume")
    with st.form("resume_form", clear_on_submit=True):
        job_title = st.text_input("Job Title (optional)")
        job_id = st.text_input("Job ID / Requisition (optional)")
        job_desc = st.text_area("Job Description (paste text; optional)", height=150)
        file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
        submitted = st.form_submit_button("Submit Application")
    if submitted:
        if not file:
            st.error("Please upload a resume file.")
        else:
            text = extract_text_from_file(file)
            if not text.strip():
                st.warning("Could not extract text from the file. You can still submit.")
            score = simple_match_score(text, job_desc) if job_desc else 0.0
            phase = "Applied"
            row = {
                "id": str(uuid.uuid4()),
                "user_id": st.session_state.auth["user_id"],
                "resume_text": text[:90000],  # safety cut
                "job_id": job_id or None,
                "job_title": job_title or None,
                "job_description": (job_desc or "")[:90000],
                "match_score": score,
                "phase": phase,
                "created_at": datetime.utcnow().isoformat(),
            }
            try:
                sb.table("applications").insert(row).execute()
                st.success(f"Application submitted. Initial phase: {phase}. Match score: {score}%")
            except Exception as e:
                st.error(f"Failed to save application: {e}")

    st.divider()
    st.subheader("🧭 Your applications & phases")
    try:
        apps = (
            sb.table("applications")
            .select("id, job_title, job_id, match_score, phase, created_at")
            .eq("user_id", st.session_state.auth["user_id"])
            .order("created_at", desc=True)
            .execute()
        )
        if not apps.data:
            st.info("No applications yet.")
        else:
            import pandas as pd
            df = pd.DataFrame(apps.data)
            st.dataframe(df, hide_index=True, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load applications: {e}")

# ---- HR VIEW ----
def hr_view():
    st.sidebar.success(f"Logged in as {st.session_state.auth['email']} (HR)")
    st.sidebar.button("Sign out", on_click=sign_out)

    st.header("🧑‍💼 HR Review")
    # Filters
    colf1, colf2, colf3 = st.columns([2,2,1])
    with colf1:
        job_id_filter = st.text_input("Filter by Job ID (optional)")
    with colf2:
        stage_filter = st.selectbox("Filter by Phase", ["(all)"] + PIPELINE, index=0)
    with colf3:
        st.write("")
        refresh = st.button("Reload")

    # Load applications
    q = sb.table("applications").select(
        "id, user_id, job_title, job_id, match_score, phase, created_at"
    )
    if job_id_filter:
        q = q.ilike("job_id", f"%{job_id_filter}%")
    if stage_filter != "(all)":
        q = q.eq("phase", stage_filter)
    q = q.order("created_at", desc=True)

    try:
        apps = q.execute()
    except Exception as e:
        st.error(f"Failed to load applications: {e}")
        return

    if not apps.data:
        st.info("No applications match your filters.")
        return

    # Map user_id -> email
    user_ids = list({a["user_id"] for a in apps.data})
    users = {}
    try:
        if user_ids:
            resp = sb.table("users").select("id, email, full_name").in_("id", user_ids).execute()
            for u in resp.data or []:
                users[u["id"]] = u
    except Exception:
        pass

    # Render rows with controls
    for a in apps.data:
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([3,2,1,1,2])
            with c1:
                st.markdown(
                    f"**{a.get('job_title') or 'Untitled Role'}**  \n"
                    f"Job ID: `{a.get('job_id') or '-'}`"
                )
                u = users.get(a["user_id"], {})
                st.caption(f"Candidate: {u.get('full_name') or u.get('email') or a['user_id']}")
            with c2:
                st.metric("Match Score", f"{a.get('match_score', 0)}%")
            with c3:
                st.write("Phase")
                st.code(a["phase"])
            with c4:
                st.write("Action")
                # buttons need unique keys
                pass_key = f"pass_{a['id']}"
                fail_key = f"fail_{a['id']}"
                advanced = st.button("Pass ➜", key=pass_key)
                rejected = st.button("Fail ✖", key=fail_key)
            with c5:
                if st.button("View resume text", key=f"view_{a['id']}"):
                    # fetch full row to show resume_text
                    try:
                        full = sb.table("applications").select("resume_text, job_description").eq("id", a["id"]).single().execute()
                        with st.expander("Resume & Job Description", expanded=True):
                            st.text_area("Resume (extracted)", full.data.get("resume_text") or "", height=200)
                            st.text_area("Job Description", full.data.get("job_description") or "", height=150)
                    except Exception as e:
                        st.error(f"Load details failed: {e}")

            # Handle actions
            if "action_result" not in st.session_state:
                st.session_state.action_result = None

            if 'advanced' in locals() and advanced:
                current = a["phase"]
                if current in ("Hired", "Rejected"):
                    st.warning("This application is final. No further advancement.")
                else:
                    new_phase = ADVANCE_TARGET.get(current, "Interview")
                    try:
                        sb.table("applications").update({"phase": new_phase}).eq("id", a["id"]).execute()
                        st.success(f"Advanced to **{new_phase}**.")
                        st.session_state.action_result = True
                    except Exception as e:
                        st.error(f"Update failed: {e}")

            if 'rejected' in locals() and rejected:
                try:
                    sb.table("applications").update({"phase": "Rejected"}).eq("id", a["id"]).execute()
                    st.success("Marked as **Rejected**.")
                    st.session_state.action_result = True
                except Exception as e:
                    st.error(f"Update failed: {e}")

    if st.session_state.get("action_result"):
        # redraw to show updated phases
        st.session_state.action_result = None
        st.rerun()

# ---- ROUTER ----
def main():
    st.sidebar.title("Resume Analyzer")
    if not st.session_state.auth:
        auth_view()
        return

    role = st.session_state.auth["role"]
    if role == "hr":
        hr_view()
    else:
        candidate_view()

if __name__ == "__main__":
    main()
