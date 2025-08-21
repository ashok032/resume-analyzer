# db.py
from typing import Any

def get_user(supabase, email: str) -> dict | None:
    res = supabase.table("users").select("*").eq("email", email).single().execute()
    return res.data if res.data else None

def insert_user(supabase, email: str, password_hash: str, role: str) -> Any:
    return supabase.table("users").insert({
        "email": email,
        "password_hash": password_hash,
        "role": role
    }).execute()

def get_jobs(supabase) -> list:
    res = supabase.table("jobs").select("*").order("created_at", desc=True).execute()
    return res.data or []

def get_job_by_id(supabase, job_id: str) -> dict | None:
    res = supabase.table("jobs").select("*").eq("id", job_id).single().execute()
    return res.data if res.data else None

def create_job(supabase, title: str, description: str, skills: list) -> Any:
    return supabase.table("jobs").insert({
        "title": title,
        "description": description,
        "skills": skills
    }).execute()

def update_job(supabase, job_id: str, title: str, description: str, skills: list) -> Any:
    return supabase.table("jobs").update({
        "title": title,
        "description": description,
        "skills": skills
    }).eq("id", job_id).execute()

def delete_job(supabase, job_id: str) -> Any:
    return supabase.table("jobs").delete().eq("id", job_id).execute()

def insert_application(supabase, payload: dict) -> Any:
    # payload should include: user_id, candidate_name, candidate_email, candidate_phone, job_id, resume_text, skills (json or array), match_score, phase, status
    return supabase.table("applications").insert(payload).execute()

def get_applications(supabase) -> list:
    res = supabase.table("applications").select("*").order("created_at", desc=True).execute()
    return res.data or []

def update_application_phase(supabase, app_id: str, updates: dict) -> Any:
    return supabase.table("applications").update(updates).eq("id", app_id).execute()
