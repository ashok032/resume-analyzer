# match_skills.py
import re
from sentence_transformers import SentenceTransformer, util

# load model once
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def extract_keywords_simple(text: str) -> list:
    """
    Lightweight tokenizer: extract alpha tokens of length >=2, lowercase, unique.
    """
    if not text:
        return []
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    # remove common tiny words manually if desired (optional)
    stopset = {"and","or","the","for","with","that","this","from","your","you","are","will","have","has","in","on","at"}
    tokens = [t for t in tokens if t not in stopset]
    return list(dict.fromkeys(tokens))  # preserve order unique

def match_resume_to_job(resume_text: str, job_skills: list[str], threshold: float = 0.6):
    """
    resume_text: raw resume text
    job_skills: list of skill strings (from DB)
    returns: (score_percent, matched_skills_list, missing_skills_list)
    """
    resume_tokens = extract_keywords_simple(resume_text)
    if not resume_tokens or not job_skills:
        return 0.0, [], job_skills

    # encode tokens (resume tokens) and job_skills (as provided)
    resume_emb = model.encode(resume_tokens, convert_to_tensor=True)
    job_emb = model.encode(job_skills, convert_to_tensor=True)

    sims = util.cos_sim(resume_emb, job_emb)  # shape (len(resume_tokens), len(job_skills))

    matched = []
    for j_idx, job_skill in enumerate(job_skills):
        # find best resume token sim for this job skill
        best_sim = float(sims[:, j_idx].max())
        if best_sim >= threshold:
            matched.append(job_skill)

    matched = list(dict.fromkeys(matched))
    missing = [s for s in job_skills if s not in matched]
    score = round((len(matched) / len(job_skills)) * 100, 2) if job_skills else 0.0
    return score, matched, missing
