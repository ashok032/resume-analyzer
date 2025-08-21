# match_skills.py
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import re
import os

# Ensure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_keywords(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [re.sub(r'\W+', '', token) for token in tokens if token.isalpha()]
    return list(set(tokens))

def match_resume_to_job(resume_text, job_description):
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description)

    if not resume_keywords or not job_keywords:
        return 0.0, [], []

    resume_embeddings = model.encode(resume_keywords, convert_to_tensor=True)
    job_embeddings = model.encode(job_keywords, convert_to_tensor=True)

    cosine_scores = util.cos_sim(resume_embeddings, job_embeddings)

    matched_skills = []
    missing_skills = []

    for i, job_kw in enumerate(job_keywords):
        max_score = max([cosine_scores[j][i].item() for j in range(len(resume_keywords))])
        if max_score > 0.6:
            matched_skills.append(job_kw)
        else:
            missing_skills.append(job_kw)

    score = (len(matched_skills) / len(job_keywords)) * 100 if job_keywords else 0
    return round(score, 2), matched_skills, missing_skills
