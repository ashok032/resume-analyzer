import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

# ✅ Ensure NLTK resources are available (prevents LookupError in deployment)
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

# ✅ Load sentence transformer model (semantic similarity)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_keywords(text):
    """
    Extract keywords from resume or job description text.
    Removes stopwords, punctuation, and numbers.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    keywords = [
        word for word in tokens 
        if word.isalpha() and word not in stop_words
    ]
    return list(set(keywords))

def match_resume_to_job(resume_text, job_description):
    """
    Match resume skills with job description using semantic similarity.
    Returns:
        - match_score (%)
        - matched_skills (list)
        - missing_skills (list)
    """
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description)

    matched_skills = []
    missing_skills = []

    if not job_keywords:
        return 0, matched_skills, missing_skills

    # Encode once for efficiency
    resume_embeddings = model.encode(resume_keywords, convert_to_tensor=True)
    job_embeddings = model.encode(job_keywords, convert_to_tensor=True)

    # Compute similarity matrix
    cosine_scores = util.cos_sim(resume_embeddings, job_embeddings)

    for i, job_kw in enumerate(job_keywords):
        # Find the most similar resume word
        best_match_idx = cosine_scores[:, i].argmax().item()
        score = cosine_scores[best_match_idx, i].item()
        if score > 0.6:  # threshold for semantic match
            matched_skills.append(job_kw)
        else:
            missing_skills.append(job_kw)

    match_score = (len(matched_skills) / len(job_keywords)) * 100
    return round(match_score, 2), matched_skills, missing_skills
