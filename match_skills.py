# match_skills.py

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import util

# Load stop words once when the module is imported
stop_words = set(stopwords.words('english'))

def extract_keywords(text: str) -> list:
    """
    Extracts and cleans keywords from the resume text.
    It tokenizes the text, converts to lowercase, removes stop words,
    and filters for alphanumeric words.
    """
    if not text:
        return []
    
    # Normalize text: convert to lowercase and remove non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Tokenize the text (split into words)
    tokens = word_tokenize(text)
    
    # Remove stop words and non-alphabetic tokens, and filter by length
    keywords = [
        word for word in tokens 
        if word.isalpha() and word not in stop_words and len(word) > 1
    ]
    
    # Return unique keywords to avoid duplicates
    return list(set(keywords))

def match_resume_to_job(resume_keywords: list, jd_skills: list, model) -> tuple:
    """
    Calculates the match score between resume keywords and job description skills.
    
    Args:
        resume_keywords (list): A list of keywords extracted from the resume.
        jd_skills (list): A list of required skills from the job description.
        model: The pre-loaded SentenceTransformer model.

    Returns:
        tuple: A tuple containing (matched_skills, missing_skills, match_score).
    """
    if not resume_keywords or not jd_skills:
        return [], jd_skills, 0.0

    # Convert lists to sets for efficient operations
    resume_skills_set = set(resume_keywords)
    jd_skills_set = set(jd_skills)

    # Find direct matches (case-insensitive)
    matched_skills = list(resume_skills_set.intersection(jd_skills_set))
    missing_skills = list(jd_skills_set.difference(resume_skills_set))

    # Calculate a score based on direct keyword matches
    if not jd_skills_set:
        return [], [], 100.0 # Or handle as an error case
        
    direct_match_score = (len(matched_skills) / len(jd_skills_set)) * 100

    # For a more advanced score, you could use the model to find semantic similarity
    # between missing skills and the resume content, but for now, we'll rely on the direct match.
    # Example for semantic similarity (can be added later):
    #
    # if missing_skills:
    #     resume_text = " ".join(resume_keywords)
    #     resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    #     missing_skills_embeddings = model.encode(missing_skills, convert_to_tensor=True)
    #     cosine_scores = util.pytorch_cos_sim(resume_embedding, missing_skills_embeddings)
    #     # Add logic to augment score based on high cosine similarity...

    return matched_skills, missing_skills, direct_match_score
