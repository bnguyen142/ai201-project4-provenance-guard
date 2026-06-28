import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"

# Thresholds from planning.md Section 2 — asymmetric: the "likely AI" boundary sits
# further from 0.5 than the "likely human" boundary, so confidently labeling something
# AI-generated requires stronger evidence than confidently labeling it human-written.
LIKELY_AI_THRESHOLD = 0.75
LIKELY_HUMAN_THRESHOLD = 0.40

VALID_ATTRIBUTIONS = {"likely_ai", "uncertain", "likely_human"}
