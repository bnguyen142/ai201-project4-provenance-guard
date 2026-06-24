from config import LIKELY_AI_THRESHOLD, LIKELY_HUMAN_THRESHOLD

_LABELS = {
    "likely_ai": "This content was likely created with AI assistance.",
    "likely_human": "This content appears to be written by a human.",
    "uncertain": (
        "We can't confidently tell whether this content was AI-assisted or "
        "human-written — read it with that uncertainty in mind."
    ),
}


def combine_scores(llm_score: float, stylometric_score: float) -> float:
    """
    Combine the two signal scores into one confidence score, per planning.md
    Section 1. Asymmetric: a verdict leaning "AI" (base > 0.5) requires squared
    agreement between the signals to survive disagreement; a verdict leaning
    "human" only needs linear agreement. This makes confidently calling
    something "likely AI" harder than confidently calling it "likely human",
    reflecting that a false positive against a human writer is the worse
    failure mode here.
    """
    base = (llm_score + stylometric_score) / 2
    agreement = 1 - abs(llm_score - stylometric_score)

    if base > 0.5:
        confidence = 0.5 + (base - 0.5) * (agreement ** 2)
    else:
        confidence = 0.5 + (base - 0.5) * agreement

    return confidence


def get_attribution(confidence: float) -> str:
    """Map a confidence score to one of the three attribution bands."""
    if confidence >= LIKELY_AI_THRESHOLD:
        return "likely_ai"
    if confidence <= LIKELY_HUMAN_THRESHOLD:
        return "likely_human"
    return "uncertain"


def get_label(attribution: str) -> str:
    """Return the verbatim transparency label text for an attribution band."""
    return _LABELS[attribution]
