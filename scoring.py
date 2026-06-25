from config import LIKELY_AI_THRESHOLD, LIKELY_HUMAN_THRESHOLD

_LABELS = {
    "likely_ai": "This content was likely created with AI assistance.",
    "likely_human": "This content appears to be written by a human.",
    "uncertain": (
        "We can't confidently tell whether this content was AI-assisted or "
        "human-written — read it with that uncertainty in mind."
    ),
}


_WEIGHTS = {"llm": 0.5, "stylometric": 0.3, "marker": 0.2}


def combine_scores(llm_score: float, stylometric_score: float, marker_score: float) -> float:
    """
    Combine the three signal scores into one confidence score, per planning.md
    "Ensemble Detection" section. Weighted average (LLM weighted highest, marker
    phrase weighted lowest, since it has the narrowest blind spot) gives `base`.

    "agreement" is the weighted standard deviation of the three scores around
    that same weighted base, not a plain max-min spread — using the weights
    here too means an outlier carries less disagreement weight if it's a
    signal we trust less, and two signals that agree get real credit even when
    a third disagrees, rather than being penalized as if all three disagreed
    equally. Normalized against 0.5, the maximum possible weighted std for
    scores bounded in [0, 1] (reached when the weighted mean sits at 0.5).

    The asymmetric squaring rule is unchanged: a verdict leaning "AI"
    (base > 0.5) requires squared agreement to survive disagreement, a verdict
    leaning "human" only needs linear agreement — confidently calling
    something "likely AI" stays harder than confidently calling it "likely
    human", since a false positive against a human writer is the worse
    failure mode here.
    """
    scores = [llm_score, stylometric_score, marker_score]
    weights = [_WEIGHTS["llm"], _WEIGHTS["stylometric"], _WEIGHTS["marker"]]
    base = sum(w * s for w, s in zip(weights, scores))
    variance = sum(w * (s - base) ** 2 for w, s in zip(weights, scores))
    std = variance ** 0.5
    agreement = max(0.0, 1 - 2 * std)

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
