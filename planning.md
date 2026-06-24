# Provenance Guard — Planning

<!--
Fill this out BEFORE writing implementation code (per Milestone 1-2).
Answer every question below with specific, implementation-ready detail —
vague answers here produce vague code later. Delete these instruction
comments once answered.
-->

## 1. Detection Signals

### Signal 1 — LLM-based classification (Groq, llama-3.3-70b-versatile)

- Measures: holistic semantic/stylistic coherence — the model judges whether the text "reads as" human- or AI-written, based on patterns learned from large amounts of both.
- Output: a score between 0.0-1.0 (`llm_score`), where higher = more AI-like.
- **Groq must be forced to return structured JSON, not free-form prose.** Use Groq's JSON mode / `response_format` constraint so the response is directly parseable. Expected shape from the Groq call:

  ```json
  {
    "llm_score": 0.81,
    "reasoning": "short string explaining the judgment"
  }
  ```

  If the response fails to parse as JSON, handle it explicitly (e.g. retry once or fall back to a neutral score) rather than letting the request crash — see `CLAUDE.md` guardrail #3.
- Blind spot: black-box judgment with no inspectable reasoning; can be fooled by AI text that's been lightly edited by a human, and can penalize unusual-but-genuine human voices (non-native speakers, very formal writers) it has seen less of in training.

### Signal 2 — Stylometric heuristics (pure Python)

- Measures: structural/statistical regularity — sentence-length variability (coefficient of variation, i.e. std/mean, so it's comparable across excerpts of different overall length), type-token ratio (vocabulary diversity, calibrated against a ~0.90 baseline since short excerpts naturally run high), and punctuation density. AI text tends toward uniformity; human writing toward variability.
- Output: a score between 0.0-1.0 (`stylometric_score`), where higher = more AI-like (more structurally uniform).
- Blind spot: purely structural, no understanding of meaning. A human who writes very consistently (technical writers, non-native speakers following learned sentence patterns) looks "AI-like" by this signal alone even though the content is original.

The two signals are genuinely independent: each runs on the same raw text without seeing the other's output (see Architecture diagram below) — one is semantic, one is structural.

**Combination into a single confidence score**: `llm_score` and `stylometric_score` are combined into one 0.0-1.0 `confidence` score, where higher still means more AI-like (consistent with both raw signals). The combination is intentionally **asymmetric**: the system is biased against confidently labeling something "likely AI" unless both signals strongly agree — false positives against human writers are worse than false negatives on AI text (see the false-positive scenario below). When the two signals disagree significantly, the combined score is pulled toward the "uncertain" band rather than naively averaged into a confident verdict.

Exact formula:

```python
def combine_scores(llm_score: float, stylometric_score: float) -> float:
    base = (llm_score + stylometric_score) / 2
    agreement = 1 - abs(llm_score - stylometric_score)  # 1.0 = perfect agreement, 0.0 = max disagreement

    if base > 0.5:
        # Leaning "AI" — require strong agreement (squared) before trusting it.
        # One signal alone, or two signals that mildly agree, isn't enough.
        confidence = 0.5 + (base - 0.5) * (agreement ** 2)
    else:
        # Leaning "human" — a single linear agreement penalty is enough.
        # We don't need as much corroboration to lean toward "human."
        confidence = 0.5 + (base - 0.5) * agreement

    return confidence
```

Why this is asymmetric and not a naive average: any disagreement between signals shrinks the
distance from 0.5 (pulling toward uncertain) in both directions, but the AI-leaning branch
squares the agreement term, so partial disagreement collapses the AI verdict much faster than
the human verdict. E.g. `llm_score=0.9, stylometric_score=0.5` (agreement=0.6, base=0.7) yields
confidence ≈ 0.5 + 0.2*0.36 = 0.572 (uncertain) — but if both scores were swapped to lean human
(`llm_score=0.1, stylometric_score=0.5`, base=0.3), confidence ≈ 0.5 - 0.2*0.6 = 0.38, still
inside the "likely human" band. The same amount of signal disagreement is forgiven on the human
side and punished on the AI side — by design.

## 2. Uncertainty Representation

- A confidence score of 0.6 means: the system leans toward AI-generated but not strongly enough to assert it confidently — this should land in (or near) the "uncertain" band, not "likely AI."
- Mapping approach: raw signal outputs (`llm_score`, `stylometric_score`) are combined with the asymmetric rule above (described in Section 1) — agreement between signals pushes toward a confident verdict in either direction; disagreement pushes toward "uncertain."
- Exact threshold values separating likely AI / uncertain / likely human:
  - `confidence >= 0.75` → **likely_ai**
  - `0.40 < confidence < 0.75` → **uncertain**
  - `confidence <= 0.40` → **likely_human**
  - The "likely AI" boundary sits 0.25 from the center (0.5), while the "likely human" boundary
    sits only 0.10 from the center. This means it takes much stronger, more decisive evidence to
    land in "likely AI" than to land in "likely human" — directly reflecting the false-positive
    asymmetry: an innocent human writer is more likely to fall into "uncertain" (where they can
    appeal) than be confidently mislabeled "likely AI."
  - These thresholds will be sanity-checked against the 4 calibration inputs in Milestone 4
    (clear AI, clear human, 2 borderline) before being treated as final.

## 3. Transparency Label Design

- Exact text for a high-confidence AI result (`confidence >= 0.75`):
  > "This content was likely created with AI assistance."
- Exact text for a high-confidence human result (`confidence <= 0.40`):
  > "This content appears to be written by a human."
- Exact text for an uncertain result (`0.40 < confidence < 0.75`):
  > "We can't confidently tell whether this content was AI-assisted or human-written — read it with that uncertainty in mind."

All three are plain language with no technical jargon ("classifier output," "AI probability: 78%," "logit score" are explicitly avoided), and the uncertain label is phrased as an honest limitation of the system rather than a half-accusation of the creator.

## 4. Appeals Workflow

- Who can submit an appeal? The creator who submitted the content (identified by matching `content_id` from their own earlier `/submit` response — no separate appeal ID or lookup-by-search feature; kept intentionally simple).
- What information do they provide? `content_id` and their reasoning for disagreeing with the label (`creator_reasoning`).
- What does the system do when an appeal is received? Looks up the submission by `content_id`, updates its status from `"classified"` to `"under_review"`, and records the creator's reasoning alongside the original classification entry in the audit log (same record, not a separate one).
- Only one appeal is allowed per `content_id` (no re-appeal/edit flow) — keeps the system simple, matches the spec's framing of appeal as a one-time "flag for human review" action rather than an ongoing dialogue.
- What would a human reviewer see when they open the appeal queue? Each `under_review` entry showing the original text's attribution, confidence, both signal scores, and the creator's appeal reasoning side by side — everything needed to judge whether the original verdict was fair, with no automated re-classification required.

## 5. Anticipated Edge Cases

- **Edge case 1 — false positive on a non-native English speaker's formal writing**: A non-native English speaker submits a careful, grammatically consistent blog post. Signal 2 (stylometric) sees low sentence-length variance and consistent vocabulary — patterns that skew toward "AI-like" — and scores it high on the AI axis. Signal 1 (LLM) may also lean slightly AI-suspicious due to the formality, even though the content includes genuine personal anecdotes. Naively averaging the two signals could push this into "high-confidence AI" — the most damaging outcome for an innocent human writer. The asymmetric combination rule (Section 1) is designed specifically to catch this case and land it in "uncertain" instead, with a clear path to appeal.
- **Edge case 2**: a poem or short piece with heavy repetition and simple, deliberate vocabulary (a stylistic choice) that the stylometric heuristics might score as AI-generated due to low lexical diversity, even though it's a clear human creative choice.

## API Contract

### `POST /submit`

Request:

```json
{
  "text": "string, the content to analyze",
  "creator_id": "string"
}
```

Response:

```json
{
  "content_id": "uuid",
  "attribution": "likely_ai | uncertain | likely_human",
  "confidence": 0.78,
  "label": "exact transparency label text shown to the reader",
  "signals": {
    "llm_score": 0.81,
    "stylometric_score": 0.74
  },
  "status": "classified"
}
```

### `GET /log`

Request: none.

Response:

```json
{
  "entries": [
    {
      "content_id": "uuid",
      "creator_id": "string",
      "timestamp": "ISO 8601",
      "attribution": "likely_ai | uncertain | likely_human",
      "confidence": 0.78,
      "llm_score": 0.81,
      "stylometric_score": 0.74,
      "status": "classified | under_review",
      "appeal_reasoning": "string | null"
    }
  ]
}
```

### `POST /appeal`

Request:

```json
{
  "content_id": "uuid, must match an existing submission",
  "creator_reasoning": "string, the creator's explanation"
}
```

Response:

```json
{
  "content_id": "uuid",
  "status": "under_review",
  "message": "Appeal received and logged for review"
}
```

Design notes:

- `signals` is nested in the `/submit` response so individual scores are visible alongside the combined `confidence` — satisfies the rubric's "individual signal scores shown alongside the combined score."
- The audit log entry shape and the `/submit` response shape are nearly identical — one record format covers both, no separate serialization logic needed.
- `appeal_reasoning` starts `null` and gets filled in by `/appeal` on the same record — not a separate log entry — so the appeal stays visible alongside the original decision.

## Architecture

```
SUBMISSION FLOW
===============

   Creator  <<------------------------------------+
       |                                          |
       |  POST /submit { text, creator_id }       |
       v                                          |
  [Flask app]                                      |
       |                                          |
       v                                          |
  [Rate Limiter]                                   |
       |                                          |
       | too many requests in too short a time?    |
       | -- yes >> 429 response >>------------------+
       |
       | no
       |
       | raw text          raw text
       +-----------+-----------+
                   v                       v
        [Signal 1: Groq LLM]    [Signal 2: Stylometric heuristics]
                   |                       |
              llm_score             stylometric_score
                   |                       |
                   +-----------+-----------+
                               v
                   [Confidence Scorer]
                               |
                               | llm_score + stylometric_score -> combined confidence
                               v
                   [Label Generator]
                               |
                               | combined confidence -> label text
                               v
                   [Audit Logger]
                               |
                               | writes entry to audit log
                               v
                   content_id, attribution, confidence, label
                               |
                               v
                   Creator (success response)


APPEAL FLOW
===========

Creator
  |  POST /appeal { content_id, creator_reasoning }
  v
[Flask app]
  |
  v
[Content Store] -- finds matching submission by content_id, sets status = "under_review"
  |
  v
[Audit Logger] -- appends appeal info onto the existing entry (single appeal per content_id)
  |
  | content_id, status, confirmation message
  v
Creator (appeal response)

  ... later ...

Grader/Reviewer
  |  GET /log
  v
[Audit Logger] -- returns all entries, original decision + appeal_reasoning + status together
```

**Narrative**: A submission flows from the creator through the rate limiter — which either short-circuits straight back to the creator with a 429 if they've made too many requests too quickly, or lets the text through to two independent signals (Groq LLM + stylometric heuristics) running on the same raw input. Their scores are combined asymmetrically into one confidence score, mapped to a label, logged, and returned to the creator (who also receives the `content_id` needed for any future appeal, even though a casual reader on the platform only ever sees the label text). An appeal later looks up that same `content_id`, flips status to `under_review`, and appends the creator's reasoning to the original log entry — visible together to anyone hitting `GET /log`.

## AI Tool Plan

For each implementation milestone, specify which spec sections you'll provide, what you'll ask the AI tool to generate, and how you'll verify the output.

- **M3 (submission endpoint + first signal)**:
  - Spec sections provided: Section 1 (Signal 1 — LLM-based classification) + the Architecture diagram.
  - What's requested: a Flask app skeleton with a `POST /submit` route stub, plus a standalone `get_llm_score(text)` function that calls Groq with `response_format={"type": "json_object"}` and returns `(llm_score, reasoning)`, matching the exact JSON shape in Section 1.
  - Verification: call `get_llm_score()` directly (no Flask) on 2-3 sample inputs and confirm it returns valid floats in `[0, 1]` and doesn't crash on a malformed Groq response before wiring it into the route.

- **M4 (second signal + confidence scoring)**:
  - Spec sections provided: Section 1 (Signal 2 — stylometric heuristics) + Section 2 (Uncertainty Representation, including the `combine_scores` formula and thresholds) + the Architecture diagram.
  - What's requested: a standalone `get_stylometric_score(text)` function (pure Python — no nltk/textstat/spacy, per CLAUDE.md guardrail #4) computing sentence-length variance, type-token ratio, and punctuation density, plus the `combine_scores(llm_score, stylometric_score)` function exactly as specified in Section 1.
  - Verification: run the 4 calibration inputs from the project spec (clear AI, clear human, 2 borderline) through both signals and `combine_scores`, and manually check that the AI-leaning branch is harder to trigger than the human-leaning branch, per the worked example in Section 1 — correct any divergence from the documented formula before integrating.

- **M5 (production layer — labels, appeals)**:
  - Spec sections provided: Section 3 (Transparency Label Design, exact text) + Section 4 (Appeals Workflow) + the Architecture diagram (appeal flow).
  - What's requested: a `get_label(confidence)` function returning the exact verbatim label strings from Section 3 for all three bands, and the `POST /appeal` endpoint per the API Contract (`/appeal`) that looks up by `content_id`, flips `status` to `under_review`, and appends `creator_reasoning` to the existing log entry (never a new entry — CLAUDE.md guardrail #6).
  - Verification: call `get_label()` with a confidence value in each of the three bands and diff the output against the verbatim strings in Section 3; submit a test appeal and confirm via `GET /log` that the same `content_id`'s entry now shows `status: "under_review"` with `appeal_reasoning` populated, rather than a second entry appearing.

## Stretch Features (update before starting any)

- [ ] Ensemble detection
- [ ] Provenance certificate
- [ ] Analytics dashboard
- [ ] Multi-modal support
