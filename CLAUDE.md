# Provenance Guard — AI Code-Gen Guardrails

This project (CodePath AI201 Project 4) intentionally uses AI-assisted code generation per its
own spec — that's expected and graded. But AI tools have known failure patterns on this exact
shape of project. Check generated code against this list before accepting it. Read this file at
the start of any session touching this project.

Source of truth for design decisions: `planning.md`. If generated code conflicts with
`planning.md`, `planning.md` wins — fix the code, don't silently let the AI's assumption stand.

## Checklist — verify before accepting AI-generated code

1. **Confidence score must have two real threshold boundaries, not one.**
   "Uncertain" must be a genuine middle band (`low_threshold < score < high_threshold`), not a
   binary if/else with the uncertain label bolted on as an afterthought.

2. **Confidence scoring must be asymmetric, not a naive average.**
   Per `planning.md`: don't confidently label "likely AI" unless both signals strongly agree.
   Signal disagreement should pull the score toward "uncertain," not get averaged away. Check the
   actual formula, not just whether output numbers look plausible.

3. **Force structured/JSON output from the Groq call.**
   Free-text responses from the LLM signal will intermittently fail to parse. Use Groq's JSON
   mode / response format constraint, and handle parse failures without crashing the request.

4. **Stylometric heuristics must stay pure Python — no nltk/textstat/spacy.**
   AI tools default to these libraries out of habit. If an import shows up, rewrite with
   `str.split()`, `Counter`, basic math instead. This isn't a style preference — it's a
   requirements.txt and setup constraint.

5. **Audit log must be structured, never print() statements.**
   Every entry must be a JSON line or SQLite row, not stdout text. The rubric explicitly
   disqualifies unformatted console output.

6. **Appeals update the existing record in place — never create a new log entry.**
   One record per `content_id`. Status flips `classified` -> `under_review`, and
   `creator_reasoning` is added to that same entry. Appeal must remain visible alongside the
   original decision in `GET /log`.

7. **Label text must come from `planning.md` verbatim, never invented by the AI.**
   Paste the exact label strings into the prompt. Watch for jargon creeping in
   ("classifier output", "AI probability: 78%", "logit score") — labels must be plain language a
   non-technical reader understands.

8. **Flask-Limiter requires `storage_uri="memory://"` explicitly.**
   A lot of AI training data predates this Flask-Limiter 3.x requirement and will generate code
   that errors on startup. Use the exact pattern from the project spec.

9. **Single appeal per `content_id`, no lookup/search endpoint.**
   Don't let the AI add a "browse my submissions" feature or multi-appeal/edit flow — out of
   scope, adds complexity not required by the rubric.

10. **API response shapes must match the contract already decided in `planning.md`** —
    `content_id`, `attribution`, `confidence`, `label`, nested `signals: {llm_score,
    stylometric_score}` for `/submit`; don't let the AI rename or restructure fields ad hoc.

## When in doubt

If generated code makes a design choice not covered here, check it against `planning.md` section
by section before accepting it. If `planning.md` is silent on it too, that's a sign to update
`planning.md` first, then regenerate — not to let the AI's default stand uninspected.
