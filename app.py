import os
import uuid

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import audit
from scoring import combine_scores, get_attribution, get_label
from signals import get_llm_score, get_stylometric_score

os.makedirs("logs", exist_ok=True)
audit.init_db()

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    creator_id = data.get("creator_id")

    if not text or not creator_id:
        return jsonify({"error": "Both 'text' and 'creator_id' are required."}), 400

    llm_score, _reasoning = get_llm_score(text)
    stylometric_score, _metrics = get_stylometric_score(text)
    confidence = combine_scores(llm_score, stylometric_score)
    attribution = get_attribution(confidence)
    label = get_label(attribution)

    content_id = str(uuid.uuid4())
    audit.log_submission(
        content_id=content_id,
        creator_id=creator_id,
        attribution=attribution,
        confidence=confidence,
        llm_score=llm_score,
        stylometric_score=stylometric_score,
    )

    return jsonify(
        {
            "content_id": content_id,
            "attribution": attribution,
            "confidence": confidence,
            "label": label,
            "signals": {
                "llm_score": llm_score,
                "stylometric_score": stylometric_score,
            },
            "status": "classified",
        }
    )


@app.route("/appeal", methods=["POST"])
def appeal():
    data = request.get_json(silent=True) or {}
    content_id = data.get("content_id")
    creator_reasoning = data.get("creator_reasoning")

    if not content_id or not creator_reasoning:
        return jsonify({"error": "Both 'content_id' and 'creator_reasoning' are required."}), 400

    updated = audit.record_appeal(content_id, creator_reasoning)
    if not updated:
        return jsonify({"error": "No classified submission found for that content_id, or it has already been appealed."}), 404

    return jsonify(
        {
            "content_id": content_id,
            "status": "under_review",
            "message": "Appeal received and logged for review",
        }
    )


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": audit.get_log()})


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)
