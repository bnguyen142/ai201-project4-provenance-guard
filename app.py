import os
import uuid

from flask import Flask, jsonify, render_template_string, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import audit
from scoring import combine_scores, get_attribution, get_label
from signals import get_llm_score, get_marker_score, get_stylometric_score

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
    marker_score, _marker_metrics = get_marker_score(text)
    confidence = combine_scores(llm_score, stylometric_score, marker_score)
    attribution = get_attribution(confidence)
    label = get_label(attribution)

    content_id = str(uuid.uuid4())
    audit.log_submission(
        content_id=content_id,
        creator_id=creator_id,
        text=text,
        attribution=attribution,
        confidence=confidence,
        llm_score=llm_score,
        stylometric_score=stylometric_score,
        marker_score=marker_score,
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
                "marker_score": marker_score,
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


_ANALYTICS_TEMPLATE = """
<!doctype html>
<html>
<head><title>Provenance Guard — Analytics</title></head>
<body style="font-family: sans-serif; max-width: 640px; margin: 2rem auto;">
  <h1>Provenance Guard — Analytics</h1>
  <p>Total submissions: <strong>{{ data.total_submissions }}</strong></p>

  <h2>Detection Pattern</h2>
  <table border="1" cellpadding="6" style="border-collapse: collapse; width: 100%;">
    <tr><th>Attribution</th><th>Count</th><th>Percentage</th></tr>
    {% for label, stats in data.detection_pattern.items() %}
    <tr><td>{{ label }}</td><td>{{ stats.count }}</td><td>{{ stats.percentage }}%</td></tr>
    {% endfor %}
  </table>

  <h2>Appeal Rate</h2>
  <p><strong>{{ (data.appeal_rate * 100) | round(1) }}%</strong> of submissions have been appealed.</p>

  <h2>Average Confidence</h2>
  <p>Mean confidence score across all submissions: <strong>{{ data.average_confidence }}</strong></p>

  <p><a href="/log">View raw audit log (JSON)</a></p>
</body>
</html>
"""


@app.route("/analytics", methods=["GET"])
def analytics():
    data = audit.get_analytics()
    if request.args.get("format") == "json":
        return jsonify(data)
    return render_template_string(_ANALYTICS_TEMPLATE, data=data)


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)
