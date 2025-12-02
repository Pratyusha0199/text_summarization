from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)  
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    print("Error loading summarization model:", e)
    summarizer = None


@app.route("/summarize", methods=["POST"])
def summarize():

    if summarizer is None:
        return jsonify({"error": "Summarization model is not available."}), 500

    input_data = request.get_json(silent=True)

    if not input_data or "text" not in input_data:
        return jsonify({"error": "Invalid input or no text provided."}), 400

    text = input_data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text is empty."}), 400

    MAX_CHARS = 4000
    if len(text) > MAX_CHARS:
        return jsonify({"error": f"Input text is too long. Limit is {MAX_CHARS} characters."}), 400

    try:
        summary = summarizer(
            text,
            max_length=120,  
            min_length=40,   
            do_sample=False
        )
        return jsonify({"summary": summary[0]["summary_text"]})
    except Exception as e:
        print("Error during summarization:", e)
        return jsonify({"error": "Failed to generate summary."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
