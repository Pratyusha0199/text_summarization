from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask-cors
from transformers import pipeline
# Initialize the Flask app and enable CORS
app = Flask(__name__)
CORS(app)  # This allows all origins to access the Flask app

# Initialize the summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route("/summarize", methods=["POST"])
def summarize():
    input_data = request.get_json(silent=True)
    if not input_data or "text" not in input_data:
        return jsonify({"error": "Invalid input or no text provided"}), 400

    text = input_data.get("text", "")
    if len(text) > 100024:
        return jsonify({"error": "Input text is too long"}), 400

    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return jsonify({"summary": summary[0]["summary_text"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

