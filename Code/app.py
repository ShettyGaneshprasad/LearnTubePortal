from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

app = Flask(__name__)

# # Add this route for debugging purposes
# @app.route("/", methods=["GET"])
# def home():
#     return "Flask server is running!"
# Load the pre-trained zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_query(query):
    # Define the candidate labels
    candidate_labels = ["Learning", "Technology", "Other"]

    # Perform zero-shot classification
    result = classifier(query, candidate_labels=candidate_labels)

    # Check the highest score and corresponding label
    top_label = result['labels'][0]
    top_score = result['scores'][0]

    # Return "W" for learning/technology-related, "L" otherwise
    if top_label in ["Learning", "Technology"] and top_score > 0.5:
        print(top_label,top_score)
        return "W"  # Learning/Technology-related
    print(top_label,top_score)
    return "L"  # Non-learning-related

# Render the HTML page when accessing root URL

@app.route("/")
def home():
    return render_template("index.html")




@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query not provided"}), 400
    classification = classify_query(query)
    return jsonify({"classification": classification})

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If query is passed as a command-line argument, classify it directly
        query = sys.argv[1]
        classification = classify_query(query)
        print(f"Query: {query}")
        print(f"Classification: {classification}")
    else:
        # Otherwise, run Flask app
        app.run(debug=True)
