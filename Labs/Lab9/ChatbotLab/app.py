from flask import Flask, request, jsonify
from flask_cors import CORS
from ollama import chat

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat_route():
    data = request.json
    user_message = data["message"]

    response = chat(
        model="tinyllama",
        messages=[{"role": "user", "content": user_message}],
    )

    return jsonify({"reply": response.message.content})

if __name__ == "__main__":
    app.run(debug=True)
