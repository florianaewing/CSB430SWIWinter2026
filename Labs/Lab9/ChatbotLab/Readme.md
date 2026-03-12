# Local Ollama Chatbot Lab

# Learning Goasl

- Understand how LLMs interface with a web client
- Explore building an API using Flask
- Implement a client application using HTML and CSS
- Observe how JavaScript handles HTTP requests
- Understand the request–response cycle
- Make a request from a frontend client to a backend API
- Implement Ollama in a Python application

---

## Step 1. Install Ollama

Download and install Ollama:

https://ollama.com/download

---

## Step 2. Open Your Terminal

### Windows

Use **Git Bash** or **WSL**.
Do **not** use PowerShell or the default Windows Command Prompt.

If you are using VS Code, it will likely default to PowerShell.
Double-check that your terminal is set to **Git Bash**.

### Mac / Linux

Use your regular terminal.

---

### Pull and Run a Model

Ollama is a tool that allows you to run a large language model (LLM) locally on your machine. For this lab, we will use a very small model to ensure it runs on most computers without performance issues.

In this example, we will first pull then run the model directly in the terminal to verify that everything is working before integrating it into a Python application.

In your terminal, run:

    ollama run tinyllama

This will download the `tinyllama` model and start it.

You may choose a larger model if your computer can handle it, but `tinyllama` should work on most machines. If it is still too large for your system, contact your instructor as soon as possible.

---

## Step 3. Test the Model

Once it launches, try a prompt like:

    What should I eat for lunch?

If it responds, it works.

Exit using:

    /bye

or press:

    Ctrl + D

---

## Step 4. Set Up Your Python Environment

To run this locally, we need to set up a Python environment.

The first two commands create and activate a virtual environment for running Python.  
The third command installs the `ollama` package using `pip`, Python’s package manager.  
The final command, `pip freeze`, creates a `requirements.txt` file that records the installed libraries and their versions.

In your terminal:

    python3 -m venv venv
    source venv/bin/activate
    pip install ollama
    pip freeze > requirements.txt

---

## Step 5. Use Ollama in Python

Lets first make sure we can get this working in python at all.

Create a file called:

    app.py

Add the following:

```
    from ollama import chat

    response = chat(
        model='tinyllama',
        messages=[{'role': 'user', 'content': 'Hello!'}],
    )

    print(response.message.content)
```

Run:

    `python3 app.py` in the terminal

If it prints a response, everything is working.

---

## Step 6. Turn Your App Into an API

### Let’s Talk About Web APIs

Web pages use a pattern called the **request–response cycle**.  
A frontend client application (like a web page) sends a **request** to a backend server for data. The backend server then processes that request and sends back a **response** with requested data.

[You can learn more about this cycle here](https://dev.to/marlinekhavele/http-requestresponse-cycle-mb6)

This means we need to create two things:

1. A backend server that accepts requests from the frontend and sends responses
2. A frontend client that sends requests to the backend

Let’s start with the backend API. We will be using Flask for this. Make sure to read the comments in the code, since they explain what is happening.

Install additional dependencies in the terminal:

    pip install flask flask-cors
    ollama pull tinyllama
    pip freeze > requirements.txt

Replace your `app.py` code with the following and read through the comments:

```
# These are the libraries we are using.
from flask import Flask, request, jsonify
from flask_cors import CORS
from ollama import chat

# Flask is a lightweight web framework that lets us build an API.
# This line initializes our Flask application.
app = Flask(__name__)

# CORS (Cross-Origin Resource Sharing) is a browser security feature.
# Since our frontend and backend may run on different ports,
# enabling CORS allows the browser to communicate with our API.
CORS(app)

# In order for our API to communicate, it needs a route.
# A route is like a web address that points to a specific function.
# The base address is where the server is hosted,
# and "/chat" is the endpoint.
# Example: http://127.0.0.1:5000/chat
#
# Every route also specifies an HTTP method.
# POST means the client is sending data to the server.
@app.route("/chat", methods=["POST"])
def chat_route():

    # The incoming data is sent as JSON.
    # Flask automatically parses it for us.
    data = request.json

    # Extract the user's message from the JSON object.
    # The key "message" is defined in script.js.
    user_message = data["message"]

    # Call our local LLM using Ollama and pass it the user_message
    response = chat(
        model="tinyllama",
        messages=[{"role": "user", "content": user_message}],
    )

    # Convert the response back into JSON and send it to the client.
    return jsonify({
        "reply": response.message.content
    })

if __name__ == "__main__":
    app.run(debug=True)
```

---

## Step 7. Test Your Server

Run in the terminal:

    `python3 app.py`

The server will start, but nothing will happen yet because we have not built the frontend.

---

## Step 8. Build the Frontend

If you haven't done the Kahn acadmy content yet, STOP and go do that first.

- FIRST: Read the script.js file. I can't teach you JS in a week but I can walk you through what's happening.

Create a file called:

    `index.html`

Add the following:

```
<!-- This tells the browser what type of document we are using.
     Once upon a time, there were multiple document types for the web,
     but that is a story for another time. -->
<!doctype html>

<!-- The rest of our file is structured as a nested tree.
     This is our root (base) tag.
     The "lang" attribute specifies the language of the document. -->
<html lang="en">
  <head>
    <!-- The head contains metadata.
         This is information about the document that is not displayed in the browser. -->
    <meta charset="UTF-8" />
    <title>My First Chatbot</title>

    <!-- JavaScript files should load after the HTML is parsed.
         The "defer" attribute tells the browser to wait until
         the HTML document has finished loading before running script.js.

         Why use defer?
         Some of our JavaScript functions look for elements in the HTML.
         If the script runs before those elements exist,
         it will be trying to access an empty document. -->
    <script src="script.js" defer></script>
  </head>

  <!-- The body contains everything that will be displayed in the browser. -->
  <body>
    <h1>Name Your Chatbot Here</h1>

    <!-- This input field allows the user to type a message. -->
    <input type="text" id="chat-message" placeholder="Ask me anything" />

    <!-- This button triggers the sendMessage function in our JavaScript file. -->
    <button onclick="sendMessage()">Send</button>

    <!-- This is where the model's response will appear. -->
    <pre id="output"></pre>
  </body>
</html>

```

---

## Step 9. Test Everything

If you do not have the **Live Server** extension installed in VS Code, install it now.

1. Click “Go Live” in the bottom right corner.
2. Your HTML page will open in the browser.
3. Enter a question and click Send.
4. Wait for the response.

It may take a minute the first time, but it should not take more than a few seconds on most machines.

---

## Step 10. Make It Pretty

Now improve your interface.

### Add CSS

- Change colors
- Modify fonts
- Adjust layout
- Improve spacing

Try to do this without AI first, but you may use it if needed.

### Add More HTML

Enhance your page by adding:

- Instructions for how to use your chatbot
- A list of suggested prompts
- Additional layout elements
- Anything else that improves usability

---

### AI Transparency Statement

AI was used as a drafting and editing aid, all instructional design decisions,examples and learning objectives were determined by the instructor.
