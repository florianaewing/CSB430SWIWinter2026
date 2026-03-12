async function sendMessage() {
  const input = document.querySelector("#chat-message");
  const message = input.value;

  if (!message) return;

  const response = await fetch("http://127.0.0.1:5000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message: message,
    }),
  });

  const data = await response.json();
  console.log(data);

  document.getElementById("output").textContent = data.reply;

  input.value = "";
}
