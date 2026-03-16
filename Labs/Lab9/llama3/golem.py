import tkinter as tk
import threading
from ollama import chat

history = []

def send_message(event=None):
    user_input = entry.get().strip()
    if not user_input:
        return
    entry.delete(0, tk.END)
    append_message("You", user_input)
    history.append({"role": "user", "content": user_input})
    send_btn.config(state=tk.DISABLED)
    entry.config(state=tk.DISABLED)
    threading.Thread(target=get_reply, daemon=True).start()

def get_reply():
    try:
        response = chat(model="llama3.2:3b", messages=history)
        reply = response.message.content
        history.append({"role": "assistant", "content": reply})
        root.after(0, lambda: append_message("Golem", reply))
    except Exception as e:
        root.after(0, lambda: append_message("Error", str(e)))
    finally:
        root.after(0, re_enable)

def re_enable():
    send_btn.config(state=tk.NORMAL)
    entry.config(state=tk.NORMAL)
    entry.focus()

def append_message(sender, text):
    chat_box.config(state=tk.NORMAL)
    chat_box.insert(tk.END, sender + ":" + chr(10) + text + chr(10) * 2)
    chat_box.config(state=tk.DISABLED)
    chat_box.see(tk.END)

root = tk.Tk()
root.title("Golem")
root.geometry("600x500")
root.resizable(True, True)

# Pack the input frame FIRST so tkinter reserves space for it
frame = tk.Frame(root, bg="#2b2b2b")
frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 8))

entry = tk.Entry(frame, font=("Segoe UI", 11), bg="#2b2b2b", fg="#e0e0e0",
                 insertbackground="white", relief=tk.FLAT)
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 6))
entry.bind("<Return>", send_message)
entry.focus()

send_btn = tk.Button(frame, text="Send", command=send_message,
                     bg="#444", fg="white", relief=tk.FLAT,
                     font=("Segoe UI", 11), padx=12)
send_btn.pack(side=tk.RIGHT)

# Chat box fills remaining space after the input frame is sized
chat_box = tk.Text(root, state=tk.DISABLED, wrap=tk.WORD, font=("Segoe UI", 11),
                   bg="#1e1e1e", fg="#e0e0e0", padx=8, pady=8, relief=tk.FLAT)
chat_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

append_message("Golem", "Hello! I am Golem. How can I help you today?")

root.mainloop()
