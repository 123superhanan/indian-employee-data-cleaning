import tkinter as tk
import os
from difflib import get_close_matches

APPS = {
    "spotify": r"c:\Users\Admin\OneDrive\Desktop\Spotify.lnk",
    "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "calculator": "calc.exe",
    "notepad": "notepad.exe",
    "vs code": r"C:\Users\Admin\AppData\Local\Programs\Microsoft VS Code\Code.exe"
}

def open_app():
    user_input = entry.get().lower()

    match = get_close_matches(user_input, APPS.keys(), n=1, cutoff=0.6)
    if match:
        os.startfile(APPS[match[0]])
        status_label.config(text=f"Opened {match[0]}")
    else:
        status_label.config(text="App not found")

root = tk.Tk()
root.title("My PC Assistant")
root.geometry("400x200")
root.resizable(False, False)

tk.Label(root, text="Type app name", font=("Segoe UI", 12)).pack(pady=10)

entry = tk.Entry(root, font=("Segoe UI", 12))
entry.pack(pady=5)

tk.Button(
    root,
    text="Open App",
    font=("Segoe UI", 11),
    command=open_app
).pack(pady=10)

status_label = tk.Label(root, text="", font=("Segoe UI", 10))
status_label.pack()

root.mainloop()


# pyinstaller --onefile --windowed launcher_app.py
