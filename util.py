import sqlite3
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk


# --------------------- UI ELEMENT HELPERS ---------------------

def get_button(window, text, color, command, fg='white'):
    return tk.Button(
        window,
        text=text,
        command=command,
        fg=fg,
        bg=color,
        activebackground="black",
        activeforeground="white",
        height=2,
        width=20,
        font=('Helvetica bold', 20)
    )

def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label

def get_text_label(window, text):
    label = tk.Label(window, text=text, font=("sans-serif", 21), justify="left")
    return label

def get_entry_text(window):
    return tk.Text(window, height=2, width=15, font=("Arial", 20))

def msg_box(title, description):
    messagebox.showinfo(title, description)

# --------------------- FACE RECOGNITION ---------------------

def recognize(img, db_path="face_data.db"):
    embeddings_unknown = face_recognition.face_encodings(img)
    if not embeddings_unknown:
        return 'no_persons_found'

    embeddings_unknown = embeddings_unknown[0]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, embedding FROM users")
    registered_users = cursor.fetchall()
    conn.close()

    for student_id, embedding_blob in registered_users:
        stored_encoding = np.frombuffer(embedding_blob, dtype=np.float64)
        match = face_recognition.compare_faces([stored_encoding], embeddings_unknown)[0]
        if match:
            return student_id

    return 'unknown_person'

# --------------------- ATTENDANCE LOGS ---------------------

def get_attendance_logs(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, action, timestamp FROM attendance ORDER BY timestamp DESC")
    logs = cursor.fetchall()
    conn.close()
    return logs

# --------------------- LECTURER PANEL UI ---------------------

def build_lecturer_panel(parent, on_register, on_show_attendance):
    parent.title("Lecturer Panel")
    parent.geometry("600x400")
    parent.config(bg="#eaf6f6")

    # Title
    title = tk.Label(
        parent,
        text="📘 Lecturer Control Panel",
        font=("Segoe UI", 22, "bold"),
        bg="#eaf6f6",
        fg="#0a3d62"
    )
    title.pack(pady=30)

    # Button Frame
    frame = tk.Frame(parent, bg="#eaf6f6")
    frame.pack(pady=10)

    # Register Button
    tk.Button(
        frame,
        text="➕ Register New User",
        font=("Segoe UI", 14),
        bg="#636e72",
        fg="white",
        activebackground="#2d3436",
        padx=20,
        pady=10,
        width=25,
        command=on_register
    ).pack(pady=10)

    # Show Attendance Button
    tk.Button(
        frame,
        text="📋 Show Attendance",
        font=("Segoe UI", 14),
        bg="#00b894",
        fg="white",
        activebackground="#019875",
        padx=20,
        pady=10,
        width=25,
        command=on_show_attendance
    ).pack(pady=10)

    # Reward Dashboard Button
    tk.Button(
        frame,
        text="🎁 View Reward Dashboard",
        font=("Segoe UI", 14),
        bg="#6c5ce7",
        fg="white",
        activebackground="#341f97",
        padx=20,
        pady=10,
        width=25,
        command=show_reward_dashboard  # ✅ this opens the reward dashboard
    ).pack(pady=10)


def create_animated_emoji(main_window, play_sound_callback=None, x=700, y=500, emoji="😄"):
    font_size = 50
    label = tk.Label(
        main_window,
        text=emoji,
        font=("Arial", font_size, "bold"),
        fg="green",
        bg="white"
    )
    label.place(x=x, y=y)

    if play_sound_callback:
        play_sound_callback()

    # Animation settings
    steps = 10
    duration = 500
    interval = duration // steps
    bounce_height = 10

    def animate(step=0):
        if step <= steps:
            # Fade in + bounce
            scale = 1.0 + 0.05 * (1 - abs(step - steps // 2) / (steps // 2))
            size = int(font_size * scale)
            offset = int(bounce_height * (1 - abs(step - steps // 2) / (steps // 2)))
            label.config(font=("Arial", size, "bold"))
            label.place(x=x, y=y - offset)
            main_window.after(interval, lambda: animate(step + 1))
        elif step <= 2 * steps:
            # Fade out
            fade_step = step - steps
            gray = int(255 * (fade_step / steps))
            label.config(fg=f"#{gray:02x}{gray:02x}{gray:02x}")
            main_window.after(interval, lambda: animate(step + 1))
        else:
            label.destroy()

    animate()


def show_reward_dashboard():
    window = tk.Toplevel()
    window.title("🎁 Reward Dashboard")
    window.geometry("700x400")
    window.configure(bg="#f5f6fa")

    title = tk.Label(
        window,
        text="🎓 Student Reward Summary",
        font=("Segoe UI", 20, "bold"),
        bg="#f5f6fa",
        fg="#2d3436"
    )
    title.pack(pady=10)

    # Table frame
    table_frame = tk.Frame(window)
    table_frame.pack(fill="both", expand=True)

    # Table with Scrollbar
    tree = ttk.Treeview(table_frame, columns=("ID", "Tokens", "NFT Earned", "Wallet"), show="headings")
    tree.heading("ID", text="Student ID")
    tree.heading("Tokens", text="Tokens")
    tree.heading("NFT Earned", text="NFT Awarded")
    tree.heading("Wallet", text="Wallet Address")

    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Fetch data from DB
    conn = sqlite3.connect("face_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, attendance_days, nft_awarded, wallet_address FROM users")
    data = cursor.fetchall()
    conn.close()

    # Insert into table
    for student_id, tokens, nft, wallet in data:
        tree.insert("", "end", values=(student_id, tokens, "Yes" if nft else "No", wallet or "N/A"))
