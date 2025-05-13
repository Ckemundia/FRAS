import sqlite3
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import messagebox


def get_button(window, text, color, command, fg='white'):
    return tk.Button(
        window, text=text, activebackground="black", activeforeground="white",
        fg=fg, bg=color, command=command, height=2, width=20, font=('Helvetica bold', 20)
    )


def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window):
    return tk.Text(window, height=2, width=15, font=("Arial", 20))


def msg_box(title, description):
    messagebox.showinfo(title, description)


def recognize(img, db_path="face_data.db"):
    embeddings_unknown = face_recognition.face_encodings(img)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found'
    embeddings_unknown = embeddings_unknown[0]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, embedding FROM users")
    registered_users = cursor.fetchall()
    conn.close()

    for student_id, embedding_blob in registered_users:
        stored_encoding = np.frombuffer(embedding_blob, dtype=np.float64)
        if face_recognition.compare_faces([stored_encoding], embeddings_unknown)[0]:
            return student_id

    return 'unknown_person'


def get_attendance_logs(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, action, timestamp FROM attendance ORDER BY timestamp DESC")
    logs = cursor.fetchall()
    conn.close()
    return logs
