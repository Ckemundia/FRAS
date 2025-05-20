import sqlite3
import face_recognition
import numpy as np
import tkinter as tk
import platform
import subprocess
from tkinter import messagebox
from tkinter import ttk
from tkinter import simpledialog


from blockchain_utils import send_token_to_wallet as _send_token_to_wallet
from blockchain_utils import mint_student_nft as _mint_student_nft

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

def create_blinking_label(parent, text, font, fg, bg, x, y, interval=500):
    label = tk.Label(parent, text=text, font=font, fg=fg, bg=bg)
    label.place(x=x, y=y)

    def blink():
        current_color = label.cget("fg")
        label.config(fg=bg if current_color != bg else fg)
        parent.after(interval, blink)

    blink()
    return label
def _create_rounded_button(parent, text, fg, bg, command):
    button = tk.Button(
        parent,
        text=text,
        fg=fg,
        bg=bg,
        activebackground="#4d4dff",
        activeforeground="white",
        font=("Segoe UI", 12, "bold"),
        bd=0,
        relief="flat",
        padx=15,
        pady=5,
        cursor="hand2"
    )
    # Rounded effect workaround (you can't truly round buttons in plain Tkinter)
    button.configure(highlightthickness=0, borderwidth=0)
    return button

def add_hover_effect(widget, normal_bg, hover_bg):
    def on_enter(e):
        widget['background'] = hover_bg
    def on_leave(e):
        widget['background'] = normal_bg
    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)


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
    cursor.execute("SELECT student_id, date, time FROM attendance ORDER BY date DESC, time DESC")
    logs = cursor.fetchall()
    conn.close()
    return logs


# --------------------- LECTURER PANEL UI ---------------------

def build_lecturer_panel(parent, on_register, on_show_attendance, on_back):

    parent.title("Lecturer Panel")
    parent.geometry("600x600")
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
        command=show_reward_dashboard
    ).pack(pady=10)

    # Fixed Token and NFT buttons
    tk.Button(
        frame,
        text="💰 Send Token",
        font=("Segoe UI", 14),
        bg="#fdcb6e",
        fg="black",
        width=25,
        command=lambda: _send_token_gui(parent)
    ).pack(pady=10)

    tk.Button(
        frame,
        text="🎨 Mint NFT",
        font=("Segoe UI", 14),
        bg="#00cec9",
        fg="black",
        width=25,
        command=lambda: _mint_nft_gui(parent)
    ).pack(pady=10)
    # Back Button
    back_button = tk.Button(
        parent,  # or use a frame if you want it inside one
        text="🔙Back",
        font=("Segoe UI", 10),  # smaller font
        bg="#d63031",
        fg="white",
        activebackground="#c0392b",
        padx=5,
        pady=3,
        width=10,  # smaller width
        command=on_back
    )
    back_button.place(x=10, y=20)  # position it exactly


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

    # Make sure necessary columns exist
    cursor.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cursor.fetchall()]
    if "attendance_days" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN attendance_days INTEGER DEFAULT 0")
    if "nft_awarded" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN nft_awarded INTEGER DEFAULT 0")
    if "wallet" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN wallet TEXT")

    conn.commit()

    # Update attendance_days based on attendance table
    cursor.execute("SELECT student_id, COUNT(*) FROM attendance GROUP BY student_id")
    attendance_counts = dict(cursor.fetchall())
    for sid, count in attendance_counts.items():
        cursor.execute("UPDATE users SET attendance_days=? WHERE student_id=?", (count, sid))
    conn.commit()

    cursor.execute("SELECT student_id, attendance_days, nft_awarded, wallet FROM users")
    data = cursor.fetchall()
    conn.close()

    # Insert into table
    for student_id, tokens, nft, wallet in data:
        tree.insert("", "end", values=(student_id, tokens, "Yes" if nft else "No", wallet or "N/A"))


# --------------------- EMOJI ANIMATION ---------------------

def create_animated_emoji(main_window, play_sound_callback=None, x=350, y=500, emoji="😄"):
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

    steps = 10
    duration = 500
    interval = duration // steps
    bounce_height = 50

    def animate(step=0):
        if step <= steps:
            scale = 1.0 + 0.05 * (1 - abs(step - steps // 2) / (steps // 2))
            size = int(font_size * scale)
            offset = int(bounce_height * (1 - abs(step - steps // 2) / (steps // 2)))
            label.config(font=("Arial", size, "bold"))
            label.place(x=x, y=y - offset)
            main_window.after(interval, lambda: animate(step + 1))
        elif step <= 2 * steps:
            fade_step = step - steps
            gray = int(255 * (fade_step / steps))
            label.config(fg=f"#{gray:02x}{gray:02x}{gray:02x}")
            main_window.after(interval, lambda: animate(step + 1))
        else:
            label.destroy()

    animate()

def play_success_sound(self):
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            elif platform.system() == "Darwin":  # macOS
                subprocess.call(['afplay', '/System/Library/Sounds/Glass.aiff'])
            else:  # Linux
                subprocess.call(['aplay', '/usr/share/sounds/alsa/Front_Center.wav'])
        except:
            print("🔇 Could not play sound")

# --------------------- WEB3 WRAPPERS ---------------------

def manual_send_token_reward(wallet_address, amount=1):
    return _send_token_to_wallet(wallet_address, amount)

def mint_nft_if_eligible(wallet_address, token_uri):
    return _mint_student_nft(wallet_address, token_uri)

# --------------------- GUI HELPERS FOR WEB3 ---------------------

def prompt_for_student_id(parent):
    return simpledialog.askstring("Student ID", "Enter Student ID:", parent=parent)

def _send_token_gui(parent):
    sid = prompt_for_student_id(parent)
    if not sid:
        return

    conn = sqlite3.connect("face_data.db")
    cur = conn.cursor()
    cur.execute("SELECT wallet FROM users WHERE student_id=?", (sid,))
    row = cur.fetchone()
    conn.close()

    if not row or not row[0]:
        messagebox.showerror("Error", "No wallet found for that student.")
        return

    wallet = row[0]
    receipt = manual_send_token_reward(wallet, 1)
    if receipt:
        messagebox.showinfo("Token Sent", f"Tx hash:\n{receipt.transactionHash.hex()}")
    else:
        messagebox.showerror("Error", "Token transfer failed.")

def _mint_nft_gui(parent):
    sid = prompt_for_student_id(parent)
    if not sid:
        return

    conn = sqlite3.connect("face_data.db")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM attendance WHERE student_id=?", (sid,))
    count = cur.fetchone()[0]
    cur.execute("SELECT nft_awarded, wallet FROM users WHERE student_id=?", (sid,))
    row = cur.fetchone()
    conn.close()

    if not row:
        messagebox.showerror("Error", "Student not found.")
        return

    nft_awarded, wallet = row
    if nft_awarded:
        messagebox.showinfo("Already Awarded", "This student already has an NFT.")
        return
    if count < 100:
        messagebox.showinfo("Not Eligible", f"Only {count} attendances — need 100.")
        return

    token_uri = simpledialog.askstring("Metadata URI", "Enter IPFS tokenURI:", parent=parent)
    if not token_uri:
        return

    receipt = mint_nft_if_eligible(wallet, token_uri)
    if receipt:
        conn = sqlite3.connect("face_data.db")
        cur = conn.cursor()
        cur.execute("UPDATE users SET nft_awarded=1 WHERE student_id=?", (sid,))
        conn.commit()
        conn.close()
        messagebox.showinfo("NFT Minted", f"Tx hash:\n{receipt.transactionHash.hex()}")
    else:
        messagebox.showerror("Error", "NFT minting failed.")
