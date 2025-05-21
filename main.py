import os
import sqlite3
import datetime
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import mediapipe as mp
import numpy as np
import util
import csv
import threading
import time
import platform
import subprocess
from util import create_animated_emoji
from util import show_reward_dashboard
from util import manual_send_token_reward, mint_nft_if_eligible
from tkinter import font as tkfont
import re

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1000x600+350+100")
        self.main_window.title("Facial Recognition Attendance System")

        self.main_window.configure(bg="#eaf6f6")

        try:
            self.bg_canvas.destroy()
        except AttributeError:
            pass  # Canvas hasn't been created yet

        # Update feedback labels with matching background
        self.shadow_label = tk.Label(
            self.main_window,
            text="",
            font=("Segoe UI", 22, "bold"),
            fg="gray30",
            bg="#eaf6f6"
        )
        self.shadow_label.place(x=402, y=522)  # Keeping shadow offset

        self.attendance_feedback_label = tk.Label(
            self.main_window,
            text="",
            font=("Segoe UI", 22, "bold"),
            fg="#004d4d",  # Optional: darker text for contrast
            bg="#eaf6f6"
        )
        self.attendance_feedback_label.place(x=10, y=500)


        self.lecturer_panel_button = tk.Button(
            self.main_window, text="Lecturer Panel",
            bg="#009999", fg="white",
            font=("Arial", 12, "bold"),
            activebackground="#007f7f", activeforeground="white",
            relief="flat",
            command=self.open_lecturer_window
        )
        self.lecturer_panel_button.place(x=680, y=400, width=200, height=60)

        self.student_panel_button = tk.Button(
            self.main_window, text="Student Panel",
            bg="#009999", fg="white",
            font=("Arial", 12, "bold"),
            activebackground="#007f7f", activeforeground="white",
            relief="flat",
        )
        self.student_panel_button.place(x=680, y=500, width=200, height=60)

        # Webcam label background match
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.config(bg="white", bd=2, relief="groove")  # Optional subtle border
        self.webcam_label.place(x=10, y=0, width=600, height=400)

        # Update blinking label background to match
        self.hand_hint_label = util.create_blinking_label(
            self.main_window,
            text="👋 Raise your hand to sign",
            font=("Helvetica", 16, "bold"),
            fg="#007acc",
            bg="#eaf6f6",
            x=680,
            y=100
        )

        self.db_path = 'face_data.db'
        self.initialize_db()

        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

        self.user_embeddings = self.load_user_embeddings()

        self.cap = cv2.VideoCapture(0)

        self.running = True
        self.thread = threading.Thread(target=self.process_webcam)
        self.thread.daemon = True
        self.thread.start()
        self.last_face_check_time = 0
        self.face_check_interval = 3
        self.mp_drawing = mp.solutions.drawing_utils
        self.recently_marked = {}
        self.mark_cooldown = 60
        self.last_seen_encoding = None
        self.recently_marked = {}
        self.mark_cooldown = 60

    def _draw_gradient(self, canvas, width, height, color1, color2):
        r1, g1, b1 = self.main_window.winfo_rgb(color1)
        r2, g2, b2 = self.main_window.winfo_rgb(color2)
        r_ratio = (r2 - r1) / height
        g_ratio = (g2 - g1) / height
        b_ratio = (b2 - b1) / height
        for i in range(height):
            nr = int(r1 + (r_ratio * i))
            ng = int(g1 + (g_ratio * i))
            nb = int(b1 + (b_ratio * i))
            color = f'#{nr>>8:02x}{ng>>8:02x}{nb>>8:02x}'
            canvas.create_line(0, i, width, i, fill=color)

    def _create_rounded_button(self, master, text, fg, bg, command):
        btn = tk.Canvas(master, width=140, height=45, bg=master['bg'], highlightthickness=0)
        radius = 20
        width, height = 140, 45

        # Rounded rectangle background
        btn.create_arc(0, 0, radius*2, radius*2, start=90, extent=90, fill=bg, outline=bg)
        btn.create_arc(width-radius*2, 0, width, radius*2, start=0, extent=90, fill=bg, outline=bg)
        btn.create_arc(0, height-radius*2, radius*2, height, start=180, extent=90, fill=bg, outline=bg)
        btn.create_arc(width-radius*2, height-radius*2, width, height, start=270, extent=90, fill=bg, outline=bg)
        btn.create_rectangle(radius, 0, width-radius, height, fill=bg, outline=bg)
        btn.create_rectangle(0, radius, width, height-radius, fill=bg, outline=bg)

        text_item = btn.create_text(width//2, height//2, text=text, fill=fg, font=("Segoe UI", 13, "bold"))

        def on_click(event):
            command()

        def on_enter(event):
            btn.config(cursor="hand2")
            btn.itemconfig(text_item, fill="#ccccff")

        def on_leave(event):
            btn.config(cursor="")
            btn.itemconfig(text_item, fill=fg)

        btn.bind("<Button-1>", on_click)
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    def play_success_sound(self):
        util.play_success_sound(self)


    def initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create users table with name and wallet
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    student_id TEXT PRIMARY KEY,
                    name TEXT,
                    wallet TEXT,
                    embedding BLOB
                )
            ''')

            # Create attendance table with more fields
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    name TEXT,
                    date TEXT,
                    time TEXT,
                    unit TEXT,
                    FOREIGN KEY(student_id) REFERENCES users(student_id)
                )
            ''')

    def load_user_embeddings(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, embedding FROM users")
        rows = cursor.fetchall()
        conn.close()

        return [(sid, np.frombuffer(emb, dtype=np.float64)) for sid, emb in rows]

    def is_hand_raised(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(image_rgb)
        if results.multi_hand_landmarks:
            print("✅ Hand detected")
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                tip = hand_landmarks.landmark[12]
                if tip.y < wrist.y:
                    print("✅ Hand is raised")
                    return True
                else:
                    print("❌ Hand detected but not raised")
        else:
            print("❌ No hand detected")
        return False

    def process_webcam(self):
        while self.running:
            start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                continue

            self.most_recent_capture_arr = frame.copy()
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_rgb)

            # ✅ Visualize hand landmarks (for debug)
            image_for_drawing = frame.copy()
            results = self.hands_detector.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(image_for_drawing, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(image_for_drawing, cv2.COLOR_BGR2RGB)))

            def update_gui():
                self.webcam_label.imgtk = imgtk
                self.webcam_label.configure(image=imgtk)

            self.webcam_label.after(0, update_gui)

            # ✅ Run face recognition if hand is raised AND enough time has passed
            if self.is_hand_raised(frame):
                now = time.time()
                if now - self.last_face_check_time > self.face_check_interval:
                    self.last_face_check_time = now

                    face_locations = face_recognition.face_locations(img_rgb)
                    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

                    print(f"🔍 Found {len(face_encodings)} face(s)")

                    for encoding in face_encodings:
                        # ✅ Skip very similar encoding just processed
                        if self.last_seen_encoding is not None:
                            dist = np.linalg.norm(encoding - self.last_seen_encoding)
                            if dist < 0.3:
                                print("⚠️ Similar face already processed recently.")
                                continue

                        self.last_seen_encoding = encoding

                        for student_id, known_encoding in self.user_embeddings:
                            match = face_recognition.compare_faces([known_encoding], encoding)[0]
                            if match:
                                print(f"🎯 Face matched with {student_id}")
                                self.mark_attendance(student_id)
                                break
                            else:
                                print(f"❌ No match for {student_id}")

            print(f"⏱️ Frame time: {time.time() - start:.2f}s")
            time.sleep(0.05)

    def mark_attendance(self, student_id):
        now = datetime.datetime.now()

        #  Avoid duplicate marking
        last_mark = self.recently_marked.get(student_id)
        if last_mark and (now - last_mark).total_seconds() < self.mark_cooldown:
            print(f"⏳ {student_id} recently marked. Skipping.")
            self.show_attendance_feedback(f"⏳ Already marked recently")
            return

        # Connect to DB and get name + wallet
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, wallet FROM users WHERE student_id = ?", (student_id,))
        row = cursor.fetchone()

        if row:
            name, wallet_address = row
        else:
            print(f"❌ Student ID {student_id} not found in users table.")
            conn.close()
            return

        # Insert attendance
        cursor.execute("""
            INSERT INTO attendance (student_id, name, date, time, unit)
            VALUES (?, ?, ?, ?, ?)
        """, (
            student_id,
            name,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            "UnitPlaceholder"
        ))

        # Count attendance for NFT eligibility (do this *before* closing)
        cursor.execute("SELECT COUNT(*) FROM attendance WHERE student_id=?", (student_id,))
        count = cursor.fetchone()[0]

        conn.commit()
        conn.close()  # ✅ Close only once at the end of DB operations

        self.recently_marked[student_id] = now
        print(f"✅ Attendance marked for {name} ({student_id})")
        self.show_attendance_feedback(f"✅ Welcome {name}!")
        self.animate_success()

        #  Blockchain Token Reward
        if wallet_address:
            from util import manual_send_token_reward
            receipt = manual_send_token_reward(wallet_address, 1)
            if receipt:
                print(f"🎉 Token sent to {wallet_address}")
            else:
                print(f"❌ Failed to send token to {wallet_address}")
        else:
            print(f"⚠️ No wallet registered for {student_id}")

        # NFT Minting after 100 attendances
        if count >= 100:
            token_uri = "https://ipfs.io/ipfs/YOUR_TOKEN_URI.json"
            from util import mint_nft_if_eligible
            nft_receipt = mint_nft_if_eligible(wallet_address, token_uri)
            if nft_receipt:
                print("🖼️ NFT minted!")
            else:
                print("❌ NFT minting failed.")

    def animate_success(self, emoji="😄"):
        create_animated_emoji(self.main_window, self.play_success_sound, emoji=emoji)

    def open_lecturer_window(self):
        code_window = tk.Toplevel(self.main_window)
        code_window.title("Lecturer Login")

        window_width = 400
        window_height = 200

        # Get the screen's width and height
        screen_width = code_window.winfo_screenwidth()
        screen_height = code_window.winfo_screenheight()

        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))

        code_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        code_window.config(bg="#eaf6f6")

        tk.Label(code_window, text="Enter Security Code:",bg="#eaf6f6", font=("Arial", 14)).pack(pady=10)
        code_entry = tk.Entry(code_window, font=("Arial", 14), show="*")
        code_entry.pack(pady=5)

        def verify_code():
            if code_entry.get() == "1234":
                code_window.destroy()
                self.show_lecturer_panel()
            else:
                util.msg_box("Error", "Incorrect security code.")
                code_entry.delete(0, tk.END)

        tk.Button(code_window, text="Submit", font=("Arial", 12), command=verify_code).pack(pady=10)

    def show_lecturer_panel(self):
        self.main_window.withdraw()
        self.lecturer_window = tk.Toplevel(self.main_window)

        util.build_lecturer_panel(
            self.lecturer_window,
            self.register_new_user,
            self.show_attendance,
            self.back_to_main_window

        )

    def register_new_user(self):
        self.lecturer_window.withdraw()

        self.register_new_user_window = tk.Toplevel(self.lecturer_window)
        self.register_new_user_window.geometry("1100x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user
        )
        self.accept_button_register_new_user_window.place(x=750, y=320)

        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Try Again', 'red', self.try_again_register_new_user
        )
        self.try_again_button_register_new_user_window.place(x=750, y=420)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=650, height=500)
        self.add_img_to_label(self.capture_label)

        # Label and Entry for Full Name
        self.name_label = tk.Label(self.register_new_user_window, text="Full Name:", font=("Arial", 12))
        self.name_label.place(x=750, y=40)
        self.name_entry = tk.Entry(self.register_new_user_window, font=("Arial", 12), width=30)
        self.name_entry.place(x=750, y=70)

        # Label and Entry for Student ID
        self.id_label = tk.Label(self.register_new_user_window, text="Student ID:", font=("Arial", 12))
        self.id_label.place(x=750, y=110)
        self.id_entry = tk.Entry(self.register_new_user_window, font=("Arial", 12), width=30)
        self.id_entry.place(x=750, y=140)

        # Label and Entry for Wallet Address
        self.wallet_label = tk.Label(self.register_new_user_window, text="Wallet Address:", font=("Arial", 12))
        self.wallet_label.place(x=750, y=180)
        self.wallet_entry = tk.Entry(self.register_new_user_window, font=("Arial", 12), width=30)
        self.wallet_entry.place(x=750, y=210)

        #Label and Entry for verify wallet address
        self.verify_wallet_label = tk.Label(self.register_new_user_window, text="Verify WalletAddress:", font=("Arial", 12))
        self.verify_wallet_label.place(x=750,y=240)
        self.verify_wallet_entry = tk.Entry(self.register_new_user_window, font=("Arial", 12), width=30)
        self.verify_wallet_entry.place(x=750, y=270)

        #Going to main window button
        self.back_button = tk.Button(
            self.register_new_user_window,
            text='🏠Home',
            bg='#009999',
            fg='white',
            font=('Arial', 12),
            command=self.back_to_main_window
        )

        self.back_button.place(x=1000, y=10,)

        self.back_button = tk.Button(
            self.register_new_user_window,
            text='👈Back',
            bg='#009999',
            fg='white',
            font=('Arial', 12),
            command=self.back_to_lecturer_panel
        )
        self.back_button.place(x=920, y=10)

    def back_to_main_window(self):
        self.register_new_user_window.destroy()
        self.lecturer_window.destroy()
        self.main_window.deiconify()

    def return_home(self):
        self.lecturer_window.destroy()
        self.main_window.deiconify()

    def back_to_lecturer_panel(self):
        self.register_new_user_window.destroy()
        self.lecturer_window.deiconify()

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    import re  # Make sure this is at the top of your file

    def accept_register_new_user(self):
        name = self.name_entry.get().strip()
        student_id = self.id_entry.get().strip()
        wallet = self.wallet_entry.get().strip()
        verify_wallet = self.verify_wallet_entry.get().strip()

        # validation
        if not name or not student_id or not wallet or not verify_wallet:
            util.msg_box('Error', 'All fields are required!')
            return

        # Check if wallet addresses match
        if wallet != verify_wallet:
            util.msg_box('Error', 'Wallet addresses do not match. Please verify and try again.')
            return

        # Ethereum wallet format validation
        if not re.fullmatch(r"0x[a-fA-F0-9]{40}", wallet):
            util.msg_box('Error', 'Invalid Ethereum wallet address format.')
            return

        # Face encoding
        embeddings = face_recognition.face_encodings(self.register_new_user_capture)
        if len(embeddings) == 0:
            util.msg_box('Error', 'No face detected. Try again!')
            return

        embeddings = embeddings[0]

        # Check if wallet already exists


        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (student_id, name, wallet, embedding) VALUES (?, ?, ?, ?)",
                       (student_id, name, wallet, embeddings.tobytes()))
        conn.commit()
        conn.close()

        # Add to local list
        self.user_embeddings.append((student_id, embeddings))

        util.msg_box('Success!', 'User was registered successfully!')
        self.register_new_user_window.destroy()

    def show_attendance(self):
        logs = util.get_attendance_logs(self.db_path)
        if not logs:
            util.msg_box("Attendance Log", "No attendance records found.")
            return

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        today_logs = [log for log in logs if log[2].startswith(today)]  # timestamp is at index 2

        if not today_logs:
            util.msg_box("Attendance Log", "No attendance records for today.")
            return

        file_path = os.path.join(os.getcwd(), f"attendance_log_{today}.csv")

        with open(file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Student ID", "Date", "Time"])
            for student_id, date, time in today_logs:
                writer.writerow([student_id, date, time])

        util.msg_box("Export Successful", f"Today's attendance saved as:\n{file_path}")

    def show_attendance_feedback(self, message):
        self.attendance_feedback_label.config(text=message)
        self.attendance_feedback_label.after(2000, lambda: self.attendance_feedback_label.config(text=""))

    def start(self):
        self.main_window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.main_window.mainloop()

    def on_close(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        self.main_window.destroy()


if __name__ == "__main__":
    app = App()
    app.start()
