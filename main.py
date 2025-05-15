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


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x600+350+100")
        self.main_window.title("Facial Recognition Attendance System")
        self.attendance_feedback_label = tk.Label(
            self.main_window,
            text="",
            font=("Arial", 20, "bold"),
            fg="green",
            bg="white"
        )
        self.attendance_feedback_label.place(x=400, y=520)

        self.db_path = 'face_data.db'
        self.initialize_db()

        # ✅ Initialize Mediapipe hands detector once
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

        # ✅ Cache embeddings from DB
        self.user_embeddings = self.load_user_embeddings()

        self.lecturer_panel_button = util.get_button(
            self.main_window, 'Lecturer Panel', 'blue', self.open_lecturer_window
        )
        self.lecturer_panel_button.place(x=750, y=250)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.cap = cv2.VideoCapture(0)

        # ✅ Start background thread
        self.running = True
        self.thread = threading.Thread(target=self.process_webcam)
        self.thread.daemon = True
        self.thread.start()
        self.last_face_check_time = 0
        self.face_check_interval = 3  # seconds
        self.mp_drawing = mp.solutions.drawing_utils  # For visualizing hand landmarks
        self.recently_marked = {}  # student_id: last_mark_time
        self.mark_cooldown = 60  # seconds
        self.last_seen_encoding = None
        self.recently_marked = {}  # student_id: timestamp
        self.mark_cooldown = 60

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

    def initialize_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS users (
            student_id TEXT PRIMARY KEY,
            embedding BLOB
        )""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            action TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES users(student_id)
        )""")
        conn.commit()
        conn.close()

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

        # Avoid duplicate marking within cooldown
        last_mark = self.recently_marked.get(student_id)
        if last_mark and (now - last_mark).total_seconds() < self.mark_cooldown:
            print(f"⏳ {student_id} recently marked. Skipping.")
            self.show_attendance_feedback(f"⏳ Already marked recently")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO attendance (student_id, action) VALUES (?, 'Check-in')
        """, (student_id,))
        conn.commit()
        conn.close()

        self.recently_marked[student_id] = now
        print(f"✅ Attendance marked for {student_id}")
        self.show_attendance_feedback(f"✔ Marked: {student_id}")
        self.animate_success()

    def animate_success(self):
        emoji = "😄"
        font_size = 50
        label = tk.Label(
            self.main_window,
            text=emoji,
            font=("Arial", font_size, "bold"),
            fg="green",
            bg="white"
        )
        label.place(x=650, y=420)

        self.play_success_sound()

        # Animation: fade in, bounce, then fade out
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
                label.place(x=650, y=420 - offset)
                self.main_window.after(interval, lambda: animate(step + 1))
            elif step <= 2 * steps:
                # Fade out
                fade_step = step - steps
                gray = int(255 * (fade_step / steps))
                label.config(fg=f"#{gray:02x}{gray:02x}{gray:02x}")
                self.main_window.after(interval, lambda: animate(step + 1))
            else:
                label.destroy()

        animate()

    def open_lecturer_window(self):
        code_window = tk.Toplevel(self.main_window)
        code_window.title("Lecturer Login")
        code_window.geometry("400x200")

        tk.Label(code_window, text="Enter Security Code:", font=("Arial", 14)).pack(pady=10)
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
        self.lecturer_window = tk.Toplevel(self.main_window)
        self.lecturer_window.title("Lecturer Panel")
        self.lecturer_window.geometry("1200x600+350+100")

        register_btn = util.get_button(self.lecturer_window, "Register New User", "gray", self.register_new_user, fg='black')
        register_btn.place(x=800, y=150)

        show_attendance_btn = util.get_button(self.lecturer_window, "Show Attendance", "green", self.show_attendance)
        show_attendance_btn.place(x=800, y=250)

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.lecturer_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user
        )
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Try Again', 'red', self.try_again_register_new_user
        )
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)
        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(
            self.register_new_user_window, 'Please, enter Student ID:'
        )
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def accept_register_new_user(self):
        student_id = self.entry_text_register_new_user.get(1.0, "end-1c").strip()
        if not student_id:
            util.msg_box('Error', 'Student ID cannot be empty!')
            return

        embeddings = face_recognition.face_encodings(self.register_new_user_capture)
        if len(embeddings) == 0:
            util.msg_box('Error', 'No face detected. Try again!')
            return

        embeddings = embeddings[0]
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (student_id, embedding) VALUES (?, ?)", (student_id, embeddings.tobytes()))
        conn.commit()
        conn.close()

        # ✅ Update cached embeddings
        self.user_embeddings.append((student_id, embeddings))

        util.msg_box('Success!', 'User was registered successfully!')
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        if hasattr(self, "most_recent_capture_pil"):
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            label.imgtk = imgtk
            label.configure(image=imgtk)
            self.register_new_user_capture = self.most_recent_capture_arr.copy()
        else:
            util.msg_box("Error", "No image captured yet. Please try again.")

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
            writer.writerow(["Student ID", "Action", "Timestamp"])
            for student_id, action, timestamp in today_logs:
                writer.writerow([student_id, action, timestamp])

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
