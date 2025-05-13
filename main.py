import os
import sqlite3
import datetime
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import util
import csv


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x600+350+100")
        self.main_window.title("Facial Recognition Attendance System")

        self.db_path = 'face_data.db'
        self.initialize_db()

        # Open Lecturer Panel Button
        self.lecturer_panel_button = util.get_button(
            self.main_window, 'Lecturer Panel', 'blue', self.open_lecturer_window
        )
        self.lecturer_panel_button.place(x=750, y=250)

        # Webcam display
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)
        self.add_webcam(self.webcam_label)

    def initialize_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                student_id TEXT PRIMARY KEY,
                embedding BLOB
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                action TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(student_id) REFERENCES users(student_id)
            )
        """)
        conn.commit()
        conn.close()

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)
        self._label.after(20, self.process_webcam)

    def open_lecturer_window(self):
        # Security Code Prompt
        code_window = tk.Toplevel(self.main_window)
        code_window.title("Lecturer Login")
        code_window.geometry("400x200")

        tk.Label(code_window, text="Enter Security Code:", font=("Arial", 14)).pack(pady=10)
        code_entry = tk.Entry(code_window, font=("Arial", 14), show="*")
        code_entry.pack(pady=5)

        def verify_code():
            entered_code = code_entry.get()
            if entered_code == "1234":
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
        today_logs = [log for log in logs if log[3].startswith(today)]

        if not today_logs:
            util.msg_box("Attendance Log", "No attendance records for today.")
            return

        file_path = os.path.join(os.getcwd(), f"attendance_log_{today}.csv")

        with open(file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Student ID", "Unit", "Action", "Timestamp"])
            for student_id, unit, action, timestamp in today_logs:
                writer.writerow([student_id, unit, action, timestamp])

        util.msg_box("Export Successful", f"Today's attendance saved as:\n{file_path}")

    def start(self):
        self.main_window.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
