import tkinter as tk
from PIL import Image, ImageTk
import cv2
from detection import DrowsinessDetector


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AwakeGuard")
        self.root.geometry("900x600")
        self.root.configure(bg="#0f172a")

        self.detector = DrowsinessDetector()
        self.cap = None
        self.running = False

        # Title
        title = tk.Label(root,
                         text="AwakeGuard - Driver Monitoring",
                         font=("Segoe UI", 22, "bold"),
                         fg="white",
                         bg="#0f172a")
        title.pack(pady=20)

        # Video Frame
        self.video_label = tk.Label(root, bg="#0f172a")
        self.video_label.pack()

        # Buttons Frame
        btn_frame = tk.Frame(root, bg="#0f172a")
        btn_frame.pack(pady=20)

        # Start Button
        self.start_btn = tk.Button(
            btn_frame,
            text="Start",
            font=("Segoe UI", 14),
            bg="#22c55e",
            fg="white",
            width=12,
            command=self.start
        )
        self.start_btn.grid(row=0, column=0, padx=10)

        # Stop Button
        self.stop_btn = tk.Button(
            btn_frame,
            text="Stop",
            font=("Segoe UI", 14),
            bg="#ef4444",
            fg="white",
            width=12,
            command=self.stop
        )
        self.stop_btn.grid(row=0, column=1, padx=10)

    def start(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.update_frame()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def update_frame(self):
        if self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)

                # Process with detector
                frame = self.detector.process_frame(frame)

                # Convert for Tkinter
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.video_label.after(10, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()