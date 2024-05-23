import threading
import tkinter as tk
from tkinter import messagebox

import cv2
import dlib
import imutils
from imutils import face_utils
from pygame import mixer
from scipy.spatial import distance

# Initialize Pygame mixer
mixer.init()
mixer.music.load(r"C:\Users\91752\Downloads\music.wav")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Global variable to control the detection state
is_running = False

# Drowsiness detection function
def detection_loop():
    global is_running
    thresh = 0.25
    frame_check = 20
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor(r"C:\Users\91752\Downloads\shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    cap = cv2.VideoCapture(0)
    flag = 0

    while True:
        if not is_running:
            continue

        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                flag = 0
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()

# Function to start the detection in a new thread
def start_detection():
    global is_running
    is_running = True
    toggle_button.config(text="Pause Detection")
    detection_thread = threading.Thread(target=detection_loop)
    detection_thread.daemon = True
    detection_thread.start()

# Toggle the state of the detection
def toggle_detection():
    global is_running
    is_running = not is_running
    if is_running:
        toggle_button.config(text="Pause Detection")
    else:
        toggle_button.config(text="Resume Detection")
        
# Function to exit the detection
def exit_detection():
    global is_running
    is_running = False
    root.destroy()

# Function to show info message
def show_info():
    info_text = (
        "Click 'Start Detection' to begin the drowsiness detection system.\n"
        "Click 'Pause Detection' to pause the drowsiness detection system.\n"
        "Click 'Resume Detection' to resume the drowsiness detection system.\n"
        "Click 'Exit Detection' to exit the drowsiness detection system."
    )
    messagebox.showinfo("Info", info_text)
    
def contact_us():
    import webbrowser
    webbrowser.open("https://example.com/contact")  # Replace with your contact URL

def meet_us():
    import webbrowser
    webbrowser.open("https://example.com/meet")  # Replace with your meet us URL

def leave_feedback():
    import webbrowser
    webbrowser.open("https://example.com/feedback")  # Replace with your feedback URL
    
# Create the main Tkinter window
root = tk.Tk()
root.title("Drowsiness Detection System")
root.geometry("500x400")

# Set background color
root.configure(bg='#ADD8E6')

# Add a header label
header_label = tk.Label(root, text="AwakeGuard Drowsiness Detection System", font=("Verdana", 22, "bold"), bg='#ADD8E6')
header_label.pack(pady=22)

# Create a frame for buttons
button_frame = tk.Frame(root, bg='#ADD8E6')
button_frame.pack(pady=20)

# Add a button to start the detection
start_button = tk.Button(button_frame, text="Start Detection", command=start_detection, font=("Helvetica", 14), bg='#32CD32', fg='white', padx=10, pady=5)
start_button.grid(row=0, column=0, padx=20, pady=10)

# Add a toggle button to pause/resume the detection
toggle_button = tk.Button(button_frame, text="Pause Detection", command=toggle_detection, font=("Helvetica", 14), bg='#FFD700', fg='black', padx=10, pady=5)
toggle_button.grid(row=1, column=0, padx=20, pady=10)

# Add an exit button to stop detection and exit the application
exit_button = tk.Button(button_frame, text="Exit Detection", command=exit_detection, font=("Helvetica", 14), bg='#FF6347', fg='white', padx=10, pady=5)
exit_button.grid(row=2, column=0, padx=20, pady=10)

# Add an info button
info_button = tk.Button(button_frame, text="Info", command=show_info, font=("Helvetica", 14), bg='#1E90FF', fg='white', padx=10, pady=5)
info_button.grid(row=3, column=0, padx=20, pady=10)

# Create another frame for the bottom buttons
bottom_button_frame = tk.Frame(root, bg='#ADD8E6')
bottom_button_frame.pack(side=tk.BOTTOM, pady=20)

# Add a Contact Us button
contact_button = tk.Button(bottom_button_frame, text="Contact Us", command=contact_us, font=("Helvetica", 14), bg='#FFA500', fg='white', padx=10, pady=5)
contact_button.grid(row=0, column=0, padx=10)

# Add a Meet Us button
meet_button = tk.Button(bottom_button_frame, text="Meet Our Team", command=meet_us, font=("Helvetica", 14), bg='#FF4500', fg='white', padx=10, pady=5)
meet_button.grid(row=0, column=1, padx=10)

# Add a Leave Feedback button
feedback_button = tk.Button(bottom_button_frame, text="Leave Feedback", command=leave_feedback, font=("Helvetica", 14), bg='#FF6347', fg='white', padx=10, pady=5)
feedback_button.grid(row=0, column=2, padx=10)

root.mainloop()