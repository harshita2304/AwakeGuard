import cv2
import numpy as np
import mediapipe as mp
import time
from math import dist
import pygame


class DrowsinessDetector:
    def __init__(self):
        # 🔊 Initialize sound
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound("beep_1s.wav")
        self.is_playing = False

        # MediaPipe setup
        self.base_options = mp.tasks.BaseOptions(
            model_asset_path="face_landmarker.task"
        )

        self.options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1
        )

        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(
            self.options
        )

        # EAR config
        self.EAR_THRESHOLD = 0.25
        self.FRAME_CHECK = 20
        self.counter = 0

        # Eye landmarks
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def eye_aspect_ratio(self, eye):
        A = dist(eye[1], eye[5])
        B = dist(eye[2], eye[4])
        C = dist(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def process_frame(self, frame):
        h, w = frame.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        timestamp = int(time.time() * 1000)
        result = self.detector.detect_for_video(mp_image, timestamp)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            def get_eye(indices):
                return np.array([
                    (landmarks[i].x * w, landmarks[i].y * h)
                    for i in indices
                ])

            left_eye = get_eye(self.LEFT_EYE)
            right_eye = get_eye(self.RIGHT_EYE)

            leftEAR = self.eye_aspect_ratio(left_eye)
            rightEAR = self.eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eyes
            for (x, y) in np.concatenate((left_eye, right_eye)):
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            # 🚨 Drowsiness logic + sound
            if ear < self.EAR_THRESHOLD:
                self.counter += 1

                if self.counter >= self.FRAME_CHECK:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 3)

                    # 🔊 Play sound (only once until reset)
                    if not self.is_playing:
                        self.alert_sound.play(-1)  # loop
                        self.is_playing = True
            else:
                self.counter = 0
                # 🔇 Stop sound when eyes open
                if self.is_playing:
                    self.alert_sound.stop()
                    self.is_playing = False

        return frame