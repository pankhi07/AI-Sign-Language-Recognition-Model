import cv2
import mediapipe as mp
import numpy as np
import os

mp_holistic = mp.solutions.holistic


def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False
    ) as holistic:

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # OpenCV uses BGR, MediaPipe expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(frame_rgb)

            landmarks = []

            # -------------------------
            # Pose: 33 landmarks
            # -------------------------
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([[0, 0, 0]] * 33)

            # -------------------------
            # Left hand: 21 landmarks
            # -------------------------
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([[0, 0, 0]] * 21)

            # -------------------------
            # Right hand: 21 landmarks
            # -------------------------
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([[0, 0, 0]] * 21)

            frames.append(landmarks)

    cap.release()

    return np.array(frames, dtype=np.float32)


# -------------------------
# Main
# -------------------------

video_path = "videos/Hello/Hello.mp4"
output_path = "landmarks/Hello/Hello.npy"

data = extract_landmarks(video_path)

os.makedirs(os.path.dirname(output_path), exist_ok=True)

np.save(output_path, data)

print("Saved:", output_path)
print("Shape:", data.shape)