import cv2
import mediapipe as mp
import numpy as np
import os

# ============================================================
# SETTINGS
# ============================================================

INPUT_FOLDER = "Videos"
OUTPUT_FOLDER = "landmarks"

# Supported video formats
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic


# ============================================================
# FUNCTION: EXTRACT LANDMARKS FROM ONE VIDEO
# ============================================================

def extract_landmarks(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    if not cap.isOpened():
        print("ERROR: Could not open video:", video_path)
        return None

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

            # OpenCV gives BGR.
            # MediaPipe expects RGB.
            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Run MediaPipe
            results = holistic.process(frame_rgb)

            landmarks = []

            # ==================================================
            # POSE
            # 33 LANDMARKS
            # ==================================================

            if results.pose_landmarks:

                for lm in results.pose_landmarks.landmark:

                    landmarks.append([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

            else:

                landmarks.extend(
                    [[0.0, 0.0, 0.0]] * 33
                )

            # ==================================================
            # LEFT HAND
            # 21 LANDMARKS
            # ==================================================

            if results.left_hand_landmarks:

                for lm in results.left_hand_landmarks.landmark:

                    landmarks.append([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

            else:

                landmarks.extend(
                    [[0.0, 0.0, 0.0]] * 21
                )

            # ==================================================
            # RIGHT HAND
            # 21 LANDMARKS
            # ==================================================

            if results.right_hand_landmarks:

                for lm in results.right_hand_landmarks.landmark:

                    landmarks.append([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

            else:

                landmarks.extend(
                    [[0.0, 0.0, 0.0]] * 21
                )

            # Add this frame
            frames.append(landmarks)

    cap.release()

    # Convert list → NumPy array
    data = np.array(
        frames,
        dtype=np.float32
    )

    return data


# ============================================================
# MAIN PROGRAM
# ============================================================

print("=" * 60)
print("ISL VIDEO → LANDMARK CONVERTER")
print("=" * 60)

total_videos = 0
successful_videos = 0
failed_videos = 0

# Walk through all folders inside Videos/
for root, dirs, files in os.walk(INPUT_FOLDER):

    for filename in files:

        # Ignore files that aren't videos
        if not filename.lower().endswith(VIDEO_EXTENSIONS):
            continue

        total_videos += 1

        # Full input path
        video_path = os.path.join(
            root,
            filename
        )

        # Find sign name.
        # Example:
        #
        # Videos/Hello/video1.mp4
        #
        # sign = Hello

        relative_path = os.path.relpath(
            video_path,
            INPUT_FOLDER
        )

        parts = relative_path.split(os.sep)

        if len(parts) < 2:

            print(
                "WARNING: Video is not inside a sign folder:",
                video_path
            )

            failed_videos += 1
            continue

        sign_name = parts[0]

        # Create output folder
        sign_output_folder = os.path.join(
            OUTPUT_FOLDER,
            sign_name
        )

        os.makedirs(
            sign_output_folder,
            exist_ok=True
        )

        # Change extension from .mp4 → .npy
        filename_without_extension = os.path.splitext(
            filename
        )[0]

        output_path = os.path.join(
            sign_output_folder,
            filename_without_extension + ".npy"
        )

        print()
        print("-" * 60)
        print("Processing:", video_path)
        print("Sign:", sign_name)

        # Extract landmarks
        data = extract_landmarks(video_path)

        # Check result
        if data is None or len(data) == 0:

            print("FAILED:", video_path)

            failed_videos += 1
            continue

        # Save .npy
        np.save(
            output_path,
            data
        )

        print("Saved:", output_path)
        print("Shape:", data.shape)

        successful_videos += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)

print("Total videos:", total_videos)
print("Successful:", successful_videos)
print("Failed:", failed_videos)

print()
print("Output folder:")
print(OUTPUT_FOLDER)

print("=" * 60)