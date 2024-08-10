import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

import random
import pygame
import time  # Import the time module

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Emotion thresholds based on landmark distances (for other emotions)
emotion_thresholds = {
    'happy': (0.45, 0.3),  # Example values; adjust based on testing
    'surprised': (0.15, 0.25),
    'angry': (0.05, 0.1),
}

# Initialize pygame mixer
pygame.mixer.init()

# Function to play music based on the dominant emotion
def play_music(emotion):
    try:
        music_folder = f'my_music/{emotion}/'
        songs = os.listdir(music_folder)
        song_to_play = random.choice(songs)
        pygame.mixer.music.load(os.path.join(music_folder, song_to_play))
        pygame.mixer.music.play()
        print(f'Playing {song_to_play} for emotion: {emotion}')
    except Exception as e:
        print(f"Error playing music: {e}")

def calculate_distances(landmarks):
    # Get landmark points for mouth corners and lips
    left_mouth = landmarks[61]  # Landmark for left corner of the mouth
    right_mouth = landmarks[291]  # Landmark for right corner of the mouth
    upper_lip = landmarks[13]  # Landmark for the center of the upper lip
    lower_lip = landmarks[14]  # Landmark for the center of the lower lip
    left_eye = landmarks[33]  # Landmark for left eye
    right_eye = landmarks[133]  # Landmark for right eye
    left_brow_inner = landmarks[70]  # Landmark for inner left eyebrow
    right_brow_inner = landmarks[300]  # Landmark for inner right eyebrow
    left_brow_outer = landmarks[105]  # Landmark for outer left eyebrow
    right_brow_outer = landmarks[334]  # Landmark for outer right eyebrow

    # Calculate distances
    mouth_distance = np.linalg.norm(np.array(left_mouth) - np.array(right_mouth))  # Mouth width
    mouth_height = np.linalg.norm(np.array(upper_lip) - np.array(lower_lip))  # Mouth height
    eyebrow_inner_distance = np.abs(landmarks[70][1] - landmarks[300][1])  # Vertical distance between inner eyebrows
    eyebrow_outer_distance = np.abs(landmarks[105][1] - landmarks[334][1])  # Vertical distance between outer eyebrows
    eye_openness_left = np.abs(landmarks[159][1] - landmarks[145][1])  # Distance between upper and lower eyelids of the left eye
    eye_openness_right = np.abs(landmarks[386][1] - landmarks[374][1])  # Distance between upper and lower eyelids of the right eye

    return (mouth_distance, mouth_height, eyebrow_inner_distance,
            eyebrow_outer_distance, eye_openness_left, eye_openness_right, left_brow_inner, right_brow_inner)

def detect_emotion(mouth_distance, mouth_height, eyebrow_inner_distance, eyebrow_outer_distance, eye_openness_left, eye_openness_right, left_brow_inner, right_brow_inner):
    # Sadness detection based on combined features
    mouth_threshold = 0.1  # Adjust based on observed sad mouth shape (close, slightly downturned)
    brow_inner_threshold = 0.05  # Small difference between inner eyebrow heights
    brow_outer_threshold = 0.07  # Small difference between outer eyebrow heights
    eye_openness_threshold = 0.15  # Slightly narrowed eyes

    # Detecting sadness based on combined features
    if (mouth_height < mouth_threshold and
        eyebrow_inner_distance > brow_inner_threshold and
        eyebrow_outer_distance < brow_outer_threshold and
        eye_openness_left < eye_openness_threshold and
        eye_openness_right < eye_openness_threshold):
        return 'sad'

    # Other emotions detection logic (example for happy, surprised, angry)
    elif mouth_height > emotion_thresholds['surprised'][0]:  # Check mouth height for surprised
        return 'surprised'
    elif mouth_distance > emotion_thresholds['happy'][0]:  # Adjust happy logic to still check height if needed
        return 'happy'
    elif left_brow_inner[1] > 0.6 and right_brow_inner[1] > 0.6:  # Adjust based on observation for angry
        return 'angry'

    return 'neutral'

def main():
    try:
        cap = cv2.VideoCapture(0)

        # Emotion counter
        emotion_counter = {emotion: 0 for emotion in emotion_thresholds.keys()}
        emotion_counter['sad'] = 0  # Add sadness to the counter
        emotion_threshold_count = 40  # Threshold for playing music
        last_played_emotion = None  # Store the last emotion that played music

        # Initialize face detection
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to capture frame from camera.")
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(frame_rgb)

                    if results.detections:
                        for detection in results.detections:
                            try:
                                bboxC = detection.location_data.relative_bounding_box
                                h, w, _ = frame.shape

                                # Get bounding box coordinates
                                x_min = int(bboxC.xmin * w)
                                y_min = int(bboxC.ymin * h)
                                x_max = int((bboxC.xmin + bboxC.width) * w)
                                y_max = int((bboxC.ymin + bboxC.height) * h)

                                # Crop the detected face
                                face_crop = frame[y_min:y_max, x_min:x_max]
                                face_crop_resized = cv2.resize(face_crop, (200, 200))  # Normalize size

                                # Process resized face with MediaPipe Face Mesh
                                face_mesh_results = face_mesh.process(cv2.cvtColor(face_crop_resized, cv2.COLOR_BGR2RGB))

                                if face_mesh_results.multi_face_landmarks:
                                    face_landmarks = face_mesh_results.multi_face_landmarks[0]
                                    landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]

                                    (mouth_distance, mouth_height, eyebrow_inner_distance, eyebrow_outer_distance,
                                     eye_openness_left, eye_openness_right, left_brow_inner, right_brow_inner) = calculate_distances(landmarks)


                                    print(f"mouth_distance: {mouth_distance}")
                                    print(f"mouth_height: {mouth_height}")
                                    print(f"eyebrow_inner_distance: {eyebrow_inner_distance}")
                                    print(f"eyebrow_outer_distance: {eyebrow_outer_distance}")
                                    print(f"eye_openness_left: {eye_openness_left}")
                                    print(f"eye_openness_right: {eye_openness_right}")

                                    emotion = detect_emotion(mouth_distance, mouth_height, eyebrow_inner_distance,
                                                             eyebrow_outer_distance, eye_openness_left, eye_openness_right, left_brow_inner, right_brow_inner)

                                    # Update emotion counter
                                    if emotion in emotion_counter:
                                        emotion_counter[emotion] += 1
                                        print(f"Emotion {emotion} counter: {emotion_counter[emotion]}")  # Debugging output

                                        # Check if any emotion has crossed the threshold
                                        if emotion_counter[emotion] > emotion_threshold_count and emotion != last_played_emotion:
                                            play_music(emotion)
                                            last_played_emotion = emotion  # Update the last played emotion

                                            # Reset all emotion counters
                                            for key in emotion_counter.keys():
                                                emotion_counter[key] = 0

                                    # Draw landmarks as green dots on the original frame
                                    for landmark in face_landmarks.landmark:
                                        cx, cy = int(landmark.x * (x_max - x_min) + x_min), int(landmark.y * (y_max - y_min) + y_min)
                                        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)  # Green color

                                    # Display emotion on the frame
                                    cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            except Exception as e:
                                print(f"An error occurred while processing detection: {e}")

                    cv2.imshow('Emotion Recognition', frame)
                    if cv2.waitKey(5) & 0xFF == 27:  # Esc key to exit
                        break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
