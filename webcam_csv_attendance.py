import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime

# --- Load encodings from dataset ---
dataset_path = "dataset"

known_encodings = []
known_names = []

for student_name in os.listdir(dataset_path):
    student_folder = os.path.join(dataset_path, student_name)
    
    if not os.path.isdir(student_folder):
        continue
    
    for file in os.listdir(student_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(student_folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_img)
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(student_name)

print(f"✅ Loaded encodings for {len(set(known_names))} students")

# --- Attendance CSV setup ---
today_date = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"attendance_{today_date}.csv"

if not os.path.exists(csv_filename):
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])  # header row

# --- Webcam recognition loop ---
video_capture = cv2.VideoCapture(0)
attendance_marked = set()  # keep track to avoid duplicate marking

print("📸 Starting camera... Press 'q' to quit")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Check for quit key FIRST, before processing faces
    key = cv2.waitKey(1)
    if key == ord("q") or key == 27:   # 27 = ESC key
        break
    
    # Resize frame for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_names[best_match_index]

            # Mark attendance if not already done
            if name not in attendance_marked:
                with open(csv_filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, datetime.now().strftime("%H:%M:%S")])
                attendance_marked.add(name)
                print(f"✅ Attendance marked for {name}")

        # Draw box around face
        top, right, bottom, left = [v * 4 for v in face_location]  # scale back
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Face Recognition Attendance", frame)

video_capture.release()
cv2.destroyAllWindows()
