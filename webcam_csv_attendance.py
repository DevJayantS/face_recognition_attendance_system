import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime

# --- Load encodings from dataset with improved accuracy ---
dataset_path = "dataset"

known_encodings = []
known_names = []
student_encodings = {}  # Store multiple encodings per student

for student_name in os.listdir(dataset_path):
    student_folder = os.path.join(dataset_path, student_name)
    
    if not os.path.isdir(student_folder):
        continue
    
    student_encodings[student_name] = []
    
    # Process all images for each student
    for file in os.listdir(student_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(student_folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Resize large images for better processing
            height, width = img.shape[:2]
            if max(height, width) > 1280:
                scale = 1280 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Try multiple face detection strategies
            face_locations = face_recognition.face_locations(rgb_img, model="hog")
            if len(face_locations) == 0:
                # Try with upsampling for small faces
                face_locations = face_recognition.face_locations(rgb_img, model="hog", number_of_times_to_upsample=1)
            
            if len(face_locations) > 0:
                # If multiple faces, use the largest one
                if len(face_locations) > 1:
                    def face_area(face_location):
                        top, right, bottom, left = face_location
                        return (bottom - top) * (right - left)
                    face_locations = [max(face_locations, key=face_area)]
                
                encodings = face_recognition.face_encodings(rgb_img, face_locations)
                if len(encodings) > 0:
                    student_encodings[student_name].append(encodings[0])

# Add all encodings to the main lists
for student_name, encodings in student_encodings.items():
    if encodings:  # Only add if we have at least one encoding
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(student_name)

print(f"✅ Loaded {len(set(known_names))} students with {len(known_encodings)} total encodings")

# Improved face recognition with confidence scoring
def recognize_face_with_confidence(face_encoding, known_encodings, known_names, confidence_threshold=0.6):
    """
    Recognize a face with confidence scoring
    Returns (name, confidence) or (None, 0) if below threshold
    """
    if len(known_encodings) == 0:
        return None, 0
    
    # Calculate face distances
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
    
    # Find the best match
    best_match_index = np.argmin(face_distances)
    best_distance = face_distances[best_match_index]
    
    # Convert distance to confidence (0-1 scale, higher is better)
    confidence = max(0, 1 - (best_distance / 0.6))
    
    if confidence >= confidence_threshold:
        return known_names[best_match_index], confidence
    else:
        return None, confidence

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
        # Use improved recognition with confidence scoring
        name, confidence = recognize_face_with_confidence(face_encoding, known_encodings, known_names, confidence_threshold=0.6)
        
        if name is None:
            name = "Unknown"
            confidence = 0

        # Mark attendance if not already done and confidence is high enough
        if name != "Unknown" and name not in attendance_marked:
            with open(csv_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, datetime.now().strftime("%H:%M:%S")])
            attendance_marked.add(name)
            print(f"✅ Attendance marked for {name} (confidence: {confidence:.2f})")

        # Draw box around face with color based on confidence
        top, right, bottom, left = [v * 4 for v in face_location]  # scale back
        
        # Color coding: Green for high confidence, Yellow for medium, Red for low/unknown
        if confidence >= 0.8:
            color = (0, 255, 0)  # Green
        elif confidence >= 0.6:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Display name and confidence
        display_text = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
        cv2.putText(frame, display_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Face Recognition Attendance", frame)

video_capture.release()
cv2.destroyAllWindows()
