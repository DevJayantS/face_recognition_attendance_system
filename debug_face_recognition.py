#!/usr/bin/env python3
"""
Debug script for face recognition accuracy testing
This helps you test recognition accuracy and adjust settings
"""

import cv2
import face_recognition
import numpy as np
import os
from collections import defaultdict

def load_known_faces_debug():
    """Load faces with detailed debugging info"""
    dataset_path = "dataset"
    known_encodings = []
    known_names = []
    student_encodings = {}
    
    print("🔍 Loading face encodings with debug info...")
    
    for student_name in os.listdir(dataset_path):
        student_folder = os.path.join(dataset_path, student_name)
        
        if not os.path.isdir(student_folder):
            continue
        
        student_encodings[student_name] = []
        print(f"\n📁 Processing {student_name}:")
        
        for file in os.listdir(student_folder):
            if file.endswith((".jpg", ".png")):
                img_path = os.path.join(student_folder, file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  ❌ Could not load {file}")
                    continue
                
                # Resize large images
                height, width = img.shape[:2]
                if max(height, width) > 1280:
                    scale = 1280 / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = cv2.resize(img, (new_width, new_height))
                
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Try face detection
                face_locations = face_recognition.face_locations(rgb_img, model="hog")
                if len(face_locations) == 0:
                    face_locations = face_recognition.face_locations(rgb_img, model="hog", number_of_times_to_upsample=1)
                
                if len(face_locations) > 0:
                    if len(face_locations) > 1:
                        def face_area(face_location):
                            top, right, bottom, left = face_location
                            return (bottom - top) * (right - left)
                        face_locations = [max(face_locations, key=face_area)]
                        print(f"  ⚠️  Multiple faces in {file}, using largest")
                    
                    encodings = face_recognition.face_encodings(rgb_img, face_locations)
                    if len(encodings) > 0:
                        student_encodings[student_name].append(encodings[0])
                        print(f"  ✅ {file}: Face detected and encoded")
                    else:
                        print(f"  ❌ {file}: Face detected but encoding failed")
                else:
                    print(f"  ❌ {file}: No face detected")
    
    # Add all encodings
    for student_name, encodings in student_encodings.items():
        if encodings:
            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(student_name)
            print(f"📊 {student_name}: {len(encodings)} encodings")
        else:
            print(f"⚠️  {student_name}: No valid encodings!")
    
    print(f"\n✅ Total: {len(set(known_names))} students, {len(known_encodings)} encodings")
    return known_encodings, known_names

def test_recognition_accuracy():
    """Test recognition with different confidence thresholds"""
    known_encodings, known_names = load_known_faces_debug()
    
    if len(known_encodings) == 0:
        print("❌ No encodings loaded. Check your dataset folder.")
        return
    
    print("\n🎯 Testing recognition accuracy...")
    print("Press 'q' to quit, 's' to save current frame")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Test different confidence thresholds
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            
            # Calculate confidence
            confidence = max(0, 1 - (best_distance / 0.6))
            name = known_names[best_match_index] if confidence >= 0.6 else "Unknown"
            
            # Draw results
            top, right, bottom, left = [v * 4 for v in face_location]
            
            # Color based on confidence
            if confidence >= 0.8:
                color = (0, 255, 0)  # Green
            elif confidence >= 0.6:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show distance info
            cv2.putText(frame, f"Distance: {best_distance:.3f}", (left, bottom + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Face Recognition Debug", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame for analysis
            filename = f"debug_frame_{len(os.listdir('.'))}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Saved debug frame: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()

def analyze_dataset_quality():
    """Analyze the quality of your dataset"""
    print("\n📊 Analyzing dataset quality...")
    
    dataset_path = "dataset"
    quality_report = {}
    
    for student_name in os.listdir(dataset_path):
        student_folder = os.path.join(dataset_path, student_name)
        if not os.path.isdir(student_folder):
            continue
        
        images = [f for f in os.listdir(student_folder) if f.endswith(('.jpg', '.png'))]
        quality_report[student_name] = {
            'total_images': len(images),
            'valid_faces': 0,
            'issues': []
        }
        
        for img_file in images:
            img_path = os.path.join(student_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                quality_report[student_name]['issues'].append(f"Could not load {img_file}")
                continue
            
            # Check image size
            height, width = img.shape[:2]
            if max(height, width) < 100:
                quality_report[student_name]['issues'].append(f"{img_file}: Too small ({width}x{height})")
            elif max(height, width) > 2000:
                quality_report[student_name]['issues'].append(f"{img_file}: Very large ({width}x{height})")
            
            # Check for faces
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img, model="hog")
            if len(face_locations) == 0:
                face_locations = face_recognition.face_locations(rgb_img, model="hog", number_of_times_to_upsample=1)
            
            if len(face_locations) > 0:
                quality_report[student_name]['valid_faces'] += 1
                if len(face_locations) > 1:
                    quality_report[student_name]['issues'].append(f"{img_file}: Multiple faces detected")
            else:
                quality_report[student_name]['issues'].append(f"{img_file}: No face detected")
    
    # Print report
    print("\n" + "="*60)
    print("DATASET QUALITY REPORT")
    print("="*60)
    
    for student, report in quality_report.items():
        print(f"\n👤 {student}:")
        print(f"   Images: {report['total_images']}")
        print(f"   Valid faces: {report['valid_faces']}")
        
        if report['issues']:
            print("   ⚠️  Issues:")
            for issue in report['issues']:
                print(f"      - {issue}")
        else:
            print("   ✅ No issues found")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("🔧 Face Recognition Debug Tool")
    print("="*40)
    
    while True:
        print("\nChoose an option:")
        print("1. Analyze dataset quality")
        print("2. Test recognition accuracy")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            analyze_dataset_quality()
        elif choice == "2":
            test_recognition_accuracy()
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

