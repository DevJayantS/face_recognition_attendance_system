import face_recognition
import os
import pickle
from collections import defaultdict

dataset_path = "dataset"
encodings = []
names = []
per_student_counts = defaultdict(int)

# Loop through all student folders
valid_exts = {".jpg", ".jpeg", ".png"}
for student in os.listdir(dataset_path):
    student_folder = os.path.join(dataset_path, student)

    if not os.path.isdir(student_folder):
        continue

    for file in os.listdir(student_folder):
        ext = os.path.splitext(file)[1].lower()
        if ext not in valid_exts:
            continue

        img_path = os.path.join(student_folder, file)

        try:
            # Load image and detect face locations with mild upsampling for small faces
            image = face_recognition.load_image_file(img_path)
            face_locations = face_recognition.face_locations(
                image,
                number_of_times_to_upsample=1,
                model="hog"
            )

            if len(face_locations) == 0:
                print(f"❌ No face found in {student}/{file}, skipping...")
                continue

            # If multiple faces, pick the largest face (likely the subject)
            if len(face_locations) > 1:
                def box_area(box):
                    top, right, bottom, left = box
                    return max(0, bottom - top) * max(0, right - left)
                face_locations.sort(key=box_area, reverse=True)
                face_locations = [face_locations[0]]

            face_encs = face_recognition.face_encodings(image, face_locations)
            if not face_encs:
                print(f"❌ Could not compute encoding in {student}/{file}, skipping...")
                continue

            encoding = face_encs[0]
            encodings.append(encoding)
            names.append(student)
            per_student_counts[student] += 1

        except Exception as e:
            print(f"⚠️  Error processing {student}/{file}: {e}")

# Save encodings to file
data = {"encodings": encodings, "names": names}
with open("encodings.pkl", "wb") as f:
    pickle.dump(data, f)

# Summary
unique_students = len(set(names))
total_encodings = len(encodings)
print("✅ Encodings generated & saved to encodings.pkl")
print(f"✅ Total students encoded: {unique_students}")
print(f"✅ Total encodings saved: {total_encodings}")
for student, cnt in sorted(per_student_counts.items()):
    print(f"   - {student}: {cnt} encodings")
