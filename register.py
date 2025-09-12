import cv2
import os

# Folder to save student images
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

student_name = input("Enter student name: ")
student_folder = os.path.join(dataset_path, student_name)
if not os.path.exists(student_folder):
    os.makedirs(student_folder)

# Open webcam
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Register Student", frame)

    # Save every 10 frames
    if count % 10 == 0:
        img_path = os.path.join(student_folder, f"{student_name}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")

    count += 1

    if cv2.waitKey(1) & 0xFF == ord("q") or count > 50:
        break

cap.release()
cv2.destroyAllWindows()
