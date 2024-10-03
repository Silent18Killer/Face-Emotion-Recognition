import cv2
from fer import FER
import os

def detect_emotion_in_image(image_path, width=800, height=600):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image. Please check the file path.")
        return

    detector = FER()
    emotion, score = detector.top_emotion(image)
    cv2.putText(image, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Emotion Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_emotion_in_live_camera():
    detector = FER()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion, score = detector.top_emotion(frame)
        cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Select input method:")
    print("1. Live Camera")
    print("2. Image File")
    choice = input("Enter choice (1/2): ")

    if choice == '1':
        detect_emotion_in_live_camera()
    elif choice == '2':
        image_path = input("Enter the path to the image file: ").strip().replace("\\", "/")
        if os.path.exists(image_path):
            detect_emotion_in_image(image_path)
        else:
            print("Image file not found. Please check the path and try again.")
    else:
        print("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    main()
