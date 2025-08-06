import cv2
import mediapipe as md
import numpy as np

# Initialize MediaPipe Pose
md_drawing = md.solutions.drawing_utils
md_pose = md.solutions.pose

# Initialize camera
cap = cv2.VideoCapture(0)

# Variables
count = 0
position = None

# Start Pose detection
with md_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Empty camera")
            break

        # Flip and convert color
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        imlist = []

        if result.pose_landmarks:
            md_drawing.draw_landmarks(
                image, result.pose_landmarks, md_pose.POSE_CONNECTIONS
            )

            for id, lm in enumerate(result.pose_landmarks.landmark):
                h, w, _ = image.shape
                X, Y = int(lm.x * w), int(lm.y * h)
                imlist.append([id, X, Y])

            # Push-up detection logic
            if len(imlist) != 0:
                if (imlist[12][2] and imlist[11][2] >= imlist[14][2] and imlist[13][2]):
                    position = "down"
                if (imlist[12][2] and imlist[11][2] <= imlist[14][2] and imlist[13][2]) and position == "down":
                    position = "up"
                    count += 1
                    print("Push-ups:", count)

        # Show main camera window
        cv2.imshow("Push-UP Counter", image)

        # Create counter window (separate)
        counter_display = np.zeros((200, 300, 3), dtype=np.uint8)  # Black background
        cv2.putText(counter_display, f'{count}', (90, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
        cv2.imshow("Counter", counter_display)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
