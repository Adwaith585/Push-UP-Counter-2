import cv2
import mediapipe as md
import numpy as np
import winsound  # For beep sound on Windows (use 'beep' library on Linux/macOS)

# ========== Initialize MediaPipe ==========
md_drawing = md.solutions.drawing_utils
md_pose = md.solutions.pose

# ========== Camera Initialization ==========
cap = cv2.VideoCapture(0)

# ========== Variables ==========
count = 0
position = None

# ========== Start Pose Detection ==========
with md_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Empty camera")
            break

        # Flip horizontally and convert to RGB
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image to detect pose landmarks
        result = pose.process(image_rgb)

        imlist = []

        if result.pose_landmarks:
            # Draw landmarks with custom red dots and green lines
            md_drawing.draw_landmarks(
                image,
                result.pose_landmarks,
                md_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=md_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=5),  # Red dots
                connection_drawing_spec=md_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # Green lines
            )

            # Extract and store coordinates
            for id, lm in enumerate(result.pose_landmarks.landmark):
                h, w, _ = image.shape
                X, Y = int(lm.x * w), int(lm.y * h)
                imlist.append([id, X, Y])

            # ========== Push-up Detection Logic ==========
            if len(imlist) != 0:
                # Going Down: Shoulders below elbows
                if (imlist[12][2] and imlist[11][2] >= imlist[14][2] and imlist[13][2]):
                    position = "down"

                # Going Up: Shoulders above elbows → count push-up
                if (imlist[12][2] and imlist[11][2] <= imlist[14][2] and imlist[13][2]) and position == "down":
                    position = "up"
                    count += 1
                    print("Push-ups:", count)

                    # Beep sound on each count (frequency, duration)
                    winsound.Beep(1000, 150)  # 1000 Hz, 150 ms

        # ========== Show Main Camera Feed ==========
        cv2.imshow("Push-UP Counter", image)

        # ========== Counter Display Window ==========
        counter_display = np.zeros((200, 300, 3), dtype=np.uint8)  # Black background
        cv2.putText(counter_display, f'{count}', (90, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
        cv2.imshow("Counter", counter_display)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ========== Cleanup ==========
cap.release()
cv2.destroyAllWindows()
