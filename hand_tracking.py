import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import time
import sys
import os

from gesture_controller import GestureController

# Топология соединений для отрисовки скелета
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# Индексы landmarks для вычисления центра ладони
PALM_BASE_POINTS = [0, 5, 9, 13, 17]


class HandTracker:
    def __init__(self, model_path):
        self.current_landmarks = None
        self.lock = threading.Lock()
        
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found.")
            print("Please download hand_landmarker.task from MediaPipe repository.")
            sys.exit(1)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.5,
            result_callback=self._on_results
        )
        
        try:
            self.detector = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Failed to initialize detector: {e}")
            sys.exit(1)

    def _on_results(self, result, output_image, timestamp_ms):
        with self.lock:
            if result.hand_landmarks:
                self.current_landmarks = result.hand_landmarks
            else:
                self.current_landmarks = None

    def process_frame(self, frame, gesture_controller):
        h, w, _ = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        
        self.detector.detect_async(mp_image, timestamp_ms)
        self._draw_and_process(frame, h, w, gesture_controller)
        
        return frame

    def _get_palm_center(self, landmarks, w, h):
        x_sum, y_sum = 0.0, 0.0
        count = 0
        
        for idx in PALM_BASE_POINTS:
            if idx < len(landmarks):
                x_sum += landmarks[idx].x
                y_sum += landmarks[idx].y
                count += 1
        
        if count == 0:
            return None, None
            
        return (x_sum / count) * w, (y_sum / count) * h

    def _draw_and_process(self, frame, h, w, gesture_controller):
        with self.lock:
            landmarks_list = self.current_landmarks

        if not landmarks_list:
            return

        for hand_idx, hand_landmarks in enumerate(landmarks_list):
            
            # Отрисовка скелета
            for idx_start, idx_end in HAND_CONNECTIONS:
                lm_start = hand_landmarks[idx_start]
                lm_end = hand_landmarks[idx_end]
                
                pt1 = (int(lm_start.x * w), int(lm_start.y * h))
                pt2 = (int(lm_end.x * w), int(lm_end.y * h))
                
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            for lm in hand_landmarks:
                pt = (int(lm.x * w), int(lm.y * h))
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)

            # Логика управления (только первая рука)
            if hand_idx == 0:
                center_x, center_y = self._get_palm_center(hand_landmarks, w, h)
                
                if center_x is not None and center_y is not None:
                    norm_x = center_x / w
                    norm_y = center_y / h
                    
                    # 1. Обработка свайпов
                    gesture = gesture_controller.add_coordinate(norm_x, norm_y)
                    
                    # 2. Обработка ЗУМА (новая логика)
                    landmarks_data = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                    zoom_state = gesture_controller.process_gesture(landmarks=landmarks_data, palm_y=norm_y)
                    
                    # Визуализация и логирование
                    status_text = ""
                    color = (255, 255, 255)
                    
                    if gesture != 'NONE':
                        status_text = f"Swipe: {gesture}"
                        color = (0, 255, 0)
                        print(f"Gesture detected: {gesture}")
                    
                    elif zoom_state != 'IDLE':
                        status_text = f"ZOOM: {zoom_state}"
                        color = (0, 255, 255) # Желтый для зума
                        print(f"Zoom state: {zoom_state}")
                    elif gesture_controller.zoom_mode:
                        # Если режим зума активен, но рука в нейтральной зоне
                        status_text = "ZOOM MODE (HOLD)"
                        color = (255, 165, 0) # Оранжевый

                    if status_text:
                        cv2.putText(frame, status_text, (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def close(self):
        if hasattr(self, 'detector'):
            self.detector.close()


def main():
    model_file = "hand_landmarker.task"
    
    tracker = HandTracker(model_file)
    
    gesture_controller = GestureController(
        buffer_size=15,
        x_threshold=0.15,
        y_threshold=0.05,
        cooldown_frames=30,
        pinch_threshold=0.05,      # Порог смыкания пальцев
        pinch_stability_frames=5   # ~0.08 сек задержка для защиты от шума
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        tracker.close()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("System started. Press 'q' to exit.")
    print("Controls:")
    print("- Swipe Left/Right for navigation")
    print("- Pinch fingers to activate Zoom Mode")
    print("- Move hand Up/Down while pinching to Zoom In/Out")
    print("- Release fingers to stop Zooming")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = tracker.process_frame(frame, gesture_controller)
            
            cv2.imshow("Hand Tracking", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()

if __name__ == "__main__":
    main()