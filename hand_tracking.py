import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import time
import sys
import os

# Стандартная топология соединений руки (21 точка)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

class HandTracker:
    def __init__(self, model_path):
        self.current_landmarks = None
        self.lock = threading.Lock()
        
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found.")
            print("Download: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            sys.exit(1)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.7,  # Чуть выше порог, чтобы меньше ложных срабатываний
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.5,
            result_callback=self._on_results
        )
        
        try:
            self.detector = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Failed to init detector: {e}")
            sys.exit(1)

    def _on_results(self, result, output_image, timestamp_ms):
        """Callback получает результаты из отдельного потока"""
        with self.lock:
            # Сохраняем только если руки найдены, иначе оставляем None
            if result.hand_landmarks:
                self.current_landmarks = result.hand_landmarks
            else:
                self.current_landmarks = None

    def process_frame(self, frame):
        h, w, _ = frame.shape
        
        # Подготовка для MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        
        # Отправка на анализ (асинхронно)
        self.detector.detect_async(mp_image, timestamp_ms)
        
        # Отрисовка того, что есть на данный момент
        self._draw(frame, h, w)
        
        return frame

    def _draw(self, frame, h, w):
        with self.lock:
            landmarks_list = self.current_landmarks

        # Если рук нет в кадре - ничего не рисуем (убирает "призраков")
        if not landmarks_list:
            return

        for hand_landmarks in landmarks_list:
            # Рисуем кости
            for idx_start, idx_end in HAND_CONNECTIONS:
                lm_start = hand_landmarks[idx_start]
                lm_end = hand_landmarks[idx_end]
                
                x1, y1 = int(lm_start.x * w), int(lm_start.y * h)
                x2, y2 = int(lm_end.x * w), int(lm_end.y * h)
                
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Рисуем суставы
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    def close(self):
        self.detector.close()


def main():
    model_file = "hand_landmarker.task"
    
    tracker = HandTracker(model_file)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        tracker.close()
        sys.exit(1)

    # Уменьшаем буфер ввода, чтобы снизить задержку (latency)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Можно попробовать снизить разрешение для увеличения FPS, если тормозит
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Hand tracker started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = tracker.process_frame(frame)
            
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