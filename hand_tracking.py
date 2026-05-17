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
            num_hands=2,  # Разрешаем детектить до 2 рук
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
        """Callback для получения результатов в отдельном потоке"""
        with self.lock:
            if result.hand_landmarks:
                self.current_landmarks = result.hand_landmarks
            else:
                self.current_landmarks = None

    def process_frame(self, frame, gesture_controller):
        h, w, _ = frame.shape
        
        # Конвертация для MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        
        # Асинхронная детекция
        self.detector.detect_async(mp_image, timestamp_ms)
        
        # Отрисовка и логика жестов
        self._draw_and_process(frame, h, w, gesture_controller)
        
        return frame

    def _get_palm_center(self, landmarks, w, h):
        """Вычисляет средний центр ладони по опорным точкам"""
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

        # Проходим по ВСЕМ обнаруженным рукам
        for hand_idx, hand_landmarks in enumerate(landmarks_list):
            
            # 1. Отрисовка скелета для каждой руки
            for idx_start, idx_end in HAND_CONNECTIONS:
                lm_start = hand_landmarks[idx_start]
                lm_end = hand_landmarks[idx_end]
                
                pt1 = (int(lm_start.x * w), int(lm_start.y * h))
                pt2 = (int(lm_end.x * w), int(lm_end.y * h))
                
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            for lm in hand_landmarks:
                pt = (int(lm.x * w), int(lm.y * h))
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)

            # 2. Вычисление центра и обработка жеста
            # Для управления используем только первую руку (индекс 0), 
            # чтобы избежать конфликтов сигналов, но рисуем обе.
            if hand_idx == 0:
                center_x, center_y = self._get_palm_center(hand_landmarks, w, h)
                
                if center_x is not None and center_y is not None:
                    # Нормализация координат для контроллера (0.0 - 1.0)
                    norm_x = center_x / w
                    norm_y = center_y / h
                    
                    # Получение результата жеста
                    gesture = gesture_controller.add_coordinate(norm_x, norm_y)
                    
                    # Логирование в консоль (можно убрать в продакшене)
                    if gesture != 'NONE':
                        print(f"Gesture detected: {gesture}")
                        # TODO: Добавить логику действий здесь

    def close(self):
        if hasattr(self, 'detector'):
            self.detector.close()


def main():
    model_file = "hand_landmarker.task"
    
    # Инициализация трекера
    tracker = HandTracker(model_file)
    
    # Инициализация контроллера жестов
    gesture_controller = GestureController(
        buffer_size=15,
        x_threshold=0.15,
        y_threshold=0.05,
        cooldown_frames=30
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        tracker.close()
        sys.exit(1)

    # Оптимизация буфера для минимальной задержки
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("System started. Press 'q' to exit.")

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