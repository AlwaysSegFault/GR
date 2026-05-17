#!/usr/bin/env python3
"""
Скрипт для обнаружения рук и отрисовки landmarks в реальном времени
с использованием OpenCV и MediaPipe Hands (новый API).

Требования:
    pip install opencv-python mediapipe
    
Поддерживает MediaPipe 0.10.30+ с новым API через tasks.python.vision
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import sys
import os


def create_hand_landmarker():
    """Создание детектора рук с использованием нового API."""
    try:
        # Пытаемся найти модель hand_landmarker.task
        model_path = None
        
        # Пытаемся найти модель в стандартных расположениях
        possible_paths = [
            os.path.join(os.path.dirname(mp.__file__), "data", "hand_landmarker.task"),
            os.path.join(os.path.dirname(mp.__file__), "tasks", "python", "vision", "data", "hand_landmarker.task"),
            os.path.join(os.path.dirname(mp.__file__), "tasks", "vision", "data", "hand_landmarker.task"),
            "/usr/local/lib/python3.12/site-packages/mediapipe/data/hand_landmarker.task",
            "/usr/local/lib/python3.12/site-packages/mediapipe/tasks/python/vision/data/hand_landmarker.task",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Модель найдена: {model_path}")
                break
        
        if model_path is None:
            # Если модель не найдена локально, используем URL для загрузки
            # MediaPipe автоматически загрузит модель при первом использовании
            print("Локальная модель не найдена. Будет использована модель по умолчанию.")
            print("При первом запуске может потребоваться загрузка модели из интернета.")
            base_options = BaseOptions(model_asset_path="")
        else:
            base_options = BaseOptions(model_asset_path=model_path)
        
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,  # Используем IMAGE mode для простоты
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        return HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Ошибка при создании HandLandmarker: {e}")
        import traceback
        traceback.print_exc()
        return None


def draw_landmarks(frame, hand_landmarks_list, connections_list):
    """Отрисовка landmarks и соединений на кадре."""
    import numpy as np
    
    h, w = frame.shape[:2]
    
    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        if idx < len(connections_list):
            connections = connections_list[idx]
            
            # Отрисовка точек (landmarks)
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Рисуем круги для каждой точки
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # Отрисовка соединений между точками
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    start_point = (
                        int(hand_landmarks[start_idx].x * w),
                        int(hand_landmarks[start_idx].y * h)
                    )
                    end_point = (
                        int(hand_landmarks[end_idx].x * w),
                        int(hand_landmarks[end_idx].y * h)
                    )
                    
                    cv2.line(frame, start_point, end_point, (0, 255, 255), 2)


# Стандартные соединения для рук MediaPipe
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Большой палец
    (0, 5), (5, 6), (6, 7), (7, 8),      # Указательный палец
    (0, 9), (9, 10), (10, 11), (11, 12), # Средний палец
    (0, 13), (13, 14), (14, 15), (15, 16), # Безымянный палец
    (0, 17), (17, 18), (18, 19), (19, 20), # Мизинец
    (5, 9), (9, 13), (13, 17)            # Соединения между пальцами у основания
]


def main():
    print(f"MediaPipe версия: {mp.__version__}")
    print("Использование нового API (tasks.python.vision)")
    print()

    # Открываем камеру (0 - основная камера)
    cap = cv2.VideoCapture(0)

    # Проверка успешности открытия камеры
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру.")
        print("Проверьте подключение камеры и убедитесь, что она не используется другими приложениями.")
        sys.exit(1)

    print("Камера успешно открыта. Нажмите 'q' для выхода.")

    # Создаем детектор рук
    landmarker = create_hand_landmarker()
    
    if landmarker is None:
        print("Не удалось создать детектор рук.")
        print("Попробуйте переустановить mediapipe: pip install --force-reinstall mediapipe")
        cap.release()
        sys.exit(1)

    try:
        while True:
            # Чтение кадра с камеры
            ret, frame = cap.read()

            if not ret:
                print("Ошибка: Не удалось прочитать кадр с камеры.")
                break

            # Отражение изображения по горизонтали для зеркального эффекта
            frame = cv2.flip(frame, 1)

            # Конвертация BGR в RGB (MediaPipe работает с RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Создание объекта Image для MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Обработка изображения и обнаружение рук
            detection_result = landmarker.detect(mp_image)

            # Если руки обнаружены
            if detection_result.hand_landmarks:
                draw_landmarks(frame, detection_result.hand_landmarks, 
                             [HAND_CONNECTIONS] * len(detection_result.hand_landmarks))

            # Добавление текста с информацией
            num_hands = len(detection_result.hand_landmarks) if detection_result.hand_landmarks else 0
            cv2.putText(
                frame,
                f"Hands detected: {num_hands}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Отображение результата
            cv2.imshow('Hand Tracking', frame)

            # Обработка нажатия клавиши 'q' для выхода
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Освобождение ресурсов
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Программа завершена.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем.")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
        sys.exit(1)
