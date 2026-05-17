#!/usr/bin/env python3
"""
Скрипт для обнаружения рук и отрисовки landmarks в реальном времени
с использованием OpenCV и MediaPipe Hands.

Требования:
    pip install opencv-python mediapipe==0.10.9
    
Версия MediaPipe 0.10.9 использует стабильный классический API.
Более новые версии (0.11+) требуют загрузки моделей и имеют другой API.
"""

import cv2
import mediapipe as mp
import sys


def check_mediapipe_version():
    """Проверка совместимости версии MediaPipe."""
    import re
    version = mp.__version__
    match = re.match(r'(\d+)\.(\d+)', version)
    if match:
        major, minor = int(match.group(1)), int(match.group(2))
        if major == 0 and minor <= 10:
            return True, version
        elif major >= 1 or (major == 0 and minor >= 11):
            return False, version
    return None, version


def main():
    # Проверка версии MediaPipe
    is_compatible, version = check_mediapipe_version()
    
    if is_compatible is False:
        print(f"Предупреждение: Версия MediaPipe {version} может быть несовместима.")
        print("Рекомендуется установить версию 0.10.9:")
        print("  pip install mediapipe==0.10.9")
        print("Продолжение работы с текущей версией...\n")
    
    # Инициализация MediaPipe Hands (классический API)
    try:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
    except AttributeError as e:
        print(f"Ошибка: Не удалось импортировать модули MediaPipe: {e}")
        print("Убедитесь, что установлена корректная версия: pip install mediapipe==0.10.9")
        sys.exit(1)

    # Открываем камеру (0 - основная камера)
    cap = cv2.VideoCapture(0)

    # Проверка успешности открытия камеры
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру.")
        print("Проверьте подключение камеры и убедитесь, что она не используется другими приложениями.")
        sys.exit(1)

    print("Камера успешно открыта. Нажмите 'q' для выхода.")

    # Создаем объект Hands с настройками
    with mp_hands.Hands(
        static_image_mode=False,  # Режим видео (быстрее)
        max_num_hands=2,          # Максимальное количество рук для обнаружения
        min_detection_confidence=0.5,  # Минимальная уверенность детекции
        min_tracking_confidence=0.5    # Минимальная уверенность трекинга
    ) as hands:

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

            # Обработка изображения и обнаружение рук
            results = hands.process(rgb_frame)

            # Если руки обнаружены
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Отрисовка landmarks и связей
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Добавление текста с информацией
            num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
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

    # Освобождение ресурсов
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
