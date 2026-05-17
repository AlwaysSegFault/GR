from collections import deque

class GestureController:
    def __init__(self, buffer_size=15, x_threshold=0.15, y_threshold=0.05, cooldown_frames=30):
        self.buffer_size = buffer_size
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.cooldown_frames = cooldown_frames
        
        # Буфер координат (x, y)
        self.buffer = deque(maxlen=buffer_size)
        self.cooldown_counter = 0

    def add_coordinate(self, x, y):
        """
        Принимает нормализованные координаты центра ладони (0.0 - 1.0).
        Возвращает: 'LEFT', 'RIGHT' или 'NONE'.
        """
        # Обновление кулдауна
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            # Продолжаем заполнять буфер даже во время кулдауна, 
            # чтобы следующий жест определился сразу после его окончания
            self.buffer.append((x, y))
            return 'NONE'

        self.buffer.append((x, y))

        # Ждем заполнения буфера
        if len(self.buffer) < self.buffer_size:
            return 'NONE'

        first_x, first_y = self.buffer[0]
        last_x, last_y = self.buffer[-1]

        delta_x = last_x - first_x
        delta_y = abs(last_y - first_y)

        # Проверка условий свайпа
        if abs(delta_x) > self.x_threshold and delta_y < self.y_threshold:
            self.cooldown_counter = self.cooldown_frames
            self.buffer.clear()  # Сброс буфера для следующего жеста
            
            return 'RIGHT' if delta_x > 0 else 'LEFT'

        return 'NONE'

    def reset(self):
        self.buffer.clear()
        self.cooldown_counter = 0