from collections import deque
import math

class GestureController:
    def __init__(self, buffer_size=15, x_threshold=0.15, y_threshold=0.05, cooldown_frames=30, 
                 pinch_threshold=0.05, pinch_stability_frames=5):
        """
        pinch_stability_frames: количество кадров, в которых должен удерживаться пинч 
        для активации зума (защита от случайного моргания пальцев). 
        При 60 FPS: 5 кадров ≈ 0.08 сек.
        """
        self.buffer_size = buffer_size
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.cooldown_frames = cooldown_frames
        self.pinch_threshold = pinch_threshold
        self.pinch_stability_frames = pinch_stability_frames
        
        # Буфер координат (x, y) для свайпов
        self.buffer = deque(maxlen=buffer_size)
        self.cooldown_counter = 0
        
        # Переменные для режима ЗУМ
        self.pinch_hold_counter = 0
        self.zoom_mode = False
        self.anchor_y = None  # Точка "захвата", от которой считаем отклонение

    def calculate_pinch_distance(self, landmarks):
        """
        Рассчитывает евклидово расстояние между кончиком большого пальца (landmark 4)
        и указательного пальца (landmark 8).
        """
        if len(landmarks) < 9:
            return float('inf')
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        dx = thumb_tip[0] - index_tip[0]
        dy = thumb_tip[1] - index_tip[1]
        dz = thumb_tip[2] - index_tip[2] if len(thumb_tip) > 2 else 0
        
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def is_pinch(self, distance):
        return distance < self.pinch_threshold

    def add_coordinate(self, x, y):
        """Логика свайпов (LEFT/RIGHT) осталась без изменений."""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.buffer.append((x, y))
            return 'NONE'

        self.buffer.append((x, y))

        if len(self.buffer) < self.buffer_size:
            return 'NONE'

        first_x, first_y = self.buffer[0]
        last_x, last_y = self.buffer[-1]

        delta_x = last_x - first_x
        delta_y = abs(last_y - first_y)

        if abs(delta_x) > self.x_threshold and delta_y < self.y_threshold:
            self.cooldown_counter = self.cooldown_frames
            self.buffer.clear()
            return 'RIGHT' if delta_x > 0 else 'LEFT'

        return 'NONE'

    def process_gesture(self, landmarks=None, palm_y=None):
        """
        Новая логика зума:
        1. Если пальцы сомкнуты -> входим в режим зума (запоминаем anchor_y).
        2. Пока держим пинч: сравниваем текущий palm_y с anchor_y.
        3. Если пальцы разжаты -> выходим из режима зума.
        """
        current_state = 'IDLE'

        if landmarks is not None:
            distance = self.calculate_pinch_distance(landmarks)
            is_pinch_now = self.is_pinch(distance)

            if is_pinch_now:
                # Увеличиваем счетчик стабильности
                self.pinch_hold_counter += 1
                
                # Если еще не в режиме зума, проверяем стабильность захвата
                if not self.zoom_mode:
                    if self.pinch_hold_counter >= self.pinch_stability_frames:
                        self.zoom_mode = True
                        self.anchor_y = palm_y  # Запоминаем точку старта
                else:
                    # Мы уже в режиме зума и держим пинч
                    if self.anchor_y is not None and palm_y is not None:
                        # Сравниваем с якорем. 
                        # В нормализованных координатах: 0.0 - верх, 1.0 - низ.
                        # Если рука выше (palm_y < anchor_y) -> ZOOM_IN
                        # Если рука ниже (palm_y > anchor_y) -> ZOOM_OUT
                        
                        diff = self.anchor_y - palm_y
                        
                        # Небольшой мертвый порог, чтобы не триггерить на дрожь в точке старта
                        if abs(diff) > (self.y_threshold / 2):
                            if diff > 0:
                                current_state = 'ZOOM_IN'
                            else:
                                current_state = 'ZOOM_OUT'
                        else:
                            current_state = 'IDLE' # Держим руку ровно
            else:
                # Пальцы разжаты
                self.pinch_hold_counter = 0
                self.zoom_mode = False
                self.anchor_y = None
                current_state = 'IDLE'

        return current_state

    def reset(self):
        self.buffer.clear()
        self.cooldown_counter = 0
        self.pinch_hold_counter = 0
        self.zoom_mode = False
        self.anchor_y = None