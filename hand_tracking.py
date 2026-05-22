import threading
import time
import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

PALM_POINTS = [0, 5, 9, 13, 17]


class ExponentialMovingAverage:
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.smoothed_values = None
    
    def update(self, landmarks):
        if not landmarks:
            return None
        
        current_values = []
        for lm in landmarks:
            if hasattr(lm, 'x'):
                current_values.append((lm.x, lm.y, lm.z))
            else:
                current_values.append(tuple(lm[:3]) if len(lm) >= 3 else (lm[0], lm[1], 0.0))
        
        if self.smoothed_values is None:
            self.smoothed_values = current_values
            return current_values
        
        smoothed = []
        for i, (cx, cy, cz) in enumerate(current_values):
            sx, sy, sz = self.smoothed_values[i]
            new_x = self.alpha * cx + (1 - self.alpha) * sx
            new_y = self.alpha * cy + (1 - self.alpha) * sy
            new_z = self.alpha * cz + (1 - self.alpha) * sz
            smoothed.append((new_x, new_y, new_z))
        
        self.smoothed_values = smoothed
        return smoothed
    
    def reset(self):
        self.smoothed_values = None


class HandTracker:
    def __init__(self, model_path):
        self.landmarks = None
        self.lock = threading.Lock() if 'threading' in globals() else None
        self.ema = ExponentialMovingAverage(alpha=0.25)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_path} not found")

        base_opts = base_options.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            result_callback=self._callback
        )
        
        self.detector = vision.HandLandmarker.create_from_options(opts)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _callback(self, result, output_image, timestamp_ms):
        if self.lock:
            with self.lock:
                self.landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
        else:
            self.landmarks = result.hand_landmarks[0] if result.hand_landmarks else None

    def get_frame_data(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        ts = int(time.time() * 1000)
        self.detector.detect_async(mp_img, ts)
        
        time.sleep(0.005) 

        current_lms = None
        if self.lock:
            with self.lock:
                current_lms = self.landmarks
        else:
            current_lms = self.landmarks

        if not current_lms:
            return None, None, None

        smoothed_lms = self.ema.update(current_lms)
        if not smoothed_lms:
            return None, None, None

        cx, cy = 0, 0
        for idx in PALM_POINTS:
            cx += smoothed_lms[idx][0]
            cy += smoothed_lms[idx][1]
        cx /= len(PALM_POINTS)
        cy /= len(PALM_POINTS)

        return smoothed_lms, cx, cy

    def close(self):
        self.detector.close()
        self.cap.release()