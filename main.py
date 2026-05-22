import sys
import time
import pyautogui
from hand_tracking import HandTracker
from gesture_controller import GestureController

def main():
    try:
        tracker = HandTracker("hand_landmarker.task")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Init error: {e}")
        sys.exit(1)

    controller = GestureController()
    
    print("Running... Press Ctrl+C to stop.")
    
    last_action_time = 0
    action_delay = 0.4

    try:
        while True:
            landmarks, px, py = tracker.get_frame_data()
            
            if landmarks is None:
                time.sleep(0.01)
                continue

            gesture = controller.process(landmarks, px, py)
            
            now = time.time()
            if not gesture or (now - last_action_time) < action_delay:
                continue

            if gesture == 'LEFT':
                pyautogui.press('left')
                print("< Left")
                last_action_time = now
            elif gesture == 'RIGHT':
                pyautogui.press('right')
                print("> Right")
                last_action_time = now
            elif gesture == 'ZOOM_IN':
                pyautogui.hotkey('ctrl', '=')
                print("+ Zoom In")
                last_action_time = now
            elif gesture == 'ZOOM_OUT':
                pyautogui.hotkey('ctrl', '-')
                print("- Zoom Out")
                last_action_time = now

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        tracker.close()

if __name__ == "__main__":
    main()