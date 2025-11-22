import cv2
import numpy as np
import mss
import pyautogui
import time

monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
sct = mss.mss()

last_click_time = 0
click_interval = 0.1

while True:
    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([5, 120, 120])
    upper_orange = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius > 5:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.putText(frame, f"({int(x)}, {int(y)})", (int(x)-40, int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            current_time = time.time()
            if current_time - last_click_time > click_interval:
                pyautogui.moveTo(int(x), int(y))
                pyautogui.click()
                last_click_time = current_time

    cv2.imshow("Orange Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
