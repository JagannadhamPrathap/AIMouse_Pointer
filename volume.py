import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import pyautogui

# ---------------- CONFIGURATION ---------------- #

wCam, hCam = 640, 480

frameR = 100
smoothening = 7

pTime = 0

plocX, plocY = 0, 0
clocX, clocY = 0, 0

previous_vol = 0

# ---------------- CAMERA ---------------- #

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

detector = htm.handDetector(maxHands=1)

wScr, hScr = autopy.screen.size()

# ---------------- MAIN LOOP ---------------- #

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    # Find hand landmarks
    img = detector.findHands(img)

    lmList, bbox = detector.findPosition(img)

    if lmList:

        # Index finger coordinates
        x1, y1 = lmList[8][1:]

        # Detect fingers
        fingers = detector.fingersUp()

        # Draw movement boundary
        cv2.rectangle(
            img,
            (frameR, frameR),
            (wCam - frameR, hCam - frameR),
            (255, 0, 255),
            2
        )

        # ---------------- MOUSE MOVEMENT ---------------- #

        if fingers[1] == 1 and fingers[2] == 0:

            # Convert camera coordinates to screen coordinates
            x3 = np.interp(
                x1,
                (frameR, wCam - frameR),
                (0, wScr)
            )

            y3 = np.interp(
                y1,
                (frameR, hCam - frameR),
                (0, hScr)
            )

            # Smooth mouse movement
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move mouse
            autopy.mouse.move(
                np.clip(wScr - clocX, 0, wScr - 1),
                np.clip(clocY, 0, hScr - 1)
            )

            # Draw index finger pointer
            cv2.circle(
                img,
                (x1, y1),
                15,
                (255, 0, 255),
                cv2.FILLED
            )

            plocX, plocY = clocX, clocY

        # ---------------- VOLUME CONTROL ---------------- #

        if fingers[1] == 1 and fingers[2] == 1:

            # Calculate distance between thumb and index finger
            length, img, lineInfo = detector.findDistance(
                4, 8, img
            )

            # Convert finger distance into volume
            vol = np.interp(
                length,
                [20, 200],
                [0, 100]
            )

            vol = int(np.clip(vol, 0, 100))

            # Decrease volume
            if vol < previous_vol:

                for i in range(previous_vol - vol):
                    pyautogui.press("volumedown")

                previous_vol = vol

            # Increase volume
            elif vol > previous_vol:

                for i in range(vol - previous_vol):
                    pyautogui.press("volumeup")

                previous_vol = vol

            # Display volume
            cv2.putText(
                img,
                f"Volume: {vol}",
                (20, 100),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                2
            )

    # ---------------- FPS ---------------- #

    cTime = time.time()

    fps = 1 / max(cTime - pTime, 0.001)

    pTime = cTime

    cv2.putText(
        img,
        str(int(fps)),
        (20, 50),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (255, 0, 0),
        3
    )

    # ---------------- DISPLAY ---------------- #

    cv2.imshow("AI Virtual Mouse - Volume Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---------------- CLEANUP ---------------- #

cap.release()

detector.close()

cv2.destroyAllWindows()