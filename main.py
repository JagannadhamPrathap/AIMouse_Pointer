import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import os

# ---------------- CONFIGURATION ---------------- #

wCam, hCam = 640, 480

frameR = 100
smoothening = 5

pTime = 0

plocX, plocY = 0, 0
clocX, clocY = 0, 0

leftClicking = False
rightClicking = False
driveOpening = False

# ---------------- CAMERA ---------------- #

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

detector = htm.handDetector(maxHands=1)

wScr, hScr = autopy.screen.size()

# ---------------- MAIN LOOP ---------------- #

try:

    while True:

        success, img = cap.read()

        if not success:
            print("Error: Cannot read webcam frame.")
            break

        # Mirror camera
        img = cv2.flip(img, 1)

        # Detect hands
        img = detector.findHands(img)

        lmList, bbox = detector.findPosition(img)

        if lmList:

            x1, y1 = lmList[8][1:]

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

                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # Move cursor
                autopy.mouse.move(
                    int(np.clip(clocX, 0, wScr - 1)),
                    int(np.clip(clocY, 0, hScr - 1))
                )

                cv2.circle(
                    img,
                    (x1, y1),
                    15,
                    (255, 0, 255),
                    cv2.FILLED
                )

                plocX, plocY = clocX, clocY

            # ---------------- LEFT CLICK ---------------- #

            if fingers[1] == 1 and fingers[2] == 1:

                length, img, lineInfo = detector.findDistance(
                    8, 12, img
                )

                if length < 40:

                    cv2.circle(
                        img,
                        (lineInfo[4], lineInfo[5]),
                        15,
                        (0, 255, 0),
                        cv2.FILLED
                    )

                    if not leftClicking:

                        autopy.mouse.click()
                        leftClicking = True

                else:
                    leftClicking = False

            else:
                leftClicking = False

            # ---------------- RIGHT CLICK ---------------- #

            if fingers[1] == 1 and fingers[2] == 1:

                length, img, lineInfo = detector.findDistance(
                    4, 20, img
                )

                if length < 40:

                    cv2.circle(
                        img,
                        (lineInfo[4], lineInfo[5]),
                        15,
                        (0, 255, 0),
                        cv2.FILLED
                    )

                    if not rightClicking:

                        autopy.mouse.click(
                            button=autopy.mouse.Button.RIGHT
                        )

                        rightClicking = True

                else:
                    rightClicking = False

            else:
                rightClicking = False

            # ---------------- OPEN A DRIVE ---------------- #

            if fingers[1] == 1 and fingers[2] == 1:

                length, img, lineInfo = detector.findDistance(
                    4, 16, img
                )

                if length < 40:

                    cv2.circle(
                        img,
                        (lineInfo[4], lineInfo[5]),
                        15,
                        (0, 255, 0),
                        cv2.FILLED
                    )

                    if not driveOpening:

                        try:
                            os.startfile("A:\\")
                        except OSError as e:
                            print("Cannot open A drive:", e)

                        driveOpening = True

                else:
                    driveOpening = False

            else:
                driveOpening = False

        else:

            leftClicking = False
            rightClicking = False
            driveOpening = False

        # ---------------- FPS ---------------- #

        cTime = time.time()

        fps = 1 / max(cTime - pTime, 0.001)

        pTime = cTime

        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (20, 50),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 0),
            3
        )

        # ---------------- DISPLAY ---------------- #

        cv2.imshow("AI Virtual Mouse", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# ---------------- CLEANUP ---------------- #

finally:

    cap.release()

    detector.close()

    cv2.destroyAllWindows()