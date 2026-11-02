import cv2
import mediapipe as mp
import time
import math
import os


class handDetector:

    def __init__(
        self,
        mode=False,
        maxHands=1,
        detectionCon=0.5,
        trackCon=0.5
    ):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.lmList = []
        self.results = None

        self.tipIds = [4, 8, 12, 16, 20]

        self.mpImage = mp.Image
        self.mpImageFormat = mp.ImageFormat

        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "hand_landmarker.task"
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=self.maxHands,
            min_hand_detection_confidence=self.detectionCon,
            min_hand_presence_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(
            options
        )

        self.timestamp = 0

    # ---------------- FIND HANDS ----------------

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mpImg = self.mpImage(
            image_format=self.mpImageFormat.SRGB,
            data=imgRGB
        )

        self.timestamp = max(
            self.timestamp + 1,
            int(time.monotonic() * 1000)
        )

        self.results = self.hands.detect_for_video(
            mpImg,
            self.timestamp
        )

        if draw and self.results.hand_landmarks:

            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ]

            h, w, _ = img.shape

            for handLms in self.results.hand_landmarks:

                points = []

                for lm in handLms:

                    cx = int(lm.x * w)
                    cy = int(lm.y * h)

                    points.append((cx, cy))

                    cv2.circle(
                        img,
                        (cx, cy),
                        5,
                        (255, 0, 255),
                        cv2.FILLED
                    )

                for start, end in connections:

                    cv2.line(
                        img,
                        points[start],
                        points[end],
                        (0, 255, 0),
                        2
                    )

        return img

    # ---------------- FIND POSITION ----------------

    def findPosition(self, img, handNo=0, draw=True):

        xList = []
        yList = []

        bbox = []
        self.lmList = []

        if self.results and self.results.hand_landmarks:

            if handNo >= len(self.results.hand_landmarks):
                return self.lmList, bbox

            myHand = self.results.hand_landmarks[handNo]

            h, w, _ = img.shape

            for id, lm in enumerate(myHand):

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                xList.append(cx)
                yList.append(cy)

                self.lmList.append([id, cx, cy])

            xmin = min(xList)
            xmax = max(xList)

            ymin = min(yList)
            ymax = max(yList)

            bbox = xmin, ymin, xmax, ymax

            if draw:

                cv2.rectangle(
                    img,
                    (xmin - 20, ymin - 20),
                    (xmax + 20, ymax + 20),
                    (0, 255, 0),
                    2
                )

        return self.lmList, bbox

    # ---------------- FINGERS UP ----------------

    def fingersUp(self):

        fingers = []

        if not self.lmList:
            return [0, 0, 0, 0, 0]

        # Thumb
        if self.lmList[4][1] > self.lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for id in range(1, 5):

            if (
                self.lmList[self.tipIds[id]][2]
                <
                self.lmList[self.tipIds[id] - 2][2]
            ):

                fingers.append(1)

            else:
                fingers.append(0)

        return fingers

    # ---------------- FIND DISTANCE ----------------

    def findDistance(
        self,
        p1,
        p2,
        img,
        draw=True,
        r=15,
        t=3
    ):

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if draw:

            cv2.line(
                img,
                (x1, y1),
                (x2, y2),
                (255, 0, 255),
                t
            )

            cv2.circle(
                img,
                (x1, y1),
                r,
                (255, 0, 255),
                cv2.FILLED
            )

            cv2.circle(
                img,
                (x2, y2),
                r,
                (255, 0, 255),
                cv2.FILLED
            )

            cv2.circle(
                img,
                (cx, cy),
                r,
                (0, 0, 255),
                cv2.FILLED
            )

        length = math.hypot(
            x2 - x1,
            y2 - y1
        )

        return length, img, [
            x1, y1,
            x2, y2,
            cx, cy
        ]

    # ---------------- CLOSE ----------------

    def close(self):

        if self.hands:
            self.hands.close()


# ---------------- TEST MODULE ----------------

def main():

    pTime = 0

    cap = cv2.VideoCapture(0)

    detector = handDetector(maxHands=1)

    while True:

        success, img = cap.read()

        if not success:
            break

        img = cv2.flip(img, 1)

        img = detector.findHands(img)

        lmList, bbox = detector.findPosition(img)

        if lmList:
            print(lmList[4])

        cTime = time.time()

        fps = 1 / max(cTime - pTime, 0.001)

        pTime = cTime

        cv2.putText(
            img,
            str(int(fps)),
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 255),
            3
        )

        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()

    detector.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()