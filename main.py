import cv2
from vision import UnoDetector


def main():
    detector = UnoDetector("models/uno_model.pt")

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        marked_frame, current_cards = detector.process_frame(frame)

        cv2.imshow("UnoMind Vision", marked_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
