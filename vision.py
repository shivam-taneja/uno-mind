from ultralytics import YOLO
import cv2


class UnoDetector:
    def __init__(self, model_path="models/uno_model.pt"):
        self.model = YOLO(model_path)

        self.label_translator = {
            # 0-Series: RED
            "00": "Red 0",
            "01": "Red 1",
            "02": "Red 2",
            "03": "Red 3",
            "04": "Red 4",
            "05": "Red 5",
            "06": "Red 6",
            "07": "Red 7",
            "08": "Red 8",
            "09": "Red 9",
            "0A": "Red Skip",
            "0B": "Red Reverse",
            "0C": "Red Draw 2",
            # 1-Series: YELLOW
            "10": "Yellow 0",
            "11": "Yellow 1",
            "12": "Yellow 2",
            "13": "Yellow 3",
            "14": "Yellow 4",
            "15": "Yellow 5",
            "16": "Yellow 6",
            "17": "Yellow 7",
            "18": "Yellow 8",
            "19": "Yellow 9",
            "1A": "Yellow Skip",
            "1B": "Yellow Reverse",
            "1C": "Yellow Draw 2",
            # 2-Series: GREEN
            "20": "Green 0",
            "21": "Green 1",
            "22": "Green 2",
            "23": "Green 3",
            "24": "Green 4",
            "25": "Green 5",
            "26": "Green 6",
            "27": "Green 7",
            "28": "Green 8",
            "29": "Green 9",
            "2A": "Green Skip",
            "2B": "Green Reverse",
            "2C": "Green Draw 2",
            # 3-Series: BLUE
            "30": "Blue 0",
            "31": "Blue 1",
            "32": "Blue 2",
            "33": "Blue 3",
            "34": "Blue 4",
            "35": "Blue 5",
            "36": "Blue 6",
            "37": "Blue 7",
            "38": "Blue 8",
            "39": "Blue 9",
            "3A": "Blue Skip",
            "3B": "Blue Reverse",
            "3C": "Blue Draw 2",
            # 4-Series: WILDS
            "40": "Wild",
            "41": "Wild Draw 4",
        }

    def process_frame(self, frame):
        """Runs AI on the frame, draws boxes, and returns the list of cards found."""
        results = self.model(frame)
        detected_cards = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                class_id = int(box.cls[0])
                raw_name = self.model.names[class_id]

                clean_name = self.label_translator.get(raw_name, raw_name)

                detected_cards.append(clean_name)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.rectangle(frame, (x1, y1 - 25), (x1 + 150, y1), (0, 0, 0), -1)

                cv2.putText(
                    frame,
                    clean_name,
                    (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        return frame, detected_cards
