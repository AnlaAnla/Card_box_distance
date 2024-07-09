import os

from ultralytics import YOLO
import cv2
from utils.letterbox import letterbox
from utils.match_template import MatchTemplate
import numpy as np

matchTemplate = MatchTemplate()
model = YOLO(r"C:\Code\ML\Model\Card_Box\yolov8obb_card_box05.pt")


def draw_box(img):
    img = letterbox(img, target_size=img.shape[0])
    results = model.predict(img)

    points1 = matchTemplate.match(img)
    points2 = results[0].obb.xyxyxyxy.numpy().astype(np.int32)

    cv2.polylines(img, [points1], True, (20, 250, 0), 3)
    cv2.polylines(img, [points2], True, (0, 20, 250), 3)
    return img


test_dir = r"C:\Code\ML\Image\Card_test\match_test"
save_dir = r"C:\Code\ML\Image\Card_test\match_test_result"

if __name__ == '__main__':
    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        img = draw_box(img)
        cv2.imwrite(os.path.join(save_dir, img_name), img)
        print(img_name)
