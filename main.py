import os
from ultralytics import YOLO
import cv2
from utils.letterbox import letterbox
from utils.match_template import MatchTemplate
from utils.JudgeCardCls import JudgeCardCls
import numpy as np

Templates_dir = "./Templates"
model = YOLO("Model/yolov8obb_card_box05.pt")


def put_text(frame, text):
    # 选择文本字体
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 获取文本框的尺寸
    text_size, _ = cv2.getTextSize(text, font, 1, 2)

    # 设置文本颜色和背景颜色
    text_color = (255, 255, 255)  # 白色
    bg_color = (0, 0, 0)  # 黑色

    # 设置文本框的位置
    text_x = 10
    text_y = 40

    # 在图像上绘制背景矩形
    bg_start_x = text_x
    bg_start_y = text_y - text_size[1] - 5
    bg_end_x = bg_start_x + text_size[0] + 10
    bg_end_y = text_y + 5
    cv2.rectangle(frame, (bg_start_x, bg_start_y), (bg_end_x, bg_end_y), bg_color, -1)

    # 在图像上绘制文本
    cv2.putText(frame, text, (text_x, text_y), font, 1, text_color, 2)
    return frame


def draw_box(img):
    img = letterbox(img, target_size=img.shape[0])

    results = model.predict(img)
    points1 = results[0].obb.xyxyxyxy.numpy().astype(np.int32)
    cv2.polylines(img, [points1], True, (0, 20, 250), 3)

    if matchTemplate.template_img_paths is not None:
        points2 = matchTemplate.match(img)
        cv2.polylines(img, [points2], True, (20, 250, 0), 3)

        # 计算内外框间距
        points1

    return img



if __name__ == '__main__':
    img_path = r"C:\Code\ML\Image\Match_template\Card_innerbox\CHINA SPORTS\2024_07_03___05_37_12.jpg"
    save_dir = r"C:\Code\ML\Image\Card_test\match_test_result"
    judgeCardCls = JudgeCardCls()
    cls_name = judgeCardCls.judge_card(img_path)
    print(f"预测分类: {cls_name}")

    template_img_dir = os.path.join(Templates_dir, cls_name)
    matchTemplate = MatchTemplate(template_img_dir)

    img = cv2.imread(img_path)
    img = draw_box(img)

    cv2.imwrite(os.path.join(save_dir, str(len(os.listdir(save_dir)) + 1) + ".jpg"), img)
    # cv2.imshow("img", img)
