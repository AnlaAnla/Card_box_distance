import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings


# 对4个角进行模板匹配
def match_loc(img, template_img_path):
    # 读取目标图像和模板图像
    template_img = cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    w, h = template_img.shape[::-1]

    # 使用cv2.matchTemplate进行模板匹配
    result = cv2.matchTemplate(target_img, template_img, cv2.TM_CCOEFF_NORMED)

    # 设置相似度阈值（例如0.3，较低的阈值）
    # threshold = 0.6
    # loc = np.where(result >= threshold)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # 在目标图像上绘制匹配结果

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    return [top_left, bottom_right]


class MatchTemplate:
    def __init__(self, template_img_dir):
        if not os.path.exists(template_img_dir):
            warnings.warn(f'不存在该模板目录: {template_img_dir}')
            self.template_img_paths = None
        else:
            self.template_img_paths = [os.path.join(template_img_dir, "corner1.jpg"),
                                       os.path.join(template_img_dir, "corner2.jpg"),
                                       os.path.join(template_img_dir, "corner3.jpg"),
                                       os.path.join(template_img_dir, "corner4.jpg")
                                       ]

    # 获取模板的宽度和高度

    def match(self, img: np.ndarray):
        locations = []
        for template_img_path in self.template_img_paths:
            locations.append(match_loc(img, template_img_path))

        point = [[locations[0][0][0], locations[0][0][1]], [locations[1][1][0], locations[1][0][1]],
                 [locations[3][1][0], locations[3][1][1]], [locations[2][0][0], locations[2][1][1]]]
        points = np.array(point)
        # cv2.polylines(img, [points], True, (0, 255, 0), 2)
        return points
