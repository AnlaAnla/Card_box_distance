import cv2
import numpy as np


def letterbox(img, target_size=1024):
    # 读取原始图像
    # img = cv2.imread(r"C:\Code\ML\Image\Card_test\match_test\2024_06_19___06_18_46.jpg")

    # 获取原始图像的高度和宽度
    h, w = img.shape[:2]

    # 计算目标缩放比例
    # target_size = 1024
    ratio = target_size / max(h, w)

    # 计算缩放后的尺寸
    new_h = int(h * ratio)
    new_w = int(w * ratio)

    # 执行缩放
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建一个新的画布,用黑色填充
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # 将缩放后的图像粘贴到新画布的中心
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    return canvas
