import cv2
import easyocr
import numpy as np


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)


def inpaint_text(img):
    reader = easyocr.Reader(["en"])
    read_text_result = reader.readtext(img, min_size=7)
    mask = np.zeros(img.shape[:2], dtype="uint8")
    box_list = []
    box_list.extend(row[0] for row in read_text_result)

    for box in box_list:
        x0, y0 = box[0]
        x1, y1 = box[1]
        x2, y2 = box[2]
        x3, y3 = box[3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

        thickness = int(abs(x2 - x1) + abs(y2 - y1))

        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
        img = cv2.inpaint(img[..., :3], mask, 10, cv2.INPAINT_NS)

    return img
