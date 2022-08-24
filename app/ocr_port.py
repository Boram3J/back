import importlib
import os
from pathlib import Path

import numpy as np
from munch import Munch
from torch import nn

from app.util import add_sys_path, singleton

ocr_root = Path(__file__).resolve().parent.parent / "ocr"

with add_sys_path(ocr_root):
    file_utils = importlib.import_module("file_utils")
    net_utils = importlib.import_module("net_utils")
    imgproc = importlib.import_module("imgproc")
    opt = importlib.import_module("opt")
    bubble_detect = importlib.import_module("object_detection.bubble").test_net
    cut_detect = importlib.import_module("object_detection.cut").test_opencv
    line_text_detect = importlib.import_module("text_detection.line_text").test
    line_text_recognize = importlib.import_module("text_recognition.line_text").test_net
    gen_text_to_image = importlib.import_module(
        "text_recognition.ltr_utils"
    ).gen_txt_to_image
    papago_translation = importlib.import_module("translation.papago").translation


@singleton
class ModelManager:
    def __init__(self) -> None:
        self._models = None
        self.text_detector = ocr_root / "weights/Line-Text-Detector.pth"
        self.text_recognizer = ocr_root / "weights/Line-Text-Recognizer.pth"
        self.object_detector = ocr_root / "weights/Speech-Bubble-Detector.pth"

    def load(self) -> None:
        net_cfg = Munch(
            object=True,
            ocr=True,
            text_detector=str(self.text_detector),
            text_recognizer=str(self.text_recognizer),
            object_detector=str(self.object_detector),
        )
        self._models = net_utils.load_net(net_cfg)

    @property
    def models(self) -> dict[str, nn.Module]:
        if self._models is None:
            self.load()
        return self._models


def run_ocr_and_translate(
    img: np.ndarray,
    bg_type: str = "white",
    bubble_threshold: float = 0.995,
    box_threshold: int = 7000,
    translate: bool = True,
) -> np.ndarray:
    """img: 0-255 uint8 height x width x 3 image"""
    file_utils.rm_all_dir(dir=str(ocr_root / "result"))
    file_utils.mkdir(
        dir=[
            str(ocr_root / "result"),
            str(ocr_root / "result/bubbles"),
            str(ocr_root / "result/cuts"),
            str(ocr_root / "result/demo"),
            str(ocr_root / "result/chars"),
        ]
    )

    img_blob, img_scale = imgproc.getImageBlob(img)
    models = ModelManager().models
    f_rcnn_param = [img_blob, img_scale, opt.LABEL]

    demo, image, bubbles, dets_bubbles = bubble_detect(
        model=models["bubble_detector"],
        image=img,
        params=f_rcnn_param,
        cls=bubble_threshold,
        bg=bg_type,
    )
    demo, cuts = cut_detect(image=image, demo=demo, bg=bg_type, size=box_threshold)

    str_cnt = "0000"
    demo, space, warps = line_text_detect(
        model=models["text_detector"],
        demo=demo,
        bubbles=imgproc.cpImage(bubbles),
        dets=dets_bubbles,
        img_name=str_cnt,  # ???
        save_to=str(ocr_root / "result/chars/") + "/",  # disable saving
    )

    ### optional savings
    file_utils.saveAllImages(
        save_to=str(ocr_root / "result/bubbles/") + "/",
        imgs=bubbles,
        index1=str_cnt,
        ext=".png",
    )
    file_utils.saveAllImages(
        save_to=str(ocr_root / "result/cuts/") + "/",
        imgs=cuts,
        index1=str_cnt,
        ext=".png",
    )
    ###

    file_utils.saveText(
        save_to=str(ocr_root / "result/") + "/", text=space, name="spaces"
    )

    label_mapper = file_utils.makeLabelMapper(
        load_from=str(ocr_root / "text_recognition/labels-2213.txt")
    )

    spaces, _ = file_utils.loadSpacingWordInfo(
        load_from=str(ocr_root / "result/spaces.txt")
    )

    ocr_text = ocr_root / "result/ocr.txt"

    line_text_recognize(
        model=models["text_recognizer"],
        mapper=label_mapper,
        spaces=spaces,
        load_from=str(ocr_root / "result/chars/") + "/",
        save_to=str(ocr_text),
    )

    if translate and ocr_text.read_text(encoding="utf-8").strip():
        papago_translation(
            load_from=str(ocr_text),
            save_to=str(ocr_root / "result/english_ocr.txt"),
            id=os.environ["PAPAGO_ID"],
            pw=os.environ["PAPAGO_PW"],
        )

        gen_text_to_image(
            load_from=str(ocr_root / "result/english_ocr.txt"), warp_item=warps
        )

    return demo
