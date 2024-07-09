from utils.MyModel_Resnet50 import MyModel
from utils.MyOnnxYolo import MyOnnxYolo


class JudgeCardCls:
    def __init__(self):
        self.model = MyModel("Model/resent_out51_Series01.pt")
        self.yolo = MyOnnxYolo("Model/yolo_handcard01.onnx")
        self.labels = ['2014-15 HOOPS', '2019-20 CONTENDERS', '2019-20 HOOPS PREMIUM', '2020-21 HOOPS',
                       '2020-21 HOOPS CITY EDITION',
                       '2020-21 HOOPS GOT NEXT', '2020-21 HOOPS NOW PLAYING', '2020-21 MOSAIC', '2020-21 PRIZM',
                       '2021 GOODWIN',
                       '2021-22 DONRUSS', '2021-22 DONRUSS MARVEL', '2021-22 DONRUSS RATED ROOKIE',
                       '2021-22 DONRUSS RETRO',
                       '2022-23 CONTENDERS', '2022-23 DONRUSS', '2022-23 DONRUSS OPTIC', '2022-23 PRIZM',
                       '2023 ABSOLUTE FOOTBALL',
                       '2023 ABSOLUTE NFL', '2023 BOWMAN BEST', '2023 GOODWIN', '2023 GOODWIN UD', '2023 GOOWIN',
                       '2023 KAKAWOW COSMOS', '2023 KAKAWOW MARVEL', '2023 MARVEL', '2023 METAL', '2023 PRIZM FOOTBALL',
                       '2023 PRIZM NFL', '2023 TOPPS CHROME SOCCER', '2023-24 CONTENDERS', '2023-24 DONRUSS',
                       '2023-24 DONRUSS CRAFTSMEN', '2023-24 DONRUSS CRUNCH', '2023-24 DONRUSS ELITE',
                       '2023-24 ORIGINS',
                       '2023-24 PHOTOGENIC', '2023-24 PRIZM', '2023-24 PRIZM EMERGENT', '2023-24 PRIZM GLOBAL',
                       '2023-24 PRIZM INSTANT', '2023-24 RECON', '2023-24 SELECT CONCOURSE',
                       '2023-24 SELECT CONCOURSE BLUE',
                       '2023-24 SELECT CONCOURSE_', '2023-24 SELECT MEZZANINE', '2023-24 SELECT PREMIER',
                       'CHINA SPORTS', 'POKEMEN',
                       'POKOMEN']

    def judge_card(self, img):
        self.yolo.set_result(img)
        card_img = self.yolo.get_max_img(cls_id=0)
        result = self.model.run(card_img)
        return self.labels[int(result)]
