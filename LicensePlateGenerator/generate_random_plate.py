# encoding=utf-8
import os.path

import cv2

from generate_multi_plate import MultiPlateGenerator

# out_dir = "/users/sunshangyun/LicensePlateGenerator/plate_img/"
out_dir = "e:/plates"
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
    print("[Info]: {:s} made.".format(out_dir))

multi_plate_generator = MultiPlateGenerator('plate_model', 'font_model')
i = 0
while i < 2000:
    img_plate_model, number_xy, plate_number, bg_color, is_double = multi_plate_generator.generate_plate()

    plate_model = "double"
    if is_double == False:
        plate_model = "single"

    # 指定颜色
    print("plate_model ==" + str(plate_model) + " bg_color ==" + str(bg_color))

    if plate_model == "single" and bg_color == "green_car":
        img_name = str(plate_number) + \
                   "_" + str(bg_color) \
                   + "_" + \
                   str(plate_model) \
                   + ".jpg"
        save_img_path = out_dir + "/" + img_name
        cv2.imwrite(save_img_path, img_plate_model)
        i += 1
