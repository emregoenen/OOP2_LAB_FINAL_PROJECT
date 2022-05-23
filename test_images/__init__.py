import os
from skimage import io
from random import randint


# class TestImages:
#     def __init__(self):
#         self.path = os.path.abspath(__file__)
#         self.dir_path = os.path.dirname(self.path)
#         print(self.path)
#         print(self.dir_path)
#
#     def get_image(self, image_name):
#         image_path = self.dir_path + '\\' + str(image_name) + ".png"
#         print(image_name)
#         img = io.imread(image_path)
#         return img

test_images = ["astronaut",
               "brick",
               "camera",
               "cell",
               "chelsea",
               "chessboard_GRA",
               "chessboard_RGB",
               "clock_motion",
               "coffee",
               "coins",
               "color",
               "grass",
               "gravel",
               "horse",
               "hubble_deep_field",
               "ihc",
               "logo",
               "microaneurysms",
               "moon",
               "motorcycle_left",
               "motorcycle_right",
               "page",
               "phantom",
               "retina",
               "rocket",
               "text"]


def get_test_image(image_name):
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    image_path = dir_path + '\\' + str(image_name) + ".png"
    img = io.imread(image_path)
    return img


def get_random_image():
    get_test_image(test_images[randint(0,len(test_images)-1)])