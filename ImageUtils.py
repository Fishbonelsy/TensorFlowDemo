from PIL import Image
import numpy as np


def get_image_data():
    size = 28, 28

    im = Image.open("origin4.jpg")
    im = im.convert("L")
    im = im.resize((28, 28))
    pixels = im.load()
    for x in range(im.width):
        for y in range(im.height):
            if pixels[x, y] > 155:
                pixels[x, y] = 255
            elif pixels[x, y] < 25:
                pixels[x, y] = 0

    img_convert_ndarray = 255 - np.asarray(im)
    im = Image.fromarray(img_convert_ndarray)
    #im.show()

    return img_convert_ndarray

# size = 28, 28
#
# im = Image.open("origin.jpg")
# im = im.convert("L")
# im.thumbnail(size, Image.ANTIALIAS)
#
# img_convert_ndarray = 255-np.array(im)
# im = Image.fromarray(img_convert_ndarray)
# im.show()
# print img_convert_ndarray