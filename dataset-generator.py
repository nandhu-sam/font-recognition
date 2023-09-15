
import os
import glob

import PIL.ImageFont

# import cv2 as cv

FONT_SIZE_PT

font_dir = os.path.join('font-resources')
font_fpaths = [PIL.ImageFont.truetype(fname, FONT_SIZE_PT)
               for fname in glob.glob(os.path.join(font_dir, '*.ttf'))]
