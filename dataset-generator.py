
import os
import glob

import string

import PIL.ImageFont
import PIL.ImageDraw

# import cv2 as cv

FONT_SIZE_PT = 14

font_dir = os.path.join('font-resources')
font_fpaths = (PIL.ImageFont.truetype(fname, FONT_SIZE_PT)
               for fname in glob.glob(os.path.join(font_dir, '*.ttf')))


