import itertools as it

import pathlib as plib

import string

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# import cv2 as cv

FONT_SIZE_PT: int = 14

IMG_SHAPE = (32, 32)
BGCOLOR = 255
FGCOLOR = 0

font_dir = plib.Path('font-resources')
fonts = (ImageFont.truetype(str(f), FONT_SIZE_PT)
         for f in font_dir.glob('*.ttf'))

glyphs = string.ascii_letters

for (n_font, font), glyph in it.product(enumerate(fonts), glyphs):
    glyph_img = Image.new('L', IMG_SHAPE, BGCOLOR)
    glyph_draw = ImageDraw.Draw(glyph_img)

    (wd, ht) = font.getbbox(glyph)[-2:]
    xy = ((IMG_SHAPE[0] - wd) / 2, (IMG_SHAPE[1] - ht) / 2)
    glyph_draw.text(xy, glyph, FGCOLOR, font=font)

    codepoint = hex(ord(glyph))
    n_font_str = str(n_font).zfill(2)
    save_path = plib.Path('dataset') / n_font_str
    save_path.mkdir(parents=True, exist_ok=True)
    glyph_img.save(save_path/(codepoint+'-'+glyph+'.png'))