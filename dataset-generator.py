import itertools
import pathlib as plib
import string

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import numpy as np
import cv2 as cv


def mkDatabase(fonts, glyphs,
               rotations=(-15, -10, -5, 0, 5, 10, 15),
               scales=(1.0, 0.9, 0.8)):
    for (n_font, font) in enumerate(fonts):
        db_font_dir = plib.Path('dataset') / (str(n_font).zfill(2))
        db_font_dir.mkdir(parents=True, exist_ok=True)
        mkGlyphs(glyphs, font, db_font_dir, rotations=rotations, scales=scales)


def mkGlyphs(glyphs, font, db_font_dir,
             rotations=(0,),
             scales=(1.0,)):
    IMG_SHAPE = (32, 32)
    BGCOLOR = 255
    FGCOLOR = 0
    for glyph in glyphs:
        glyph_dir = db_font_dir / (glyph + '-' + hex(ord(glyph)))
        glyph_dir.mkdir(parents=True, exist_ok=True)

        for rot, scale in itertools.product(rotations, scales):
            glyph_img = Image.new('L', IMG_SHAPE, BGCOLOR)
            glyph_draw = ImageDraw.Draw(glyph_img)

            (wd, ht) = font.getbbox(glyph)[-2:]
            xy = ((IMG_SHAPE[0] - wd) / 2, (IMG_SHAPE[1] - ht) / 2)
            glyph_draw.text(xy, glyph, FGCOLOR, font=font)

            img = np.array(glyph_img)
            center = (img.shape[0] / 2, img.shape[1] / 2)
            rotMatrix = cv.getRotationMatrix2D(center, rot, scale)
            new_img = cv.warpAffine(img, rotMatrix, IMG_SHAPE,
                                    borderValue=(255, 255, 255),

                                    borderMode=cv.BORDER_CONSTANT)

            img_name = (glyph + '-' + hex(ord(glyph)) +
                        '_rot=' + str(rot) +
                        '_scale=' + str(scale) +
                        '.png')

            cv.imwrite(str(glyph_dir / img_name), new_img)


if __name__ == '__main__':
    FONT_SIZE_PT = 18

    font_path = plib.Path('font-resources')
    all_fonts = (ImageFont.truetype(str(f), FONT_SIZE_PT)
                 for f in font_path.glob('*.ttf'))
    all_glyphs = string.ascii_letters
    mkDatabase(all_fonts, all_glyphs)
