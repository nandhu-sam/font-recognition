import itertools
import pathlib as plib
import string

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import numpy as np
import cv2 as cv


def mkDataset(ds_root: plib.Path, fonts, glyphs,
              rotations=(0.0,),
              scales=(1.0,)):
    glyph_ds_dir = ds_root / 'glyphs'
    # fonts_ds_dir = ds_root/'fonts'

    glyph_ds_dir.mkdir(parents=True, exist_ok=True)
    # fonts_ds_dir.mkdir(parents=True, exist_ok=True)

    mkGlyphDS(glyph_ds_dir, fonts, glyphs, rotations, scales)
    # mkFontDS(fonts_ds_dir, fonts, glyphs, rotations, scales)


def mkGlyphDS(ds_root: plib.Path, fonts, glyphs, rotations=(0.0,), scales=(1.0,)):
    for glyph in glyphs:
        glyph_dir = ds_root / (glyph + '-U+' + hex(ord(glyph))[2:])
        glyph_dir.mkdir(parents=True, exist_ok=True)

        for font_ix, font in enumerate(fonts):
            out_dir = glyph_dir / (str(font_ix).zfill(2))
            out_dir.mkdir(parents=True, exist_ok=True)
            genGlyphImgs(out_dir, font, font_ix, glyph, rotations, scales)


def mkFontDS(ds_root: plib.Path, fonts, glyphs,
             rotations=(0.0,),
             scales=(1.0,)):
    for font_ix, font in enumerate(fonts):
        font_dir = ds_root / (str(font_ix).zfill(2))
        font_dir.mkdir(parents=True, exist_ok=True)

        for glyph in glyphs:
            out_dir = font_dir / (glyph + '-U+' + hex(glyph)[2:])
            out_dir.mkdir(parents=True, exist_ok=True)
            genGlyphImgs(out_dir, font, font_ix, glyph, rotations, scales)


def genGlyphImgs(target_dir, font, font_ix, glyph, rotations, scales):
    IMG_SHAPE = (32, 32)
    BGCOLOR = 255
    FGCOLOR = 0

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

        img_name = (glyph + '-' +
                    '_font=' + str(font_ix) +
                    '_code=' + hex(ord(glyph))[2:] +
                    '_rot=' + str(rot) +
                    '_scale=' + str(scale) +
                    '.png')

        cv.imwrite(str(target_dir / img_name), new_img)


def getFontPaths(resource_file: plib.Path):
    return sorted(resource_file.glob('*.ttf'), key=lambda p: p.name[:2])


if __name__ == '__main__':
    FONT_SIZE_PT = 18
    ROTATIONS = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
    SCALES = (1.0, 0.9, 0.8)

    font_path = plib.Path('font-resources')
    ds_path = plib.Path('dataset')

    font_path.mkdir(parents=True, exist_ok=True)
    ds_path.mkdir(parents=True, exist_ok=True)
    all_fonts = tuple(ImageFont.truetype(str(f), FONT_SIZE_PT)
                      for f in getFontPaths(font_path))
    all_glyphs = string.ascii_letters + string.digits

    mkDataset(ds_path, all_fonts, all_glyphs,
              rotations=ROTATIONS,
              scales=SCALES)
