#!/usr/bin/env python3

import itertools
import math
import pathlib as path
import random
import string

from typing import List

import cv2 as cv
import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def mkDataset(ds_root: path.Path, fonts, glyphs,
              img_shape=(64, 64), rotations=(0.0,), scales=(1.0,)):
    ds_root = ds_root / 'train'
    ds_root.mkdir(parents=True, exist_ok=True)

    for glyph in glyphs:
        glyph_dir = ds_root / (glyph + '-U+' + hex(ord(glyph))[2:])
        glyph_dir.mkdir(parents=True, exist_ok=True)

        for font_ix, font in enumerate(fonts):
            out_dir = glyph_dir / (str(font_ix).zfill(2))
            out_dir.mkdir(parents=True, exist_ok=True)
            genGlyphImgs(out_dir, font, font_ix, glyph, img_shape, rotations, scales)


def genGlyphImgs(target_dir, font, font_ix, glyph, img_shape, rotations, scales):

    BGCOLOR = 255
    FGCOLOR = 0

    for rot, scale in itertools.product(rotations, scales):
        glyph_img = Image.new('L', img_shape, BGCOLOR)
        glyph_draw = ImageDraw.Draw(glyph_img)

        xy = (img_shape[0]/2, img_shape[1]/2)
        glyph_draw.text(xy, glyph, FGCOLOR, font=font, anchor='mm')

        img = np.array(glyph_img)
        center = (img.shape[0] / 2, img.shape[1] / 2)
        rotMatrix = cv.getRotationMatrix2D(center, rot, scale)
        new_img = cv.warpAffine(img, rotMatrix, img_shape,
                                borderValue=(255, 255, 255),
                                borderMode=cv.BORDER_CONSTANT)

        img_name = (glyph + '-' +
                    '_font=' + str(font_ix) +
                    '_code=' + hex(ord(glyph))[2:] +
                    '_rot=' + str(rot) +
                    '_scale=' + str(scale) +
                    '.png')

        cv.imwrite(str(target_dir / img_name), new_img)


def splitTrainTest(ds_root: path.Path, test_ratio=0.2):
    input_dir = ds_root / 'train'
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    def selectKfrom(seq: list, k: int):
        selected = []
        for _ in range(k):
            selection = random.choice(seq)
            selected.append(selection)
            seq.remove(selection)
        return selected

    test_dir = ds_root / 'test'
    test_dir.mkdir(parents=True, exist_ok=True)

    for glyph_dir in input_dir.iterdir():
        for font_dir in glyph_dir.iterdir():
            images = list(font_dir.iterdir())
            n_images = len(images)

            test_imgs = selectKfrom(images, math.floor(n_images * test_ratio))
            moveImgs(glyph_dir, font_dir, test_imgs, test_dir)


def moveImgs(glyph_dir: path.Path,
             font_dir: path.Path,
             images: List[path.Path],
             test_dir: path.Path):
    for img in images:
        new_path = test_dir / glyph_dir.name / font_dir.name
        new_path.mkdir(parents=True, exist_ok=True)
        img.rename(new_path / img.name)


def getFontPaths(resource_file: path.Path):
    return sorted(resource_file.glob('*.ttf'), key=lambda p: p.name[:2])


def main(img_shape=(64, 64)):

    FONT_SIZE_PX = int(np.floor(img_shape[0] * 0.8))
    ROTATIONS = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
    SCALES = (1.0, 0.9, 0.8)

    font_path = path.Path('font-resources')
    if not font_path.exists():
        raise FileNotFoundError(font_path)

    ds_path = path.Path('dataset')
    ds_path.mkdir(parents=True, exist_ok=True)

    all_fonts = tuple(ImageFont.truetype(str(f), FONT_SIZE_PX)
                      for f in getFontPaths(font_path))
    all_glyphs = string.ascii_letters + string.digits

    mkDataset(ds_path, all_fonts, all_glyphs, img_shape, rotations=ROTATIONS, scales=SCALES)
    splitTrainTest(ds_path, 0.2)


if __name__ == '__main__':
    main()
