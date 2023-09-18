#!/usr/bin/python3

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


def mkDataset(ds_root: path.Path, fonts, glyphs, rotations=(0.0,), scales=(1.0,)):
    ds_root = ds_root / 'train'
    ds_root.mkdir(parents=True, exist_ok=True)

    for glyph in glyphs:
        glyph_dir = ds_root / (glyph + '-U+' + hex(ord(glyph))[2:])
        glyph_dir.mkdir(parents=True, exist_ok=True)

        for font_ix, font in enumerate(fonts):
            out_dir = glyph_dir / (str(font_ix).zfill(2))
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


def splitTrainTest(ds_root: path.Path, test_ratio=0.2):
    input_dir = ds_root / 'train'
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    test_dir = ds_root / 'test'
    test_dir.mkdir(parents=True, exist_ok=True)

    for glyph_dir in input_dir.iterdir():
        for font_dir in glyph_dir.iterdir():
            images = list(font_dir.iterdir())
            n_images = len(images)

            test_imgs = selectKfrom(images, math.floor(n_images * test_ratio))
            moveImgs(glyph_dir, font_dir, test_imgs, test_dir)


def selectKfrom(seq: list, k: int):
    selected = []
    for _ in range(k):
        selection = random.choice(seq)
        selected.append(selection)
        seq.remove(selection)
    return selected


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


if __name__ == '__main__':
    FONT_SIZE_PT = 18
    ROTATIONS = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
    SCALES = (1.0, 0.9, 0.8)

    font_path = path.Path('font-resources')
    ds_path = path.Path('dataset')

    font_path.mkdir(parents=True, exist_ok=True)
    ds_path.mkdir(parents=True, exist_ok=True)
    all_fonts = tuple(ImageFont.truetype(str(f), FONT_SIZE_PT)
                      for f in getFontPaths(font_path))
    all_glyphs = string.ascii_letters + string.digits

    mkDataset(ds_path, all_fonts, all_glyphs,
              rotations=ROTATIONS,
              scales=SCALES)
    splitTrainTest(ds_path, 0.2)
