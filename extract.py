#!/usr/bin/env python

from PIL import Image, ImageOps
from pathlib import Path
import sys
import numpy as np
import datetime
import argparse
import progressbar
from datetime import timezone
import matplotlib.pyplot as plt
from concurrent.futures import as_completed, ProcessPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument('rootdir', nargs='+')
parser.add_argument('-o', '--output', required=False, default='waterfall.npy')
parser.add_argument('-m', '--mask', required=False, default='mask.bmp')
parser.add_argument('-d', '--direction', required=False, default='horizontal', choices={'horizontal', 'vertical'})
args = parser.parse_args()

mask = Image.open(args.mask).convert('L')
SLICE_DIRECTION = args.direction

def get_timestamp(filename):
    parts = Path(filename).name.split('_')
    year, month, day = list(map(int, parts[:3]))
    hour, minute = list(map(int, parts[4:6]))
    seconds = int(parts[6].split('.')[0])

    return datetime.datetime(year, month, day, hour, minute, seconds, tzinfo=timezone.utc)

def fetch_image_slice(index, filename):
    i = Image.open(filename)
    masked = ImageOps.fit(i, mask.size, centering=(0.5, 0.5))
    masked.putalpha(mask)
    arr = np.asarray(masked)
    alpha = arr[:, :, -1]
    red = arr[:, :, :-1].mean(axis=-1)
    m = np.ma.masked_where(alpha == 0, red)

    if SLICE_DIRECTION == 'vertical':
        flat = np.average(m, axis=1)
    else:
        flat = np.average(m, axis=0)
    return index, flat[1:-1]


all_images = []
for path in args.rootdir:
    root_path = Path(path)
    all_images.extend(list(root_path.glob('**/*.jpg')))

images = sorted(list(set(all_images)), key=get_timestamp)
nimages = len(images)

print(f'Analysing {nimages} images', file=sys.stderr)

if SLICE_DIRECTION == 'vertical':
    npix = 1200 - 2
else:
    npix = 1600 - 2

out = np.zeros((npix, nimages), dtype=np.uint8)

with ProcessPoolExecutor() as pool:
    futures = []
    for i, image in enumerate(images):
        fut = pool.submit(fetch_image_slice, i, image)
        futures.append(fut)

    for fut in as_completed(futures):
        i, slice = fut.result()
        print(f'Slice {i} complete')
        assert len(slice) == npix
        assert npix == out.shape[0]
        out[:, i] = slice

np.save(args.output, out)
