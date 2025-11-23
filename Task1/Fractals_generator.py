import numpy as np
import pandas as pd
import datashader as ds
from datashader import transfer_functions as tf
from datashader.colors import inferno, viridis
from math import sin, cos, sqrt, fabs
from colorcet import fire
from PIL import Image
import random
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# Configuration
N_POINTS = 100000
IMG_SIZE = 64
DATASET_SIZE = 20000


def trajectory(fn, x0, y0, a, b=0, c=0, d=0, e=0, f=0, n=N_POINTS):
    """Computes the trajectories of the given attractor function."""
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    for i in range(n - 1):
        x[i + 1], y[i + 1] = fn(x[i], y[i], a, b, c, d, e, f)
    return pd.DataFrame({'x': x, 'y': y})


# Attractor functions
def fractal_dream(x, y, a, b, c, d, *o):
    return sin(y * b) + c * sin(x * b), sin(x * a) + d * sin(y * a)


def hopalong(x, y, a, b, c, *o):
    return y - sqrt(fabs(b * x - c)) * np.sign(x), a - x


def de_jong(x, y, a, b, c, d, *o):
    return sin(a * y) - cos(b * x), sin(c * x) - cos(d * y)


def clifford(x, y, a, b, c, d, *o):
    return sin(a * y) + c * cos(a * x), sin(b * x) + d * cos(b * y)


def is_image_valid(agg, min_density=0.15):
    """Checks whether the image contains sufficient structural density."""
    non_zero = np.count_nonzero(agg.values)
    total = agg.values.size
    return (non_zero / total) >= min_density


def generate_quality_fractal_image(width=IMG_SIZE, height=IMG_SIZE, max_attempts=10):
    """Generates fractal images discarding those with insufficient visual density."""
    attempts = 0
    attractors = [fractal_dream, hopalong, de_jong, clifford]

    while attempts < max_attempts:
        fn = random.choice(attractors)

        if fn == fractal_dream:
            params = [random.uniform(-1, 1), random.uniform(-1, 1)] + [random.uniform(-2, 2) for _ in range(4)]
        elif fn == hopalong:
            params = [random.uniform(-1, 1), random.uniform(-1, 1)] + [random.uniform(-2, 2) for _ in range(3)]
        else:
            params = [random.uniform(-1, 1), random.uniform(-1, 1)] + [random.uniform(-2, 2) for _ in range(4)]

        try:
            df = trajectory(fn, *params, n=N_POINTS)
            cvs = ds.Canvas(plot_width=width, plot_height=height)
            agg = cvs.points(df, 'x', 'y')

            if is_image_valid(agg):
                cmap = random.choice([inferno, viridis, fire])
                img = tf.shade(agg, cmap=cmap, alpha=255)
                pil_img = tf.Image.to_pil(img)

                if pil_img.mode == 'RGBA':
                    rgb_img = Image.new("RGB", pil_img.size, (0, 0, 0))
                    rgb_img.paste(pil_img, mask=pil_img.split()[3])
                else:
                    rgb_img = pil_img.convert("RGB")

                return rgb_img, True

            attempts += 1

        except Exception:
            attempts += 1
            continue

    print("Fallback engaged after unsuccessful attempts.")
    return generate_fractal_image_fallback(width, height), False


def generate_fractal_image_fallback(width=IMG_SIZE, height=IMG_SIZE):
    """Generates a fallback fractal image using reliable parameter sets."""
    good_params = [
        [0.1, 0.1, 1.8, -1.2, 0.5, -0.8],
        [0.5, -0.5, -1.7, 1.3],
        [0.3, 0.3, 2.0, -2.0, 1.5, -1.5],
    ]
    fn = fractal_dream
    params = random.choice(good_params)

    df = trajectory(fn, *params, n=N_POINTS)
    cvs = ds.C
