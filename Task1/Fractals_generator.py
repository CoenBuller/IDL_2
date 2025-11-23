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
import matplotlib as plt

# Configuration
N_POINTS = 100000
IMG_SIZE = 64
DATASET_SIZE = 100


# ========== Trajectory Function ==========
def trajectory(fn, x0, y0, a, b=0, c=0, d=0, e=0, f=0, n=N_POINTS):
    """Generate trajectory points for fractal attractors."""
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    for i in np.arange(n - 1):
        x[i + 1], y[i + 1] = fn(x[i], y[i], a, b, c, d, e, f)
    return pd.DataFrame(dict(x=x, y=y))


# ========== Fractal Functions ==========
def Fractal_Dream(x, y, a, b, c, d, *o):
    return sin(y * b) + c * sin(x * b), sin(x * a) + d * sin(y * a)


def Hopalong1(x, y, a, b, c, *o):
    return y - sqrt(fabs(b * x - c)) * np.sign(x), a - x


def De_Jong(x, y, a, b, c, d, *o):
    return sin(a * y) - cos(b * x), sin(c * x) - cos(d * y)


def Clifford(x, y, a, b, c, d, *o):
    return sin(a * y) + c * cos(a * x), sin(b * x) + d * cos(b * y)


# ========== Quality Check ==========
def is_image_valid(agg, min_density=0.15):
    """Check if the image has enough non-empty pixels."""
    non_zero_pixels = np.count_nonzero(agg.values)
    total_pixels = agg.values.size
    density = non_zero_pixels / total_pixels
    return density >= min_density


# ========== Fractal Image Generator ==========
def generate_quality_fractal_image(width=IMG_SIZE, height=IMG_SIZE, max_attempts=10):
    """Generate a fractal image with quality filter, return image, function name, parameters."""
    attempts = 0

    while attempts < max_attempts:
        functions = [Fractal_Dream, Hopalong1, De_Jong, Clifford]
        fn = random.choice(functions)

        # Parameter ranges
        if fn == Fractal_Dream:
            vals = [random.uniform(-1, 1), random.uniform(-1, 1)] + [random.uniform(-2, 2) for _ in range(4)]
        elif fn == Hopalong1:
            vals = [random.uniform(-1, 1) for _ in range(4)]
        else:  # De_Jong and Clifford
            vals = [random.uniform(-1, 1), random.uniform(-1, 1)] + [random.uniform(-2, 2) for _ in range(4)]

        try:
            df = trajectory(fn, *vals, n=N_POINTS)
            cvs = ds.Canvas(plot_width=width, plot_height=height)
            agg = cvs.points(df, 'x', 'y')

            if is_image_valid(agg):
                colormaps = [inferno, viridis, fire]
                cmap = random.choice(colormaps)

                img = tf.shade(agg, cmap=cmap, alpha=255)
                pil_img = tf.Image.to_pil(img)

                if pil_img.mode == 'RGBA':
                    rgb_img = Image.new("RGB", pil_img.size, (0, 0, 0))
                    rgb_img.paste(pil_img, mask=pil_img.split()[3])
                else:
                    rgb_img = pil_img.convert("RGB")

                return rgb_img, True, fn.__name__, vals

            attempts += 1

        except Exception:
            attempts += 1

    # Fallback: return a basic fractal image without metadata
    fallback_img = generate_fractal_image_fallback(width, height)
    return fallback_img, False, "Fallback", None


# ========== Fallback Image ==========
def generate_fractal_image_fallback(width=IMG_SIZE, height=IMG_SIZE):
    """Generate a fallback image with slight parameter variability to avoid duplicates."""
    fallback_params = [
        [0.1, 0.1, 1.8, -1.2, 0.5, -0.8],
        [0.5, -0.5, -1.7, 1.3, 0.4, -1.0],
        [0.3, 0.3, 2.0, -2.0, 1.5, -1.5],
    ]

    fn = Fractal_Dream
    base_params = random.choice(fallback_params)

    # Add slight random variation to avoid duplicates
    vals = [p + random.uniform(-0.2, 0.2) for p in base_params]

    df = trajectory(fn, *vals, n=N_POINTS)
    cvs = ds.Canvas(plot_width=width, plot_height=height)
    agg = cvs.points(df, 'x', 'y')

    cmap = random.choice([inferno, viridis, fire])
    img = tf.shade(agg, cmap=cmap, alpha=255)
    pil_img = tf.Image.to_pil(img)

    if pil_img.mode == 'RGBA':
        rgb_img = Image.new("RGB", pil_img.size, (0, 0, 0))
        rgb_img.paste(pil_img, mask=pil_img.split()[3])
    else:
        rgb_img = pil_img.convert("RGB")

    return rgb_img


# ========== Dataset Creation ==========
def create_quality_fractal_dataset(num_images=DATASET_SIZE):
    """Generate dataset and metadata, save .npy file and metadata CSV."""
    images_array = []
    metadata = []
    rejected_count = 0

    for i in tqdm(range(num_images), desc="Generating fractal images"):
        img, is_quality, fn_name, params = generate_quality_fractal_image()
        img_array = np.array(img) / 255.0

        if is_quality:
            images_array.append(img_array)
            density = float(np.mean(img_array > 0.1))

            metadata.append({
                "index": len(images_array) - 1,
                "type": fn_name,
                "params": params,
                "density": round(density, 4)
            })
        else:
            rejected_count += 1

    dataset = np.array(images_array)
    np.save("quality_fractal_dataset_64x64_rgb.npy", dataset)

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv("fractal_metadata.csv", index=False)

    print("Dataset saved as 'quality_fractal_dataset_64x64_rgb.npy'")
    print("Metadata saved as 'fractal_metadata.csv'")
    print(f"Total valid images: {len(images_array)}")
    print(f"Rejected images: {rejected_count}")

    return dataset


# ========== Test Generation ==========
if __name__ == "__main__":
    print("Testing fractal generation with quality control...")
    #dataset = create_quality_fractal_dataset(DATASET_SIZE)
    
    #print(f"Final dataset shape: {dataset.shape}")

    # Load dataset
    dataset = np.load("quality_fractal_dataset_64x64_rgb.npy")  # shape: (N, 64, 64, 3)

