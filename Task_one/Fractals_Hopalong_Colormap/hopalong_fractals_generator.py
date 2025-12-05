import numpy as np
import pandas as pd
from math import sin, cos, sqrt, fabs
import random
from PIL import Image
import datashader as ds
from datashader import transfer_functions as tf
from datashader.colors import inferno, viridis
from colorcet import fire
from tqdm.notebook import tqdm

# =====================
#     CONFIGURAZIONE
# =====================
IMG_SIZE = 64
N_POINTS = 100000
TOTAL_IMAGES = 18000  # Dimensione dataset finale

# =====================
#   ATTRATTORI
# =====================
def trajectory(fn, x0, y0, a, b=0, c=0, d=0, e=0, f=0, n=N_POINTS):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    for i in range(n - 1):
        x[i+1], y[i+1] = fn(x[i], y[i], a, b, c, d, e, f)
    return pd.DataFrame({'x': x, 'y': y})

def hopalong(x, y, a, b, c, *o):
    return y - sqrt(fabs(b * x - c)) * np.sign(x), a - x

ATTRACTORS = [hopalong]  # solo hopalong

# =====================
#   VALIDAZIONE IMMAGINE
# =====================
def is_image_valid(agg, min_density=0.15):
    non_zero = np.count_nonzero(agg.values)
    total = agg.values.size
    return (non_zero / total) >= min_density

# =====================
#  GENERAZIONE IMMAGINE
# =====================
def generate_fractal_by_type(fn, width=IMG_SIZE, height=IMG_SIZE, max_attempts=10, current_cmap=None):
    attempts = 0
    cmap_list = [inferno, viridis, fire]

    while attempts < max_attempts:
        params = [
            random.uniform(-1, 1), random.uniform(-1, 1),
            random.uniform(-2, 2), random.uniform(-2, 2),
            random.uniform(-2, 2)
        ]

        try:
            df = trajectory(fn, *params, n=N_POINTS)
            cvs = ds.Canvas(plot_width=width, plot_height=height)
            agg = cvs.points(df, 'x', 'y')

            if is_image_valid(agg):
                cmap = cmap_list[current_cmap]
                cmap_label = current_cmap

                img = tf.shade(agg, cmap=cmap, alpha=255)
                pil_img = tf.Image.to_pil(img)

                return pil_img.convert("RGB"), (params, cmap_label)

            attempts += 1

        except Exception:
            attempts += 1

    return None, None

def generate_fractal_fallback(fn, used_signatures, current_cmap):
    cmap_list = [inferno, viridis, fire]
    fallback_params = [
        [0.1, 0.1, 1.8],
        [0.2, -0.3, 1.5],
        [0.5, 0.7, -1.2],
    ]

    for params in fallback_params:
        df = trajectory(fn, *params, n=N_POINTS)
        cvs = ds.Canvas(plot_width=IMG_SIZE, plot_height=IMG_SIZE)
        agg = cvs.points(df, 'x', 'y')

        cmap = cmap_list[current_cmap]
        cmap_label = current_cmap

        img = tf.shade(agg, cmap=cmap, alpha=255)
        pil_img = tf.Image.to_pil(img).convert("RGB")

        sig = hash(pil_img.tobytes())
        if sig not in used_signatures:
            used_signatures.add(sig)
            return pil_img, (params, cmap_label)

    return None, None

# =====================
#   CREAZIONE DATASET
# =====================
from tqdm.auto import tqdm
import time

def create_balanced_fractal_dataset(total_images=TOTAL_IMAGES):
    per_palette = total_images // 3  # divisione perfetta

    dataset = []
    labels = []
    used_signatures = set()
    stats = {0: 0, 1: 0, 2: 0}
    rejected_counts = {0: 0, 1: 0, 2: 0}

    print(f"\nGenerazione dataset hopalong bilanciato: {per_palette} immagini per palette")

    total_progress = tqdm(
        total=per_palette * 3,
        desc="Generazione dataset",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    start_time = time.time()
    fn = hopalong

    for current_cmap in [0, 1, 2]:
        count = 0
        while count < per_palette:
            img, out = generate_fractal_by_type(fn, current_cmap=current_cmap)

            if img is None:
                img, out = generate_fractal_fallback(fn, used_signatures, current_cmap)
                rejected_counts[current_cmap] += 1
                if img is None:
                    continue

            params, cmap_label = out
            sig = hash(img.tobytes())

            if sig not in used_signatures:
                used_signatures.add(sig)
                dataset.append(np.array(img) / 255.0)
                labels.append(cmap_label)
                stats[cmap_label] += 1
                count += 1
                total_progress.update(1)

    total_progress.close()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTempo totale: {elapsed // 60:.0f} minuti {elapsed % 60:.1f} secondi")

    dataset = np.array(dataset)
    labels = np.array(labels)

    np.save("hopalong_images.npy", dataset)
    np.save("hopalong_labels.npy", labels)

    print("\nDataset generato.")
    print(f"Shape immagini: {dataset.shape}")
    print(f"Shape labels:   {labels.shape}")

    print("\nDistribuzione immagini per palette:")
    print(f"inferno (0): {stats[0]}")
    print(f"viridis (1): {stats[1]}")
    print(f"fire    (2): {stats[2]}")

    print("\nFallback per palette:")
    print(f"inferno (0): {rejected_counts[0]}")
    print(f"viridis (1): {rejected_counts[1]}")
    print(f"fire    (2): {rejected_counts[2]}")

    return dataset, labels

# =====================
#   ESECUZIONE
# =====================
if __name__ == "__main__":
    create_balanced_fractal_dataset()
