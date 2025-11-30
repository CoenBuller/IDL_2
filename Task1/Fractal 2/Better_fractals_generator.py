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
IMG_SIZE = 1920
N_POINTS = 100000
TOTAL_IMAGES = 4  # <- modifica qui la dimensione del dataset finale


# =====================
#   ATTRATTORI
# =====================
def trajectory(fn, x0, y0, a, b=0, c=0, d=0, e=0, f=0, n=N_POINTS):
    """Genera traiettoria dell'attrattore scelto."""
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    for i in range(n - 1):
        x[i+1], y[i+1] = fn(x[i], y[i], a, b, c, d, e, f)
    return pd.DataFrame({'x': x, 'y': y})


def fractal_dream(x, y, a, b, c, d, *o):
    return sin(y * b) + c * sin(x * b), sin(x * a) + d * sin(y * a)


def hopalong(x, y, a, b, c, *o):
    return y - sqrt(fabs(b * x - c)) * np.sign(x), a - x


def de_jong(x, y, a, b, c, d, *o):
    return sin(a * y) - cos(b * x), sin(c * x) - cos(d * y)


def clifford(x, y, a, b, c, d, *o):
    return sin(a * y) + c * cos(a * x), sin(b * x) + d * cos(b * y)


ATTRACTORS = [fractal_dream, hopalong, de_jong, clifford]


# =====================
#   VALIDAZIONE IMMAGINE
# =====================
def is_image_valid(agg, min_density=0.15):
    """Verifica che l'immagine abbia una densità minima di pixel significativi."""
    non_zero = np.count_nonzero(agg.values)
    total = agg.values.size
    return (non_zero / total) >= min_density


# =====================
#  GENERAZIONE IMMAGINE
# =====================
def generate_fractal_by_type(fn, width=IMG_SIZE, height=IMG_SIZE, max_attempts=10):
    """Genera un'immagine frattale da un singolo attrattore."""
    attempts = 0

    while attempts < max_attempts:
        if fn.__name__ == "fractal_dream":
            params = [random.uniform(-1, 1), random.uniform(-1, 1)] + [random.uniform(-2, 2) for _ in range(4)]
        elif fn.__name__ == "hopalong":
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
                return pil_img.convert("RGB"), params

            attempts += 1

        except Exception:
            attempts += 1

    return None, None


def generate_fractal_fallback(fn, used_signatures):
    """Genera un'immagine fallback evitando duplicati."""
    fallback_params = {
        fractal_dream: [[0.1, 0.1, 1.8, -1.2, 0.5, -0.8]],
        clifford: [[0.5, -0.5, -1.7, 1.3]],
        de_jong: [[0.3, 0.3, 2.0, -2.0, 1.5, -1.5]]
    }

    for params in fallback_params.get(fn, []):
        df = trajectory(fn, *params, n=N_POINTS)
        cvs = ds.Canvas(plot_width=IMG_SIZE, plot_height=IMG_SIZE)
        agg = cvs.points(df, 'x', 'y')
        cmap = random.choice([inferno, viridis, fire])
        img = tf.shade(agg, cmap=cmap, alpha=255)
        pil_img = tf.Image.to_pil(img).convert("RGB")

        sig = hash(pil_img.tobytes())
        if sig not in used_signatures:
            used_signatures.add(sig)
            return pil_img, params

    return None, None


# =====================
#   CREAZIONE DATASET
# =====================
from tqdm.auto import tqdm
import time

def create_balanced_fractal_dataset(total_images=TOTAL_IMAGES):
    per_type = total_images // len(ATTRACTORS)

    dataset = []
    labels = []
    used_signatures = set()
    stats = {fn.__name__: 0 for fn in ATTRACTORS}
    rejected_counts = {fn.__name__: 0 for fn in ATTRACTORS}

    print(f"\nGenerazione dataset bilanciato: {per_type} immagini per ciascun attrattore")

    total_progress = tqdm(
        total=total_images,
        desc="Generazione dataset",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    start_time = time.time()

    for label, fn in enumerate(ATTRACTORS):
        count = 0
        while count < per_type:
            img, params = generate_fractal_by_type(fn)

            if img is None:
                img, params = generate_fractal_fallback(fn, used_signatures)
                rejected_counts[fn.__name__] += 1
                if img is None:
                    continue

            sig = hash(img.tobytes())
            if sig not in used_signatures:
                used_signatures.add(sig)
                dataset.append(np.array(img) / 255.0)
                labels.append(label)
                stats[fn.__name__] += 1
                count += 1
                total_progress.update(1)

    total_progress.close()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTempo totale: {elapsed // 60:.0f} minuti {elapsed % 60:.1f} secondi")

    dataset = np.array(dataset)
    labels = np.array(labels)

    np.save("fractal_images1.png", dataset)
    np.save("fractal_labels1.npy", labels)

    print("\nDataset generato correttamente.")
    print(f"Shape immagini: {dataset.shape}")
    print(f"Shape labels:   {labels.shape}")

    print("\nDistribuzione immagini per attrattore:")
    for fn, count in stats.items():
        print(f"{fn}: {count}")

    print("\nImmagini scartate per attrattore:")
    for fn, r in rejected_counts.items():
        print(f"{fn}: {r}")

    return dataset, labels

# =====================
#   ESECUZIONE
# =====================
if __name__ == "__main__":
    create_balanced_fractal_dataset()




