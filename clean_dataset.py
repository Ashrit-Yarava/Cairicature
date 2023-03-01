import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from pathlib import Path
from skimage.transform import resize


SOURCE_FOLDER = "data/caricatures/"
TARGET_FOLDER = "data/numpy/paintings/"
DIMENSIONS = (512, 512)


files = glob(f"{SOURCE_FOLDER}*")

for image_file in tqdm(files):
    target_file = f"{TARGET_FOLDER}{Path(image_file).stem}.npy"

    img = plt.imread(image_file)
    img = (img / 127.5) - 1

    img = resize(img, DIMENSIONS)

    img = np.expand_dims(img, 0)

    np.save(target_file, img)