import jax
import jax.numpy as jnp
import jax.random as rnd
import haiku as hk
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from glob import glob

from training_functions import generator, discriminator, adam, ModelState, train_step_jit
from data import get_data


CHECKPOINT_DIR = "checkpoints/"
SAMPLES_DIR = "samples/"
CHECKPOINT_FILE = ""
TRAINING_LENGTH = 1000
PAINTING_FILES = glob("data/numpy/paintings/*")
PHOTO_FILES = glob("data/numpy/photos/*")

LOGGING_INTERVAL = 100
CHECKPOINT_INTERVAL = 1000
SAMPLING_INTERVAL = 200
MAX_ITERATIONS = 3000


generator_jit = jax.jit(generator.apply)  # JIT the apply function for generating images.
rng = hk.PRNGSequence(0)
testing_image = np.load("data/numpy/photos/flickr_wild_000002.npy")

random_input = rnd.uniform(next(rng), (1, 512, 512, 3), minval=-1, maxval=1)

if not Path(CHECKPOINT_FILE).is_file():
    gpa = generator.init(next(rng), random_input)
    gpb = generator.init(next(rng), random_input)
    dpa = discriminator.init(next(rng), random_input)
    dpb = discriminator.init(next(rng), random_input)

    goa = adam.init(gpa)
    gob = adam.init(gpb)
    doa = adam.init(dpa)
    dob = adam.init(dpb)

    iteration = 0

    print("Initialized Models.")

else:
    models_state = pickle.load(open(CHECKPOINT_FILE, "rb"))
    gpa = models_state["gpa"]
    gpb = models_state["gpb"]
    dpa = models_state["dpa"]
    dpb = models_state["dpb"]

    goa = models_state["goa"]
    gob = models_state["gob"]
    doa = models_state["doa"]
    dob = models_state["dob"]

    iteration = models_state["iteration"]

ga = ModelState(gpa, goa)
gb = ModelState(gpb, gob)
da = ModelState(dpa, doa)
db = ModelState(dpb, dob)


def save_model(ga, gb, da, db, iteration):
    models_state = {
        "gpa": ga.params,
        "goa": ga.opt_state,

        "gpb": gb.params,
        "gob": gb.opt_state,

        "dpa": da.params,
        "doa": da.opt_state,

        "dpb": db.params,
        "dob": db.opt_state,

        "iteration": iteration
    }
    pickle.dump(models_state, open(f"{CHECKPOINT_DIR}{iteration}.pickle", "wb"))


def plot_image(image, i):
    image = image.squeeze(0)
    image = ((image + 1) * 127.5).astype(jnp.uint8)
    plt.imsave(f"{SAMPLES_DIR}{i}.png", image)


def generate_sample_image(params, rng, testing_image, iteration):
    generated_image = generator_jit(params, rng, testing_image)
    plot_image(generated_image, iteration)


while iteration < MAX_ITERATIONS:
    painting, photo = get_data(PAINTING_FILES, PHOTO_FILES)

    lda, ldb, da, db, lga, lgb, ga, gb = train_step_jit(ga, gb, da, db,
                                                        next(rng), painting, photo)

    if iteration % LOGGING_INTERVAL == 0:
        lda = float(lda)
        ldb = float(ldb)
        lga = float(lga)
        lgb = float(lgb)

        info_to_write = f"Iteration {iteration}: LDA: {lda}, LDB: {ldb}, LGA: {lga}, LGB: {lgb}"
        print(info_to_write)

    if iteration % SAMPLING_INTERVAL == 0:
        generate_sample_image(ga.params, next(rng), testing_image, iteration)

    if iteration % CHECKPOINT_INTERVAL == 0 and iteration != 0:
        save_model(ga, gb, da, db, iteration)

    iteration += 1

