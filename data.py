import numpy as np
import jax.numpy as jnp
import random


def get_data(paintings, photos):
    while True:
        try:
            painting_file = random.sample(paintings, 1)[0]
            photo_file = random.sample(photos, 1)[0]

            if len(painting_file.shape) == 4 and painting_file.shape[0] == 1 and \
                    painting_file.shape[2] == painting_file.shape[1] and painting_file.shape[3] == 3 and \
                    len(photo_file.shape) == 4 and photo_file.shape[0] == 1 and \
                    photo_file.shape[2] == photo_file.shape[1] and photo_file.shape[3] == 3:
                break
        except:
            pass

    return jnp.array(np.load(painting_file)), jnp.array(np.load(photo_file))
