import jax
import jax.numpy as jnp
import jax.random as rnd
import haiku as hk
import optax
from typing import NamedTuple

from models import generator_fn, discriminator_fn


LEARNING_RATE = 1e-4


class ModelState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState


def cycle_loss(g1params: hk.Params, g2params: hk.Params,
               input_image: jnp.ndarray, rng1: rnd.PRNGKey, rng2: rnd.PRNGKey):

    converted_image = generator.apply(g1params, rng1, input_image)
    reconverted_image = generator.apply(g2params, rng2, converted_image)

    l1_loss = jnp.abs(reconverted_image - converted_image)
    return l1_loss.sum(1).mean()


def image_loss(gparams: hk.Params, dparams: hk.Params, input_image: jnp.ndarray,
               rng1: rnd.PRNGKey, rng2: rnd.PRNGKey):

    generated_image = generator.apply(gparams, rng1, input_image)
    labels = discriminator.apply(dparams, rng2, generated_image)

    correct_labels = jnp.ones_like(labels)

    loss = optax.sigmoid_binary_cross_entropy(labels, correct_labels)
    return loss.sum(1).mean()


def discrim_loss(dparams: hk.Params, image: jnp.ndarray, label: int,
                 rng: rnd.PRNGKey):
    predictions = discriminator.apply(dparams, rng, image)
    labels = jnp.zeros_like(predictions) + label
    loss = optax.sigmoid_binary_cross_entropy(predictions, labels)
    return loss.sum(1).mean() * 0.5


grad_cycle_loss = jax.value_and_grad(cycle_loss)
grad_image_loss = jax.value_and_grad(image_loss)
grad_discrim_loss = jax.value_and_grad(discrim_loss)


def train_step(ga: ModelState, gb: ModelState, da: ModelState, db: ModelState,
               rng: rnd.PRNGKey, painting_sample, photo_sample):
    """
    Perform one train step of the model.
    * Models:
        * A: Photo to Painting
        * B: Painting to Photo
    * Training Iteration:
        * Train discriminators
            * discriminator A:
                1. Generate fake image.
                2. Calculate Loss and gradients.
                3. Perform Backpropogation.
                4. Update weights.
            * Same for discriminator B:
        * Train Generators:
            * For each model perform discriminator loss.
            * Finally, perform cycle consistency loss for each model.
    * Complexity:
        * 6 generator runs.
        * 4 discriminator runs.
    """

    # Train the discriminators.

    # Discriminator A (Fake Images)
    rng, key1, key2 = rnd.split(rng, 3)
    generated_painting = generator.apply(ga.params, key1, photo_sample)
    da_loss_fake, grads = grad_discrim_loss(
        da.params, generated_painting, 0, key2)
    updates, opt_state = adam.update(grads, da.opt_state, da.params)
    params = optax.apply_updates(da.params, updates)
    da = ModelState(params, opt_state)

    # Discriminator A (Real Images)
    rng, key1 = rnd.split(rng, 2)
    da_loss_real, grads = grad_discrim_loss(
        da.params, painting_sample, 1, key1)
    updates, opt_state = adam.update(grads, da.opt_state, da.params)
    params = optax.apply_updates(da.params, updates)
    da = ModelState(params, opt_state)

    da_loss = da_loss_fake + da_loss_real

    # Discriminator B (Fake Images)
    rng, key1, key2 = rnd.split(rng, 3)
    generated_photo = generator.apply(gb.params, key1, painting_sample)
    db_loss_fake, grads = grad_discrim_loss(
        db.params, generated_photo, 0, key2)
    updates, opt_state = adam.update(grads, db.opt_state, db.params)
    params = optax.apply_updates(db.params, updates)
    db = ModelState(params, opt_state)

    # Discriminator B (Real Images)
    rng, key1 = rnd.split(rng, 2)
    db_loss_real, grads = grad_discrim_loss(db.params, photo_sample, 1, key1)
    updates, opt_state = adam.update(grads, db.opt_state, db.params)
    params = optax.apply_updates(db.params, updates)
    db = ModelState(params, opt_state)

    db_loss = db_loss_fake + db_loss_real

    # Train the generators.

    # Generator A
    rng, key1, key2 = rnd.split(rng, 3)
    ga_loss_discrim, grads = grad_image_loss(
        ga.params, da.params, photo_sample, key1, key2)
    updates, opt_state = adam.update(grads, ga.opt_state, ga.params)
    params = optax.apply_updates(ga.params, updates)
    ga = ModelState(params, opt_state)

    # Generator B
    rng, key1, key2 = rnd.split(rng, 3)
    gb_loss_discrim, grads = grad_image_loss(
        gb.params, db.params, painting_sample, key1, key2)
    updates, opt_state = adam.update(grads, gb.opt_state, gb.params)
    params = optax.apply_updates(gb.params, updates)
    gb = ModelState(params, opt_state)

    # Cycle consistency

    # Generator A
    rng, key1, key2 = rnd.split(rng, 3)
    ga_loss_consistency, grads = grad_cycle_loss(
        ga.params, gb.params, photo_sample, key1, key2)
    updates, opt_state = adam.update(grads, ga.opt_state, ga.params)
    params = optax.apply_updates(ga.params, updates)
    ga = ModelState(params, opt_state)

    # Generator B
    rng, key1, key2 = rnd.split(rng, 3)
    gb_loss_consistency, grads = grad_cycle_loss(
        gb.params, ga.params, painting_sample, key1, key2)
    updates, opt_state = adam.update(grads, gb.opt_state, gb.params)
    params = optax.apply_updates(gb.params, updates)
    gb = ModelState(params, opt_state)

    ga_loss = ga_loss_discrim + ga_loss_consistency
    gb_loss = gb_loss_discrim + gb_loss_consistency

    return da_loss, db_loss, da, db, ga_loss, gb_loss, ga, gb

train_step_jit = jax.jit(train_step)


generator = hk.transform(generator_fn)
generator_jit = jax.jit(generator.apply)
discriminator = hk.transform(discriminator_fn)
adam = optax.adam(LEARNING_RATE)