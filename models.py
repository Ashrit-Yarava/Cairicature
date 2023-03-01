import jax
import jax.numpy as jnp
import haiku as hk


def downsample(x, filters, size, instance_norm=True):
    x = hk.Conv2D(filters, size, stride=2, padding="SAME", with_bias=False)(x)
    if instance_norm:
        x = hk.InstanceNorm(True, True)(x)
    x = jax.nn.leaky_relu(x, 0.3)
    return x


def upsample(x, filters, size, rng_dropout, dropout=False):
    x = hk.Conv2DTranspose(filters, size, stride=2,
                           padding="SAME", with_bias=False)(x)
    x = hk.InstanceNorm(True, True)(x)
    if dropout:
        x = hk.dropout(rng_dropout, 0.5, x)
    x = jax.nn.relu(x)
    return x


def residual(x, hidden_size):
    x_shape = x.shape[0]
    x_ = hk.Conv2DTranspose(hidden_size, 4, stride=2, padding='SAME', with_bias=False)(x)
    x_ = jax.nn.leaky_relu(x, 0.3)
    x_ = hk.Conv2D(x_shape, 4, stride=2, padding='SAME', with_bias=False)(x)
    x_ = hk.InstanceNorm(True, True)(x)
    x_ = jax.nn.leaky_relu(x, 0.3)
    return x + x_


def generator_fn(x):
    skips = []
    x = downsample(x, 128, 4, instance_norm=False)  # First
    skips.append(x)
    x = downsample(x, 256, 4)  # Second
    skips.append(x)
    x = downsample(x, 512, 4)  # Third
    skips.append(x)
    x = downsample(x, 512, 4)  # Fourth
    skips.append(x)
    x = downsample(x, 512, 4)  # Fifth
    skips.append(x)
    x = downsample(x, 512, 4)  # Sixth
    skips.append(x)
    x = downsample(x, 512, 4)  # Seventh

    x = residual(x, 768)
    x = residual(x, 768)
    x = residual(x, 768)

    skips = list(reversed(skips))

    x = upsample(x, 512, 4, hk.next_rng_key(), dropout=True)
    x = jnp.concatenate((x, skips[0]), axis=-1)
    x = upsample(x, 512, 4, hk.next_rng_key(), dropout=True)
    x = jnp.concatenate((x, skips[1]), axis=-1)
    x = upsample(x, 512, 4, hk.next_rng_key(), dropout=False)
    x = jnp.concatenate((x, skips[2]), axis=-1)
    x = upsample(x, 256, 4, hk.next_rng_key(), dropout=False)
    x = jnp.concatenate((x, skips[3]), axis=-1)
    x = upsample(x, 128, 4, hk.next_rng_key(), dropout=False)
    x = jnp.concatenate((x, skips[4]), axis=-1)
    x = upsample(x, 64, 4, hk.next_rng_key(), dropout=False)
    x = jnp.concatenate((x, skips[5]), axis=-1)
    x = hk.Conv2DTranspose(3, 4, stride=2, padding="SAME")(x)
    x = jax.nn.tanh(x)
    return x


def discriminator_fn(x):
    x = downsample(x, 128, 4, instance_norm=False)
    x = downsample(x, 256, 4)

    x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))

    x = hk.Conv2D(512, 4, stride=1, with_bias=False, padding="VALID")(x)
    x = hk.InstanceNorm(True, True)(x)
    x = jax.nn.leaky_relu(x, 0.3)

    x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))

    x = hk.Conv2D(1, 4, stride=1, padding="VALID")(x)

    return x
