import jax
import optax
from jax.experimental import jax2tf
import jax.numpy as jnp

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

path = '/theia/scratch/brussel/105/vsc10503/carbon monitoring/python/'

# Make circle mask
def make_circle_mask(radius, n, delta):

    x = (jnp.arange(n) - n/2) * delta
    y = (jnp.arange(n) - n/2) * delta
    x, y = jnp.meshgrid(x, y)

    def sigmoid(x):
      return 1 / (1 + jnp.exp(-x))

    circle_mask = sigmoid(radius**2 - (x**2 + y**2))

    return circle_mask

# Helpers: prediction and calculate diffraction

# Define function
def predict(x, model):

  pred = model(x.reshape(-1, 2)).reshape(2048, 2048, 2)

  return pred

def predict_phase(x, model):

  pred = model(x.reshape(-1, 2)).reshape(2048, 2048, 1)

  return pred

def calculate_diffraction_jax(metalens, lamb, Z,
                              n, upsampling, delta):

  # Define the size of the propagation function p(u,v)
  n_out = n * upsampling
  n1 = n_out/2
  n2 = n_out/2
  delta_out = delta / upsampling #um, pixel size

  # Define the angular spectrum coordinates
  u = jnp.arange(-n1, n1, 1) / (n_out * delta_out)
  v = jnp.arange(-n2, n2, 1) / (n_out * delta_out)
  fx, fy = jnp.meshgrid(u,v)

  # Calculate field with fourier
  metalens_upsampled = jnp.repeat(jnp.repeat(metalens, upsampling, axis=0), upsampling, axis=1)
  propagator = jnp.exp(2*jnp.pi*1j*Z*jnp.sqrt(jax.lax.complex((1/lamb)**2-fx**2-fy**2, 0.)))
  f = jnp.fft.fft2(metalens_upsampled)
  fshift = jnp.fft.fftshift(f)
  field_fourier = fshift*propagator

  #Calculate the inverse Fourier transform
  field = jnp.fft.ifft2(field_fourier)

  return jnp.abs(field)**2

# Helpers: compute efficiency and plot
def intensity_lobes(field, n_lobes, q, n, upsampling):

  x0_quadrants = [int(n/4), int((3*n)/4), int(n/4), int((3*n)/4)]
  y0_quadrants = [int(n/4), int(n/4), int((3*n)/4), int((3*n)/4)]
  x0 = x0_quadrants[q-1]
  y0 = y0_quadrants[q-1]
  pixel_radius = 70 # 7, 15, 20, 47

  x = jnp.arange(n)
  y = jnp.arange(n)
  x, y = jnp.meshgrid(x, y)

  def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

  focal_spot_mask = sigmoid(pixel_radius**2 - ((x - x0)**2 + (y - y0)**2))

  I_lobes = jnp.sum(focal_spot_mask * field) / (upsampling ** 2)

  return I_lobes

def get_efficiency(field, q, n, upsampling):

  # Compute incoming I
  incoming_I = 1917485

  # Compute focal spot
  I_1_lobe = intensity_lobes(field, n_lobes=1, q=q,
                             n=n, upsampling=upsampling)

  # Compute efficiency
  efficiency = I_1_lobe / incoming_I

  return efficiency

# Run optimization


def FOM(x, models, params):

  # Circle mask
  circle_mask = make_circle_mask(radius, n, delta)

  loss = 0.
  effs = []
  total_Is = []

  for i, lamb in enumerate(params['lamb']):

    """if lamb == 0.76:

      # Predict phi
      pred = predict_phase(x, models[f'model_{i+1}'])

      # Unpack values
      phi = jnp.mod(pred[:, :, 0] * (2*np.pi) * circle_mask, 2*np.pi)
      T = 0.7 * jnp.ones(shape=(n, n)) * circle_mask
      z = T * jnp.exp(1j * phi)

    else:

      # Predict phi and T
      pred = predict(x, models[f'model_{i+1}'])"""

    # Unpack values
    pred = predict(x, models[f'model_{i+1}'])
    z_real = pred[:, :, 0] * circle_mask
    z_imag = pred[:, :, 1] * circle_mask
    z = z_real + 1j * z_imag

    # Calculate field
    total_I = jnp.sum(jnp.abs(z)**2)
    field = calculate_diffraction_jax(z,
                                      lamb=lamb, Z=params['f'],
                                      n=params['n'], upsampling=params['upsampling'], delta=params['delta'])

    # Get efficiency
    eff = get_efficiency(field, q=i+1, n=params['n'], upsampling=params['upsampling'])
    if lamb == 0.76:
      loss += eff
    elif lamb == 1.61:
      loss += 0.8*eff
    elif lamb == 2.06:
      loss += eff
    else: 
      loss += 1.5*eff
    effs.append(eff)
    total_Is.append(total_I)

  return loss, (effs, total_Is)

# Make grad function
FOM_dx = jax.value_and_grad(FOM, has_aux=True)

# Parameters
delta = 5.120 / 4
f = 20000
radius = 1000
n = 512*4
upsampling = 1
lamb = [0.76, 1.61, 2.06, 4.8]
thickness = 2
params = {'lamb':lamb, 'f': f,
          'n': n, 'upsampling': upsampling, 'delta': delta}

# Init supercell profile
start_time = time.time()
arr = (np.load(path + 'metalens_supercell_init.npy') - 0.1) / 0.2
x_profile = jnp.array(arr)

# Load neural network
model_1 = tf.keras.models.load_model(path + 'neural networks/holes - supercell/supercell_2_holes_lamb_0.76_epochs=5000_res128.hdf5')
model_1_jax = jax2tf.call_tf(model_1)
model_2 = tf.keras.models.load_model(path + 'neural networks/holes - supercell/supercell_2_holes_lamb_1.61_epochs=5000.hdf5')
model_2_jax = jax2tf.call_tf(model_2)
model_3 = tf.keras.models.load_model(path + 'neural networks/holes - supercell/supercell_2_holes_lamb_2.06_epochs=5000.hdf5')
model_3_jax = jax2tf.call_tf(model_3)
model_4 = tf.keras.models.load_model(path + 'neural networks/holes - supercell/supercell_2_holes_lamb_4.8_epochs=1000.hdf5')
model_4_jax = jax2tf.call_tf(model_4)
models_jax = {'model_1': model_1_jax, 'model_2': model_2_jax,
              'model_3': model_3_jax, 'model_4': model_4_jax}

# Start optimizer
optimizer = optax.adam(learning_rate=0.01) # 0.1 - 0.300, 0.01, 0.001 - 0.755
opt_state = optimizer.init(x_profile)

# Training loop
for epoch in range(200):

  # Calculate efficiency
  (loss, (effs, total_Is)), grad = FOM_dx(x_profile, models_jax, params)

  # Print result
  if epoch % 1 == 0:
    incoming_I = 1917485.
    print(f'epoch {epoch}, loss = {loss:.03f}')
    for i, lamb in enumerate(params['lamb']):
      print(f'lamb={lamb}um : eff = {effs[i]:.03f}, T modulation: {total_Is[i]/incoming_I:.03f}, phase modulation: {(effs[i] * incoming_I) / total_Is[i]:.03f}')

  # Update optimizer
  updates, opt_state = optimizer.update(-grad, opt_state, x_profile)
  x_profile = optax.apply_updates(x_profile, updates)
  x_profile = jnp.clip(x_profile, 0, 1)

# Make FOM
loss, (effs, total_Is) = FOM(x_profile, models_jax, params)
print(f'final efficiency, loss = {loss:.03f}')
for i, lamb in enumerate(params['lamb']):
  print(f'lamb={lamb}um : eff = {effs[i]:.03f}, T modulation: {total_Is[i]/incoming_I:.03f}, phase modulation: {(effs[i] * incoming_I) / total_Is[i]:.03f}')

# Save profile
np.save(path + 'x_profile_supercell_weighted.npy', x_profile)

# Save fields
circle_mask = make_circle_mask(radius, n, delta)

for i, lamb in enumerate(params['lamb']):

    # Unpack values
    pred = predict(x_profile, models_jax[f'model_{i+1}'])
    z_real = pred[:, :, 0] * circle_mask
    z_imag = pred[:, :, 1] * circle_mask
    z = z_real + 1j * z_imag
 
    # Calculate field
    field = calculate_diffraction_jax(z,
                                      lamb=lamb, Z=params['f'],
                                      n=params['n'], upsampling=params['upsampling'], delta=params['delta'])
    
    plt.imshow(field, cmap='plasma')
    plt.colorbar()
    plt.savefig(path + f'figs/field_holes_single_cell_lamb={lamb}um.png')
    plt.show()

    plt.imshow(np.angle(z), cmap='plasma')
    plt.colorbar()
    plt.savefig(path + f'figs/phase_holes_single_cell_lamb={lamb}um.png')
    plt.show()

    np.save(path + f'figs/field_holes_single_cell_lamb={lamb}um.npy', field)

# Print time
end_time = time.time()
duration_seconds = end_time - start_time
minutes = int(duration_seconds // 60)
seconds = int(duration_seconds % 60)
print(f"Elapsed time: {minutes} minutes {seconds} seconds")