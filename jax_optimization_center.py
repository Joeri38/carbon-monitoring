import jax
import optax
from jax.experimental import jax2tf
import jax.numpy as jnp

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import argparse

# Parse arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("init", help='phase matching or uniform init',
                    type=str, choices=['phase_matching', 'uniform'])
parser.add_argument("method", help='method of optimizing and calculating the loss',
                    type=str, choices=['worst_case', 'weighted'])
parser.add_argument('--weights', type=float, nargs=3, default=[1., 1., 1.], 
                    help="provide 3 weights for the weighted method")
parser.add_argument("--lr", help='learning rate', type=float, default=0.001)
parser.add_argument("--epochs", help='epochs', type=int, default=200)
args = parser.parse_args()

# Set path
path = ''

# Make circle mask
def make_circle_mask(radius, n, delta):

    x = (jnp.arange(n) - n/2) * delta
    y = (jnp.arange(n) - n/2) * delta
    x, y = jnp.meshgrid(x, y)

    def sigmoid(x):
      return 1 / (1 + jnp.exp(-x))

    circle_mask = sigmoid(radius**2 - (x**2 + y**2))

    return circle_mask

# Helpers: prediction
def predict(x, model, n):

  pred = model(x.reshape(-1).astype('float16')).reshape(n, n, 2)

  return pred

# Helpers: calculate diffraction
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
def intensity_lobes(field, q, n, upsampling):

  n_sep = int(2000 / delta)
  n_center = int(n/2)
  x0_quadrants = [n_center, n_center - n_sep, n_center + n_sep]
  y0_quadrants = [n_center, n_center, n_center]
  x0 = x0_quadrants[q-1]
  y0 = y0_quadrants[q-1]
  pixel_radius_list = [9, 18, 23] # for n = 2048, [4, 9, 12] for n = 1024
  pixel_radius = pixel_radius_list[q-1]

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
  circle_mask = make_circle_mask(radius, n, delta)
  incoming_I = np.sum(circle_mask**2)

  # Compute focal spot
  I_1_lobe = intensity_lobes(field, q=q,
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
  focal_plane_Is = []

  for i, lamb in enumerate(params['lamb']):

    # Unpack values
    pred = predict(x, models[f'model_{i+1}'], params['n'])
    z_real = pred[:, :, 0] * circle_mask
    z_imag = pred[:, :, 1] * circle_mask
    z = z_real + 1j * z_imag
 
    # Calculate field
    total_I = jnp.sum(jnp.abs(z)**2)
    field = calculate_diffraction_jax(z,
                                      lamb=lamb, Z=params['f'],
                                      n=params['n'], upsampling=params['upsampling'], delta=params['delta'])
    focal_plane_I = jnp.sum(field)

    # Get efficiency
    eff = get_efficiency(field, q=i+1, n=params['n'], upsampling=params['upsampling'])
    effs.append(eff)
    total_Is.append(total_I)
    focal_plane_Is.append(focal_plane_I)

    # If weighted loss
    if args.method == 'weighted':
      if lamb == 0.76:
        loss += args.weights[0]*eff
      elif lamb == 1.61:
        loss += args.weights[1]*eff
      elif lamb == 2.06:
        loss += args.weights[2]*eff

  # Set loss to worst case
  if args.method == 'worst_case':
    loss = jnp.min(jnp.array(effs))

  return loss, (effs, total_Is, focal_plane_Is)

# Make grad function
FOM_dx = jax.value_and_grad(FOM, has_aux=True)

# Parameters
f = 50000
radius = 2500
n = 2048
delta = 2.7 # 6x6 pillars 450nm
upsampling = 1
lamb = [0.76, 1.61, 2.06]
params = {'lamb':lamb, 'f': f,
          'n': n, 'upsampling': upsampling, 'delta': delta}
print(f"pillars 1000nm, 0.76um center, radius={radius}um, f={f}um, lamb={lamb}um")

# Init supercell profile
start_time = time.time()
phase_matching_init = ((np.load(path + f'metalenses/pillars-1000nm/init_diameters_3_wavelengths_linear_0.76umcenter_delta={delta}um_clip_350nm.npy') - 0.1) / 0.35) * make_circle_mask(radius, n, delta)
uniform_init = make_circle_mask(radius, n, delta) * 0.5

if args.init == 'phase_matching':
  print('init: phase matching')
  x_profile = jnp.array(phase_matching_init)
elif args.init == 'uniform':
  print('init: uniform init')
  x_profile = jnp.array(uniform_init)

# Load neural network
model_1 = tf.keras.models.load_model(path + 'neural_networks/pillars-1000nm/lamb_0.76um.keras')
model_1_jax = jax2tf.call_tf(model_1)
model_2 = tf.keras.models.load_model(path + 'neural_networks/pillars-1000nm/lamb_1.61um.keras')
model_2_jax = jax2tf.call_tf(model_2)
model_3 = tf.keras.models.load_model(path + 'neural_networks/pillars-1000nm/lamb_2.06um.keras')
model_3_jax = jax2tf.call_tf(model_3)
models_jax = {'model_1': model_1_jax, 'model_2': model_2_jax,
              'model_3': model_3_jax}

# Print optimization method
if args.method == 'weighted':
  print(f'method: weighting, weights: {args.weights}')
if args.method == 'worst_case':
  print('method: worst case')

# Start optimizer
optimizer = optax.adam(learning_rate=args.lr) # 0.1 - 0.300, 0.01, 0.001 - 0.755
opt_state = optimizer.init(x_profile)

# Training loop
epochs = args.epochs
for epoch in range(epochs):

  # Calculate efficiency
  (loss, (effs, total_Is, focal_plane_Is)), grad = FOM_dx(x_profile, models_jax, params)

  # Print result
  if epoch % 1 == 0:
    circle_mask = make_circle_mask(radius, n, delta)
    incoming_I = np.sum(circle_mask**2)
    if args.verbose:
      print(f'epoch {epoch}, loss = {loss:.03f}')
      for i, lamb in enumerate(params['lamb']):
        print(f'lamb={lamb}um : eff = {effs[i]:.03f}, T modulation: {total_Is[i]/incoming_I:.03f}, phase modulation: {(effs[i] * incoming_I) / total_Is[i]:.03f}, focal plane efficiency: {focal_plane_Is[i]/incoming_I:.03f}')

  # Update optimizer
  updates, opt_state = optimizer.update(-grad, opt_state, x_profile)
  x_profile = optax.apply_updates(x_profile, updates)
  x_profile = jnp.clip(x_profile, 0, 250./350)

# Make FOM
circle_mask = make_circle_mask(radius, n, delta)
incoming_I = np.sum(circle_mask**2)
loss, (effs, total_Is, focal_plane_Is) = FOM(x_profile, models_jax, params)
print(f'final efficiency, loss = {loss:.03f}')
for i, lamb in enumerate(params['lamb']):
  print(f'lamb={lamb}um : eff = {effs[i]:.03f}, T modulation: {total_Is[i]/incoming_I:.03f}, phase modulation: {(effs[i] * incoming_I) / total_Is[i]:.03f}, focal plane efficiency: {focal_plane_Is[i]/incoming_I:.03f}')

# Save profile
np.save(path + f'metalenses/pillars-1000nm/x_profile_linear_0.76umcenter_delta={delta}um.npy', x_profile)

plt.imshow((x_profile*0.35) + 0.1, cmap='plasma')
plt.colorbar()
plt.savefig(path + f'figs/pillars-1000nm/center/diameters.png')
plt.clf()

# Save fields
circle_mask = make_circle_mask(radius, n, delta)
for i, lamb in enumerate(params['lamb']):

    # Unpack values
    pred = predict(x_profile, models_jax[f'model_{i+1}'], n)
    z_real = pred[:, :, 0] * circle_mask
    z_imag = pred[:, :, 1] * circle_mask
    z = z_real + 1j * z_imag
 
    # Calculate field
    field = calculate_diffraction_jax(z,
                                      lamb=lamb, Z=params['f'],
                                      n=params['n'], upsampling=params['upsampling'], delta=params['delta'])
    
    plt.imshow(field, cmap='plasma')
    plt.colorbar()
    plt.savefig(path + f'figs/pillars-1000nm/center/field_lamb={lamb}um.png')
    plt.clf()

    plt.imshow(np.angle(z), cmap='plasma')
    plt.colorbar()
    plt.savefig(path + f'figs/pillars-1000nm/center/phase_lamb={lamb}um.png')
    plt.clf()

    plt.imshow(np.abs(z)**2, cmap='plasma')
    plt.colorbar()
    plt.savefig(path + f'figs/pillars-1000nm/center/transmission_lamb={lamb}um.png')
    plt.clf()

    plt.imshow(z_real, cmap='plasma')
    plt.colorbar()
    plt.savefig(path + f'figs/pillars-1000nm/center/z_real_lamb={lamb}um.png')
    plt.clf()

    plt.imshow(z_imag, cmap='plasma')
    plt.colorbar()
    plt.savefig(path + f'figs/pillars-1000nm/center/z_imag_lamb={lamb}um.png')
    plt.clf()

    np.save(path + f'figs/pillars-1000nm/center/field_lamb={lamb}um.npy', field)

# Print time
end_time = time.time()
duration_seconds = end_time - start_time
minutes = int(duration_seconds // 60)
seconds = int(duration_seconds % 60)
print(f"Elapsed time: {minutes} minutes {seconds} seconds")
