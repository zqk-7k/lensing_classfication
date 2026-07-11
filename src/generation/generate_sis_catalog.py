#!/usr/bin/env python3
"""Generate singular-isothermal-sphere lensed simulation catalogs."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bilby
from bilby.gw.conversion import luminosity_distance_to_redshift
from bilby.gw.conversion import redshift_to_luminosity_distance
from astropy import units
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from gwpy.time import Time as GWTime
import os
from pathlib import Path
import tempfile
import shutil

# ==========================================
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVE_DIR = os.environ.get("GW_SIS_DATA_DIR", str(PROJECT_ROOT / "data" / "ligo_full" / "SIS_data"))

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"Created directory: {SAVE_DIR}")
else:
    print(f"Using directory: {SAVE_DIR}")
# ==========================================

#tmp_dir = tempfile.gettempdir()
tmp_dir = os.environ.get("GW_TMPDIR", str(PROJECT_ROOT / ".tmp"))
os.makedirs(tmp_dir, exist_ok=True)
print("Using tmp_dir:", tmp_dir)
bilby.gw.cosmology.DEFAULT_COSMOLOGY = cosmo
bilby.gw.cosmology.COSMOLOGY = [cosmo, cosmo.name]
bilby.gw.cosmology.get_cosmology()  # implementation detail






# random seed

n = 6130  # lensed




n_samples = 10000

z_min = 0.01
z_max = 1
d_L_min = redshift_to_luminosity_distance(z_min)
d_L_max = redshift_to_luminosity_distance(z_max)
print("d_L_min:", d_L_min)
print("d_L_max:", d_L_max)

m_min = 10
m_max = 100

time_start = GWTime('2015-09-14 09:50:45.39', scale='utc').gps
time_end = GWTime('2025-12-10 17:18:45.39', scale='utc').gps

time_end




n_lens = n_samples




sampling_frequency = 4096  # f_s >= 2*f_max   #
duration = 24  # len(strain_data) = f_s*duration
minimum_frequency = 20.0

# # 1. Source generation
# ## 1.1 All GW sources



bilby.core.utils.random.seed(n)


# sampling from priors
def prior_GW():
    priors = bilby.core.prior.PriorDict()
    priors['luminosity_distance'] = bilby.gw.prior.UniformComovingVolume(
        name="luminosity_distance", minimum=d_L_min, maximum=d_L_max, latex_label="r'$d_L$ (Mpc)'")
    priors['mass_1_source'] = bilby.core.prior.Uniform(m_min, m_max, 'mass_1_source')
    priors['mass_2_source'] = bilby.core.prior.Uniform(m_min, m_max, 'mass_2_source')
    priors['a_1'] = bilby.core.prior.Uniform(0, 0.99, 'a_1')
    priors['a_2'] = bilby.core.prior.Uniform(0, 0.99, 'a_2')
    priors['tilt_1'] = bilby.core.prior.Sine(name='tilt_1')
    priors['tilt_2'] = bilby.core.prior.Sine(name='tilt_2')
    priors['phi_12'] = bilby.core.prior.Uniform(0, 2 * np.pi, 'phi_12', boundary='periodic')
    priors['phi_jl'] = bilby.core.prior.Uniform(0, 2 * np.pi, 'phi_jl', boundary='periodic')
    priors['ra'] = bilby.core.prior.Uniform(0, 2 * np.pi, 'ra', boundary='periodic')
    priors['dec'] = bilby.core.prior.Cosine(name='dec')
    priors['theta_jn'] = bilby.core.prior.Sine(name='theta_jn')
    priors['psi'] = bilby.core.prior.Uniform(0, np.pi, 'psi', boundary='periodic')
    priors['phase'] = bilby.core.prior.Uniform(0, 2 * np.pi, 'phase', boundary='periodic')
    priors['geocent_time'] = bilby.core.prior.Uniform(time_start, time_end, 'geocent_time')

    return priors


priors = prior_GW()
samples = priors.sample(n_samples)
print(samples.keys())

fig, axes = plt.subplots(5, 4, figsize=(20, 15))
axes = axes.flatten()
for i, key in enumerate(samples.keys()):
    ax = axes[i]
    ax.hist(samples[key], bins=50, density=True, label=key)
    ax.set_xlabel(key)
    ax.set_ylabel('Density')
    ax.legend()
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
# plt.savefig('source_samples.pdf')


source_params = pd.DataFrame(samples)
source_params.to_csv(os.path.join(SAVE_DIR, 'source_samples.csv'), index=False)
source_params

# ## 1.2 Lensed GW index



source_params = pd.read_csv(os.path.join(SAVE_DIR, 'source_samples.csv'))

np.random.seed(n)
idx = np.random.choice(n_samples, n_lens, replace=False)
idx = np.sort(idx)

print(idx)
idx = {'lensed_index': idx}
lensed_index = pd.DataFrame(idx)
lensed_index.to_csv(os.path.join(SAVE_DIR, 'lensed_index.csv'), index=False)
lensed_index

lensed_source_1 = source_params.iloc[lensed_index['lensed_index']]
lensed_source_1

# ## 1.3 Lens modeling




np.random.seed(n)
rng = np.random.default_rng(n)

tmp_params = os.path.join(tmp_dir, "lens_params.tmp.csv")
tmp_lens = os.path.join(tmp_dir, "lens.tmp.csv")

for p in [tmp_params, tmp_lens]:
    if os.path.exists(p):
        os.remove(p)

pd.DataFrame(columns=["z_l",
                      "z_s",
                      "sigma_v",
                      "theta_E(arcsec)",
                      "y",
                      "beta_x",
                      "beta_y"]).to_csv(tmp_params, index=False)

pd.DataFrame(columns=["mu_0",
                      "mu_1",
                      "t_d"]).to_csv(tmp_lens, index=False)

for i in range(n_lens):
    dL = lensed_source_1.iloc[i]['luminosity_distance']
    z_s = luminosity_distance_to_redshift(dL, cosmology=cosmo)

    z_l = z_s / 2

    sigma_v = np.random.uniform(100, 500)

    lens_cosmo = LensCosmo(z_lens=z_l, z_source=z_s, cosmo=cosmo)

    theta_E = lens_cosmo.sis_sigma_v2theta_E(sigma_v)

    print('z_s =', z_s)
    print('z_l =', z_l)
    print('sigma_v =', sigma_v)
    print('theta_E =', theta_E)

    y = np.random.uniform(0.01, 0.3)
    phi = np.random.uniform(0, 2 * np.pi)
    beta_x = y * theta_E * np.cos(phi)
    beta_y = y * theta_E * np.sin(phi)

    print('y =', y)
    print('beta_x =', beta_x)
    print('beta_y =', beta_y)

    mu_plus = 1 + (1 / y)
    mu_minus = 1 - (1 / y) if (y <= 1) else np.nan

    Dls = (lens_cosmo.dds * units.Mpc).to(units.m).value
    Dl = (lens_cosmo.dd * units.Mpc).to(units.m).value
    Ds = (lens_cosmo.ds * units.Mpc).to(units.m).value

    t_d = 8 * 4 * np.pi ** 2 * (sigma_v * 1000) ** 4 * (1 + z_l) * Dls * Dl * y / (Ds * const.c.value ** 5)
    print(mu_plus, mu_minus)
    print(t_d, 's', '=', t_d / (24 * 3600), 'days')
    print()

    row_0 = pd.DataFrame([{
        "z_l": z_l,
        "z_s": z_s,
        "sigma_v": sigma_v,
        "theta_E(arcsec)": theta_E,
        "y": y,
        "beta_x": beta_x,
        "beta_y": beta_y}])

    row_1 = pd.DataFrame([{
        "mu_0": mu_plus,
        "mu_1": mu_minus,
        "t_d": t_d}])

    row_0.to_csv(tmp_params, mode="a", header=False, index=False)
    row_1.to_csv(tmp_lens, mode="a", header=False, index=False)

final_params = os.path.join(SAVE_DIR, "lens_params.csv")
final_lens = os.path.join(SAVE_DIR, "lens.csv")

for p in [final_params, final_lens]:
    if os.path.exists(p):
        os.remove(p)

shutil.move(tmp_params, final_params)
shutil.move(tmp_lens, final_lens)

# ## 1.4 All lensed GW source



lens = pd.read_csv(os.path.join(SAVE_DIR, 'lens.csv'))

t_d = []
for i in range(len(lens)):
    t_d.append((lens['t_d'][i]))
print(t_d)

original_times = lensed_source_1['geocent_time'].values
# print(original_times)

geocent_time_2 = original_times + t_d
# print(geocent_time_2)

lensed_source_2 = lensed_source_1.copy()
lensed_source_2['geocent_time'] = geocent_time_2

lensed_source_params = pd.concat([lensed_source_1, lensed_source_2], axis=0, ignore_index=False)
lensed_source_params.to_csv(os.path.join(SAVE_DIR, 'lensed_source_samples.csv'), index=False)

lensed_source_params

# ## 1.5 Test unlensed GW source



np.random.seed(n)

m = 2

z_1 = luminosity_distance_to_redshift(lensed_source_1['luminosity_distance'].values)
z_2 = luminosity_distance_to_redshift(lensed_source_2['luminosity_distance'].values)

d_L_new_1 = lensed_source_1['luminosity_distance'].values / np.sqrt(lens['mu_0'].values)
d_L_new_2 = lensed_source_2['luminosity_distance'].values / np.sqrt(np.abs(lens['mu_1'].values))
print(d_L_new_1)
print(d_L_new_2)

d_L_test_1 = []
time_test_1 = []
d_L_test_2 = []
time_test_2 = []

for i in range(len(d_L_new_1)):
    
    d_L_t_1 = np.maximum(d_L_new_1[i] + np.random.uniform(-100, 100, m), 1.0)
    d_L_test_1.append(d_L_t_1)
    time_t_1 = original_times[i] + np.random.uniform(-100, 100, m)  # implementation detail
    time_test_1.append(time_t_1)

    d_L_t_2 = np.maximum(d_L_new_2[i] + np.random.uniform(-100, 100, m), 1.0)
    d_L_test_2.append(d_L_t_2)
    time_t_2 = geocent_time_2[i] + np.random.uniform(-100, 100, m)  # implementation detail
    time_test_2.append(time_t_2)

print(d_L_test_1)
print(time_test_1)
print(d_L_test_2)
print(time_test_2)

tmp_test_source_1 = os.path.join(tmp_dir, "test_source_samples_1.tmp.csv")
tmp_test_source_2 = os.path.join(tmp_dir, "test_source_samples_2.tmp.csv")

for p in [tmp_test_source_1, tmp_test_source_2]:
    if os.path.exists(p):
        os.remove(p)

pd.DataFrame(columns=['luminosity_distance',
                      'mass_1',
                      'mass_2',
                      'a_1',
                      'a_2',
                      'tilt_1',
                      'tilt_2',
                      'phi_12',
                      'phi_jl',
                      'ra',
                      'dec',
                      'theta_jn',
                      'psi',
                      'phase',
                      'geocent_time']).to_csv(tmp_test_source_1, index=False)

pd.DataFrame(columns=['luminosity_distance',
                      'mass_1',
                      'mass_2',
                      'a_1',
                      'a_2',
                      'tilt_1',
                      'tilt_2',
                      'phi_12',
                      'phi_jl',
                      'ra',
                      'dec',
                      'theta_jn',
                      'psi',
                      'phase',
                      'geocent_time']).to_csv(tmp_test_source_2, index=False)

source_test_1 = lensed_source_1.copy()
source_test_2 = lensed_source_2.copy()

for i in range(len(source_test_1)):
    for j in range(m):
        para_1 = pd.DataFrame([{
            'luminosity_distance': d_L_test_1[i][j],
            'mass_1': source_test_1.iloc[i]['mass_1_source'] * (1 + z_1[i]),
            'mass_2': source_test_1.iloc[i]['mass_2_source'] * (1 + z_2[i]),
            'a_1': source_test_1.iloc[i]['a_1'],
            'a_2': source_test_1.iloc[i]['a_2'],
            'tilt_1': source_test_1.iloc[i]['tilt_1'],
            'tilt_2': source_test_1.iloc[i]['tilt_2'],
            'phi_12': source_test_1.iloc[i]['phi_12'],
            'phi_jl': source_test_1.iloc[i]['phi_jl'],
            'ra': source_test_1.iloc[i]['ra'],
            'dec': source_test_1.iloc[i]['dec'],
            'theta_jn': source_test_1.iloc[i]['theta_jn'],
            'psi': source_test_1.iloc[i]['psi'],
            'phase': source_test_1.iloc[i]['phase'],
            'geocent_time': time_test_1[i][j]
        }])

        para_2 = pd.DataFrame([{
            'luminosity_distance': d_L_test_2[i][j],
            'mass_1': source_test_2.iloc[i]['mass_1_source'] * (1 + z_1[i]),
            'mass_2': source_test_2.iloc[i]['mass_2_source'] * (1 + z_2[i]),
            'a_1': source_test_2.iloc[i]['a_1'],
            'a_2': source_test_2.iloc[i]['a_2'],
            'tilt_1': source_test_2.iloc[i]['tilt_1'],
            'tilt_2': source_test_2.iloc[i]['tilt_2'],
            'phi_12': source_test_2.iloc[i]['phi_12'],
            'phi_jl': source_test_2.iloc[i]['phi_jl'],
            'ra': source_test_2.iloc[i]['ra'],
            'dec': source_test_2.iloc[i]['dec'],
            'theta_jn': source_test_2.iloc[i]['theta_jn'],
            'psi': source_test_2.iloc[i]['psi'],
            'phase': source_test_2.iloc[i]['phase'],
            'geocent_time': time_test_2[i][j]
        }])

        para_1.to_csv(tmp_test_source_1, mode="a", header=False, index=False)
        para_2.to_csv(tmp_test_source_2, mode="a", header=False, index=False)

final_test_source_1 = os.path.join(SAVE_DIR, "test_source_samples_1.csv")
final_test_source_2 = os.path.join(SAVE_DIR, "test_source_samples_2.csv")

for p in [final_test_source_1, final_test_source_2]:
    if os.path.exists(p):
        os.remove(p)

shutil.move(tmp_test_source_1, final_test_source_1)
shutil.move(tmp_test_source_2, final_test_source_2)

test_source_samples_1 = pd.read_csv(os.path.join(SAVE_DIR, 'test_source_samples_1.csv'))
test_source_samples_2 = pd.read_csv(os.path.join(SAVE_DIR, 'test_source_samples_2.csv'))
test_source_samples_1


# # 2. Lensed waveform



# define our lensed (SIS) waveform
def lens_SIS_ampfac(mu_0, mu_1, t_d, frequencies, which_image=None):
    mu = np.array([mu_0, mu_1])
    t_d = np.array([0, t_d])
    n = np.array([0, 1 / 2])

    F = np.zeros(len(frequencies), dtype=np.complex128)

    if which_image is None:
        # Calculate the amplification factor for all images
        for i in range(len(mu)):
            F += np.sqrt(np.abs(mu[i])) * np.exp(-1j * np.pi * (2. * frequencies * t_d[i] + n[i]))

        return {'F': F, 'mu': mu, 't_d': t_d, 'n': n}

    else:
        # Calculate the amplification factor for a specific image
        i = which_image
        F = np.sqrt(np.abs(mu[i])) * np.exp(-1j * np.pi * (2. * frequencies * t_d[i] + n[i]))

        return {'F': F, 'mu': mu[i], 't_d': t_d[i], 'n': n[i]}


def lensed_waveform_F(frequency_array, mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl,
                      luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec,
                      mu_0, mu_1, t_d, which_image=None, **kwargs):
    # GW waveform
    params = {
        'mass_1': mass_1, 'mass_2': mass_2, 'a_1': a_1, 'a_2': a_2,
        'tilt_1': tilt_1, 'tilt_2': tilt_2, 'phi_12': phi_12, 'phi_jl': phi_jl,
        'luminosity_distance': luminosity_distance, 'theta_jn': theta_jn,
        'psi': psi, 'phase': phase, 'geocent_time': geocent_time,
        'ra': ra, 'dec': dec}

    h = bilby.gw.source.lal_binary_black_hole(frequency_array, **params, **kwargs)

    z_s = luminosity_distance_to_redshift(luminosity_distance)
    print('z_s:', z_s)

    F = lens_SIS_ampfac(mu_0, mu_1, t_d, frequency_array, which_image=which_image)
    print('mu:', F['mu'])
    print('time delay(s):', str(F['t_d']) + ' s' + ' = ' + str(F['t_d'] / (24 * 60 * 60)) + ' days')
    print('Morse indice:', F['n'])
    print()

    return {'plus': h['plus'] * F['F'], 'cross': h['cross'] * F['F']}


# # 3. GW data

# ## 3.1 Lensed event data
# ### 3.1.1 First image data



bilby.core.utils.random.seed(n + 1)

N = sampling_frequency * duration

waveform_arguments = dict(waveform_approximant='IMRPhenomXPHM',
                          reference_frequency=10., minimum_frequency=minimum_frequency)

# lensed_waveform
waveform_generator_lensed = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=sampling_frequency,
    duration=duration,
    frequency_domain_source_model=lensed_waveform_F,
    waveform_arguments=waveform_arguments)

lensed_source_params = pd.read_csv(os.path.join(SAVE_DIR, 'lensed_source_samples.csv'))
lens = pd.read_csv(os.path.join(SAVE_DIR, 'lens.csv'))
n_events = len(lensed_source_params) // 2

# LIGO
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])  # implementation detail
ifo_names = [ifo.name for ifo in ifos]
n_ifos = len(ifos)

# -------------------------
# -------------------------
tmp_data_1 = os.path.join(tmp_dir, "SIS_data_strain_1.tmp.npy")
tmp_h_1    = os.path.join(tmp_dir, "SIS_h_strain_1.tmp.npy")
tmp_t_1    = os.path.join(tmp_dir, "SIS_time_array_1.tmp.npy")
tmp_snr_single_1 = os.path.join(tmp_dir, "SIS_optimal_SNR_single_1.tmp.npy")
tmp_snr_net_1    = os.path.join(tmp_dir, "SIS_optimal_SNR_network_1.tmp.npy")


for p in [tmp_data_1, tmp_h_1, tmp_t_1, tmp_snr_single_1, tmp_snr_net_1]:
    if os.path.exists(p):
        os.remove(p)

# shape(n_events, n_ifos, N)
data_whiten_mm_1 = np.lib.format.open_memmap(
    tmp_data_1, mode="w+", dtype=np.float64, shape=(n_events, n_ifos, N))

h_whiten_mm_1 = np.lib.format.open_memmap(
    tmp_h_1, mode="w+", dtype=np.float64, shape=(n_events, n_ifos, N))

t_mm_1 = np.lib.format.open_memmap(
    tmp_t_1, mode="w+", dtype=np.float64, shape=(n_events, n_ifos, N))

snr_single_mm_1 = np.lib.format.open_memmap(
    tmp_snr_single_1, mode="w+", dtype=np.float64, shape=(n_events, n_ifos))  # implementation detail

snr_net_mm_1 = np.lib.format.open_memmap(
    tmp_snr_net_1, mode="w+", dtype=np.float64, shape=(n_events,))  # implementation detail



for i in range(len(lensed_source_params) // 2):
    injection_parameters = {
        **lensed_source_params.iloc[i].to_dict(),
        **lens.iloc[i].to_dict(),
        'which_image': 0
    }

    r = 1500
    target_index = N - r

    fd_waveform = waveform_generator_lensed.frequency_domain_strain(injection_parameters)

    snr_sq_network = 0.0

    for j, ifo in enumerate(ifos):
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=injection_parameters["geocent_time"] - (duration - (r / sampling_frequency))
        )

        n_t = ifo.strain_data.time_domain_strain.copy()
        t_array = ifo.strain_data.time_array.copy()

        n_f_white = ifo.whitened_frequency_domain_strain
        n_t_white = np.fft.irfft(n_f_white, n=N)

        response = ifo.get_detector_response(fd_waveform, injection_parameters)
        snr_squared = np.real(ifo.optimal_snr_squared(response))
        snr = np.sqrt(np.abs(snr_squared))

        snr_single_mm_1[i, j] = snr
        snr_sq_network += np.abs(snr_squared)

        ifo.inject_signal(
            waveform_generator=waveform_generator_lensed,
            parameters=injection_parameters
        )

        d_t = ifo.strain_data.time_domain_strain.copy()

        d_f_white = ifo.whitened_frequency_domain_strain
        d_t_white = np.fft.irfft(d_f_white, n=N)

        h_t = d_t - n_t
        h_t_white = d_t_white - n_t_white

        peak_index = np.argmax(np.abs(h_t))
        peak_index_white = np.argmax(np.abs(h_t_white))

        shift = target_index - peak_index
        shift_white = target_index - peak_index_white

        h_t_roll = np.roll(h_t, shift)
        h_t_white_roll = np.roll(h_t_white, shift_white)

        index = target_index + int(0.05 * sampling_frequency)
        index = min(index, N)

        fade_len = N - index
        if fade_len > 0:
            hann_full = np.hanning(2 * fade_len)
            fade_window = hann_full[fade_len:]  # implementation detail

            h_t_roll[index:index + fade_len] *= fade_window
            h_t_white_roll[index:index + fade_len] *= fade_window

            if index + fade_len < N:
                h_t_roll[index + fade_len:] = 0
                h_t_white_roll[index + fade_len:] = 0

        d_t_new = h_t_roll + n_t
        d_t_white_new = h_t_white_roll + n_t_white

        t_mm_1[i, j, :] = t_array
        data_whiten_mm_1[i, j, :] = d_t_white_new
        h_whiten_mm_1[i, j, :] = h_t_white_roll

    snr_net_mm_1[i] = np.sqrt(snr_sq_network)

data_whiten_mm_1.flush()
h_whiten_mm_1.flush()
t_mm_1.flush()
snr_single_mm_1.flush()
snr_net_mm_1.flush()

final_data_1 = os.path.join(SAVE_DIR, "SIS_data_strain_1.npy")
final_h_1 = os.path.join(SAVE_DIR, "SIS_h_strain_1.npy")
final_t_1 = os.path.join(SAVE_DIR, "SIS_time_array_1.npy")
final_snr_single_1 = os.path.join(SAVE_DIR, "SIS_optimal_SNR_single_1.npy")
final_snr_net_1 = os.path.join(SAVE_DIR, "SIS_optimal_SNR_network_1.npy")

for p in [final_data_1, final_h_1, final_t_1, final_snr_single_1, final_snr_net_1]:
    if os.path.exists(p):
        os.remove(p)

shutil.move(tmp_data_1, final_data_1)
shutil.move(tmp_h_1, final_h_1)
shutil.move(tmp_t_1, final_t_1)
shutil.move(tmp_snr_single_1, final_snr_single_1)
shutil.move(tmp_snr_net_1, final_snr_net_1)


# ### 3.1.2 Second image data



bilby.core.utils.random.seed(n + 2)

N = int(sampling_frequency * duration)

waveform_arguments = dict(
    waveform_approximant='IMRPhenomXPHM',
    reference_frequency=10.,
    minimum_frequency=minimum_frequency
)

# lensed_waveform
waveform_generator_lensed = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=sampling_frequency,
    duration=duration,
    frequency_domain_source_model=lensed_waveform_F,
    waveform_arguments=waveform_arguments
)

# LIGO
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifo_names = [ifo.name for ifo in ifos]
n_ifos = len(ifos)

tmp_data_2 = os.path.join(tmp_dir, "SIS_data_strain_2.tmp.npy")
tmp_h_2 = os.path.join(tmp_dir, "SIS_h_strain_2.tmp.npy")
tmp_t_2 = os.path.join(tmp_dir, "SIS_time_array_2.tmp.npy")
tmp_snr_single_2 = os.path.join(tmp_dir, "SIS_optimal_SNR_single_2.tmp.npy")
tmp_snr_net_2 = os.path.join(tmp_dir, "SIS_optimal_SNR_network_2.tmp.npy")

for p in [tmp_data_2, tmp_h_2, tmp_t_2, tmp_snr_single_2, tmp_snr_net_2]:
    if os.path.exists(p):
        os.remove(p)

data_whiten_mm_2 = np.lib.format.open_memmap(
    tmp_data_2, mode="w+", dtype=np.float64, shape=(n_events, n_ifos, N))

h_whiten_mm_2 = np.lib.format.open_memmap(
    tmp_h_2, mode="w+", dtype=np.float64, shape=(n_events, n_ifos, N))

t_mm_2 = np.lib.format.open_memmap(
    tmp_t_2, mode="w+", dtype=np.float64, shape=(n_events, n_ifos, N))

snr_single_mm_2 = np.lib.format.open_memmap(
    tmp_snr_single_2, mode="w+", dtype=np.float64, shape=(n_events, n_ifos))

snr_net_mm_2 = np.lib.format.open_memmap(
    tmp_snr_net_2, mode="w+", dtype=np.float64, shape=(n_events,))

for i in range(len(lensed_source_params) // 2):
    injection_parameters = {
        **lensed_source_params.iloc[i + (len(lensed_source_params) // 2)].to_dict(),
        **lens.iloc[i].to_dict(),
        'which_image': 1
    }

    r = 1500
    target_index = N - r

    fd_waveform = waveform_generator_lensed.frequency_domain_strain(injection_parameters)

    snr_sq_network = 0.0

    for j, ifo in enumerate(ifos):
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=injection_parameters["geocent_time"] - (duration - (r / sampling_frequency))
        )

        n_t = ifo.strain_data.time_domain_strain.copy()
        t_array = ifo.strain_data.time_array.copy()

        n_f_white = ifo.whitened_frequency_domain_strain
        n_t_white = np.fft.irfft(n_f_white, n=N)

        response = ifo.get_detector_response(fd_waveform, injection_parameters)
        snr_squared = np.real(ifo.optimal_snr_squared(response))
        snr = np.sqrt(np.abs(snr_squared))

        snr_single_mm_2[i, j] = snr
        snr_sq_network += np.abs(snr_squared)

        ifo.inject_signal(
            waveform_generator=waveform_generator_lensed,
            parameters=injection_parameters
        )

        d_t = ifo.strain_data.time_domain_strain.copy()

        d_f_white = ifo.whitened_frequency_domain_strain
        d_t_white = np.fft.irfft(d_f_white, n=N)

        h_t = d_t - n_t
        h_t_white = d_t_white - n_t_white

        peak_index = np.argmax(np.abs(h_t))
        peak_index_white = np.argmax(np.abs(h_t_white))

        shift = target_index - peak_index
        shift_white = target_index - peak_index_white

        h_t_roll = np.roll(h_t, shift)
        h_t_white_roll = np.roll(h_t_white, shift_white)

        index = target_index + int(0.05 * sampling_frequency)
        index = min(index, N)

        fade_len = N - index
        if fade_len > 0:
            hann_full = np.hanning(2 * fade_len)
            fade_window = hann_full[fade_len:]  # implementation detail

            h_t_roll[index:index + fade_len] *= fade_window
            h_t_white_roll[index:index + fade_len] *= fade_window

            if index + fade_len < N:
                h_t_roll[index + fade_len:] = 0
                h_t_white_roll[index + fade_len:] = 0

        d_t_new = h_t_roll + n_t
        d_t_white_new = h_t_white_roll + n_t_white

        t_mm_2[i, j, :] = t_array
        data_whiten_mm_2[i, j, :] = d_t_white_new
        h_whiten_mm_2[i, j, :] = h_t_white_roll

    snr_net_mm_2[i] = np.sqrt(snr_sq_network)

data_whiten_mm_2.flush()
h_whiten_mm_2.flush()
t_mm_2.flush()
snr_single_mm_2.flush()
snr_net_mm_2.flush()

final_data_2 = os.path.join(SAVE_DIR, "SIS_data_strain_2.npy")
final_h_2 = os.path.join(SAVE_DIR, "SIS_h_strain_2.npy")
final_t_2 = os.path.join(SAVE_DIR, "SIS_time_array_2.npy")
final_snr_single_2 = os.path.join(SAVE_DIR, "SIS_optimal_SNR_single_2.npy")
final_snr_net_2 = os.path.join(SAVE_DIR, "SIS_optimal_SNR_network_2.npy")

for p in [final_data_2, final_h_2, final_t_2, final_snr_single_2, final_snr_net_2]:
    if os.path.exists(p):
        os.remove(p)

shutil.move(tmp_data_2, final_data_2)
shutil.move(tmp_h_2, final_h_2)
shutil.move(tmp_t_2, final_t_2)
shutil.move(tmp_snr_single_2, final_snr_single_2)
shutil.move(tmp_snr_net_2, final_snr_net_2)


# ## 3.2 Test unlensed data
# ### 3.2.1  Test data for first image



bilby.core.utils.random.seed(n + 3)

N = int(sampling_frequency * duration)

waveform_arguments = dict(
    waveform_approximant='IMRPhenomXPHM',
    reference_frequency=10.,
    minimum_frequency=minimum_frequency
)

# unlensed_waveform
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments
)

test_params = pd.read_csv(os.path.join(SAVE_DIR, 'test_source_samples_1.csv'))
n_events_test = len(test_params)

# LIGO
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifo_names = [ifo.name for ifo in ifos]
n_ifos = len(ifos)

test_tmp_data_1 = os.path.join(tmp_dir, "test_SIS_data_strain_1.tmp.npy")
test_tmp_h_1 = os.path.join(tmp_dir, "test_SIS_h_strain_1.tmp.npy")
test_tmp_t_1 = os.path.join(tmp_dir, "test_SIS_time_array_1.tmp.npy")
test_tmp_snr_single_1 = os.path.join(tmp_dir, "test_SIS_optimal_SNR_single_1.tmp.npy")
test_tmp_snr_net_1 = os.path.join(tmp_dir, "test_SIS_optimal_SNR_network_1.tmp.npy")

for p in [test_tmp_data_1, test_tmp_h_1, test_tmp_t_1, test_tmp_snr_single_1, test_tmp_snr_net_1]:
    if os.path.exists(p):
        os.remove(p)

test_data_whiten_mm_1 = np.lib.format.open_memmap(
    test_tmp_data_1, mode="w+", dtype=np.float64, shape=(n_events_test, n_ifos, N)
)
test_h_whiten_mm_1 = np.lib.format.open_memmap(
    test_tmp_h_1, mode="w+", dtype=np.float64, shape=(n_events_test, n_ifos, N)
)
test_t_mm_1 = np.lib.format.open_memmap(
    test_tmp_t_1, mode="w+", dtype=np.float64, shape=(n_events_test, n_ifos, N)
)
test_snr_single_mm_1 = np.lib.format.open_memmap(
    test_tmp_snr_single_1, mode="w+", dtype=np.float64, shape=(n_events_test, n_ifos)
)
test_snr_net_mm_1 = np.lib.format.open_memmap(
    test_tmp_snr_net_1, mode="w+", dtype=np.float64, shape=(n_events_test,)
)

for i in range(n_events_test):
    injection_parameters = test_params.iloc[i].to_dict()

    r = 1500
    target_index = N - r

    fd_waveform = waveform_generator.frequency_domain_strain(injection_parameters)

    snr_sq_network = 0.0

    for j, ifo in enumerate(ifos):
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=injection_parameters["geocent_time"] - (duration - (r / sampling_frequency))
        )

        n_t = ifo.strain_data.time_domain_strain.copy()
        t_array = ifo.strain_data.time_array.copy()

        n_f_white = ifo.whitened_frequency_domain_strain
        n_t_white = np.fft.irfft(n_f_white, n=N)

        response = ifo.get_detector_response(fd_waveform, injection_parameters)
        snr_squared = np.real(ifo.optimal_snr_squared(response))
        snr = np.sqrt(np.abs(snr_squared))

        test_snr_single_mm_1[i, j] = snr
        snr_sq_network += np.abs(snr_squared)

        ifo.inject_signal(
            waveform_generator=waveform_generator,
            parameters=injection_parameters
        )

        d_t = ifo.strain_data.time_domain_strain.copy()

        d_f_white = ifo.whitened_frequency_domain_strain
        d_t_white = np.fft.irfft(d_f_white, n=N)

        h_t = d_t - n_t
        h_t_white = d_t_white - n_t_white

        peak_index = np.argmax(np.abs(h_t))
        peak_index_white = np.argmax(np.abs(h_t_white))

        shift = target_index - peak_index
        shift_white = target_index - peak_index_white

        h_t_roll = np.roll(h_t, shift)
        h_t_white_roll = np.roll(h_t_white, shift_white)

        index = target_index + int(0.05 * sampling_frequency)
        index = min(index, N)

        fade_len = N - index
        if fade_len > 0:
            hann_full = np.hanning(2 * fade_len)
            fade_window = hann_full[fade_len:]

            h_t_roll[index:index + fade_len] *= fade_window
            h_t_white_roll[index:index + fade_len] *= fade_window

            if index + fade_len < N:
                h_t_roll[index + fade_len:] = 0
                h_t_white_roll[index + fade_len:] = 0

        d_t_new = h_t_roll + n_t
        d_t_white_new = h_t_white_roll + n_t_white

        test_t_mm_1[i, j, :] = t_array
        test_data_whiten_mm_1[i, j, :] = d_t_white_new
        test_h_whiten_mm_1[i, j, :] = h_t_white_roll

    test_snr_net_mm_1[i] = np.sqrt(snr_sq_network)

test_data_whiten_mm_1.flush()
test_h_whiten_mm_1.flush()
test_t_mm_1.flush()
test_snr_single_mm_1.flush()
test_snr_net_mm_1.flush()

test_final_data_1 = os.path.join(SAVE_DIR, "test_SIS_data_strain_1.npy")
test_final_h_1 = os.path.join(SAVE_DIR, "test_SIS_h_strain_1.npy")
test_final_t_1 = os.path.join(SAVE_DIR, "test_SIS_time_array_1.npy")
test_final_snr_single_1 = os.path.join(SAVE_DIR, "test_SIS_optimal_SNR_single_1.npy")
test_final_snr_net_1 = os.path.join(SAVE_DIR, "test_SIS_optimal_SNR_network_1.npy")

for p in [test_final_data_1, test_final_h_1, test_final_t_1, test_final_snr_single_1, test_final_snr_net_1]:
    if os.path.exists(p):
        os.remove(p)

shutil.move(test_tmp_data_1, test_final_data_1)
shutil.move(test_tmp_h_1, test_final_h_1)
shutil.move(test_tmp_t_1, test_final_t_1)
shutil.move(test_tmp_snr_single_1, test_final_snr_single_1)
shutil.move(test_tmp_snr_net_1, test_final_snr_net_1)


# ### 3.2.2  Test data for second image



bilby.core.utils.random.seed(n + 4)

N = int(sampling_frequency * duration)

waveform_arguments = dict(
    waveform_approximant='IMRPhenomXPHM',
    reference_frequency=10.,
    minimum_frequency=minimum_frequency
)

# unlensed_waveform
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments
)

test_params = pd.read_csv(os.path.join(SAVE_DIR, 'test_source_samples_2.csv'))
n_events_test = len(test_params)

# LIGO
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifo_names = [ifo.name for ifo in ifos]
n_ifos = len(ifos)

test_tmp_data_2 = os.path.join(tmp_dir, "test_SIS_data_strain_2.tmp.npy")
test_tmp_h_2 = os.path.join(tmp_dir, "test_SIS_h_strain_2.tmp.npy")
test_tmp_t_2 = os.path.join(tmp_dir, "test_SIS_time_array_2.tmp.npy")
test_tmp_snr_single_2 = os.path.join(tmp_dir, "test_SIS_optimal_SNR_single_2.tmp.npy")
test_tmp_snr_net_2 = os.path.join(tmp_dir, "test_SIS_optimal_SNR_network_2.tmp.npy")

for p in [test_tmp_data_2, test_tmp_h_2, test_tmp_t_2, test_tmp_snr_single_2, test_tmp_snr_net_2]:
    if os.path.exists(p):
        os.remove(p)

test_data_whiten_mm_2 = np.lib.format.open_memmap(
    test_tmp_data_2, mode="w+", dtype=np.float64, shape=(n_events_test, n_ifos, N)
)
test_h_whiten_mm_2 = np.lib.format.open_memmap(
    test_tmp_h_2, mode="w+", dtype=np.float64, shape=(n_events_test, n_ifos, N)
)
test_t_mm_2 = np.lib.format.open_memmap(
    test_tmp_t_2, mode="w+", dtype=np.float64, shape=(n_events_test, n_ifos, N)
)
test_snr_single_mm_2 = np.lib.format.open_memmap(
    test_tmp_snr_single_2, mode="w+", dtype=np.float64, shape=(n_events_test, n_ifos)
)
test_snr_net_mm_2 = np.lib.format.open_memmap(
    test_tmp_snr_net_2, mode="w+", dtype=np.float64, shape=(n_events_test,)
)

for i in range(n_events_test):
    injection_parameters = test_params.iloc[i].to_dict()

    r = 1500
    target_index = N - r

    fd_waveform = waveform_generator.frequency_domain_strain(injection_parameters)

    snr_sq_network = 0.0

    for j, ifo in enumerate(ifos):
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=injection_parameters["geocent_time"] - (duration - (r / sampling_frequency))
        )

        n_t = ifo.strain_data.time_domain_strain.copy()
        t_array = ifo.strain_data.time_array.copy()

        n_f_white = ifo.whitened_frequency_domain_strain
        n_t_white = np.fft.irfft(n_f_white, n=N)

        response = ifo.get_detector_response(fd_waveform, injection_parameters)
        snr_squared = np.real(ifo.optimal_snr_squared(response))
        snr = np.sqrt(np.abs(snr_squared))

        test_snr_single_mm_2[i, j] = snr
        snr_sq_network += np.abs(snr_squared)

        ifo.inject_signal(
            waveform_generator=waveform_generator,
            parameters=injection_parameters
        )

        d_t = ifo.strain_data.time_domain_strain.copy()

        d_f_white = ifo.whitened_frequency_domain_strain
        d_t_white = np.fft.irfft(d_f_white, n=N)

        h_t = d_t - n_t
        h_t_white = d_t_white - n_t_white

        peak_index = np.argmax(np.abs(h_t))
        peak_index_white = np.argmax(np.abs(h_t_white))

        shift = target_index - peak_index
        shift_white = target_index - peak_index_white

        h_t_roll = np.roll(h_t, shift)
        h_t_white_roll = np.roll(h_t_white, shift_white)

        index = target_index + int(0.05 * sampling_frequency)
        index = min(index, N)

        fade_len = N - index
        if fade_len > 0:
            hann_full = np.hanning(2 * fade_len)
            fade_window = hann_full[fade_len:]

            h_t_roll[index:index + fade_len] *= fade_window
            h_t_white_roll[index:index + fade_len] *= fade_window

            if index + fade_len < N:
                h_t_roll[index + fade_len:] = 0
                h_t_white_roll[index + fade_len:] = 0

        d_t_new = h_t_roll + n_t
        d_t_white_new = h_t_white_roll + n_t_white

        test_t_mm_2[i, j, :] = t_array
        test_data_whiten_mm_2[i, j, :] = d_t_white_new
        test_h_whiten_mm_2[i, j, :] = h_t_white_roll

    test_snr_net_mm_2[i] = np.sqrt(snr_sq_network)

test_data_whiten_mm_2.flush()
test_h_whiten_mm_2.flush()
test_t_mm_2.flush()
test_snr_single_mm_2.flush()
test_snr_net_mm_2.flush()

test_final_data_2 = os.path.join(SAVE_DIR, "test_SIS_data_strain_2.npy")
test_final_h_2 = os.path.join(SAVE_DIR, "test_SIS_h_strain_2.npy")
test_final_t_2 = os.path.join(SAVE_DIR, "test_SIS_time_array_2.npy")
test_final_snr_single_2 = os.path.join(SAVE_DIR, "test_SIS_optimal_SNR_single_2.npy")
test_final_snr_net_2 = os.path.join(SAVE_DIR, "test_SIS_optimal_SNR_network_2.npy")

for p in [test_final_data_2, test_final_h_2, test_final_t_2, test_final_snr_single_2, test_final_snr_net_2]:
    if os.path.exists(p):
        os.remove(p)

shutil.move(test_tmp_data_2, test_final_data_2)
shutil.move(test_tmp_h_2, test_final_h_2)
shutil.move(test_tmp_t_2, test_final_t_2)
shutil.move(test_tmp_snr_single_2, test_final_snr_single_2)
shutil.move(test_tmp_snr_net_2, test_final_snr_net_2)

# d1t = np.load(os.path.join(SAVE_DIR, "test_SIS_data_strain_1.npy"))
# t1t = np.load(os.path.join(SAVE_DIR, "test_SIS_time_array_1.npy"))
# h1t = np.load(os.path.join(SAVE_DIR, "test_SIS_h_strain_1.npy"))
# snr1t = np.load(os.path.join(SAVE_DIR, "test_SIS_optimal_SNR_1.npy"))
# d2t = np.load(os.path.join(SAVE_DIR, "test_SIS_data_strain_2.npy"))
# t2t = np.load(os.path.join(SAVE_DIR, "test_SIS_time_array_2.npy"))
# h2t = np.load(os.path.join(SAVE_DIR, "test_SIS_h_strain_2.npy"))
# snr2t = np.load(os.path.join(SAVE_DIR, "test_SIS_optimal_SNR_2.npy"))
# d1 = np.load(os.path.join(SAVE_DIR, "SIS_data_strain_1.npy"))
# t1 = np.load(os.path.join(SAVE_DIR, "SIS_time_array_1.npy"))
# h1 = np.load(os.path.join(SAVE_DIR, "SIS_h_strain_1.npy"))
# snr1 = np.load(os.path.join(SAVE_DIR, "SIS_optimal_SNR_1.npy"))
# d2 = np.load(os.path.join(SAVE_DIR, "SIS_data_strain_2.npy"))
# t2 = np.load(os.path.join(SAVE_DIR, "SIS_time_array_2.npy"))
# h2 = np.load(os.path.join(SAVE_DIR, "SIS_h_strain_2.npy"))
# snr2 = np.load(os.path.join(SAVE_DIR, "SIS_optimal_SNR_2.npy"))
# i = 0
# j = 95000
# print(snr1[i], snr1t[i])
# print(snr2[i], snr2t[i])
# plt.figure()
# plt.plot(t1[i][j:], d1[i][j:])
# plt.plot(t1[i][j:], h1[i][j:])
# plt.title('lensed image 1, SNR=' + str(round(snr1[i], 2)))
# plt.figure()
# plt.plot(t1t[i][j:], d1t[i][j:])
# plt.plot(t1t[i][j:], h1t[i][j:])
# plt.title('test unlensed like image 1, SNR=' + str(round(snr1t[i], 2)))
# plt.figure()
# plt.plot(t2[i][j:], d2[i][j:])
# plt.plot(t2[i][j:], h2[i][j:])
# plt.title('lensed image 2, SNR=' + str(round(snr2[i], 2)))
# plt.figure()
# plt.plot(t2t[i][j:], d2t[i][j:])
# plt.plot(t2t[i][j:], h2t[i][j:])
# plt.title('test unlensed like image 2, SNR=' + str(round(snr2t[i], 2)))
# # In[ ]:
