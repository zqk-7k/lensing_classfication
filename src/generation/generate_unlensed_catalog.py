#!/usr/bin/env python3
"""Generate the unlensed gravitational-wave comparison catalog."""

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
SAVE_DIR = os.environ.get("GW_UNLENSED_DATA_DIR", str(PROJECT_ROOT / "data" / "ligo_full" / "Unlensed_data"))

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
n = 61301
# n=613





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




sampling_frequency = 4096  # f_s >= 2*f_max   #
duration = 24  # len(strain_data) = f_s*duration
minimum_frequency = 20.0

# # 1. Source generation
# ## All GW sources



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

# # 2. GW data

# ## ALL unlensed GW data
# #



bilby.core.utils.random.seed(n + 1)

N = sampling_frequency * duration

waveform_arguments = dict(waveform_approximant='IMRPhenomXPHM',
                          reference_frequency=10., minimum_frequency=minimum_frequency)

# unlensed_waveform
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)

source_params = pd.read_csv(os.path.join(SAVE_DIR, 'source_samples.csv'))
n_events = len(source_params)


# LIGO
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])  # implementation detail
ifo_names = [ifo.name for ifo in ifos]
n_ifos = len(ifos)

# -------------------------
# -------------------------
tmp_data = os.path.join(tmp_dir, "unlensed_data_strain.tmp.npy")
tmp_h    = os.path.join(tmp_dir, "unlensed_h_strain.tmp.npy")
tmp_t    = os.path.join(tmp_dir, "unlensed_time_array.tmp.npy")
tmp_snr_single = os.path.join(tmp_dir, "unlensed_optimal_SNR_single.tmp.npy")
tmp_snr_net    = os.path.join(tmp_dir, "unlensed_optimal_SNR_network.tmp.npy")

for p in [tmp_data, tmp_h, tmp_t, tmp_snr_single, tmp_snr_net]:
    if os.path.exists(p):
        os.remove(p)

# shape(n_events, n_ifos, N)
data_whiten_mm = np.lib.format.open_memmap(
    tmp_data, mode="w+", dtype=np.float64, shape=(n_events, n_ifos, N))

h_whiten_mm = np.lib.format.open_memmap(
    tmp_h, mode="w+", dtype=np.float64, shape=(n_events, n_ifos, N))

t_mm = np.lib.format.open_memmap(
    tmp_t, mode="w+", dtype=np.float64, shape=(n_events, n_ifos, N))

snr_single_mm = np.lib.format.open_memmap(
    tmp_snr_single, mode="w+", dtype=np.float64, shape=(n_events, n_ifos))  # implementation detail

snr_net_mm = np.lib.format.open_memmap(
    tmp_snr_net, mode="w+", dtype=np.float64, shape=(n_events,))  # implementation detail




for i in range(n_events):

    injection_parameters = source_params.iloc[i].to_dict()

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

        snr_single_mm[i, j] = snr
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

        t_target = injection_parameters["geocent_time"] + 0.05
        index = np.argmin(np.abs(t_array - t_target))

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

        t_mm[i, j, :] = t_array
        data_whiten_mm[i, j, :] = d_t_white_new
        h_whiten_mm[i, j, :] = h_t_white_roll

    snr_net_mm[i] = np.sqrt(snr_sq_network)

# flush
data_whiten_mm.flush()
h_whiten_mm.flush()
t_mm.flush()
snr_single_mm.flush()
snr_net_mm.flush()

final_data = os.path.join(SAVE_DIR, "unlensed_data_strain.npy")
final_h = os.path.join(SAVE_DIR, "unlensed_h_strain.npy")
final_t = os.path.join(SAVE_DIR, "unlensed_time_array.npy")
final_snr_single = os.path.join(SAVE_DIR, "unlensed_optimal_SNR_single.npy")
final_snr_net    = os.path.join(SAVE_DIR, "unlensed_optimal_SNR_network.npy")

for p in [final_data, final_h, final_t, final_snr_single, final_snr_net]:
    if os.path.exists(p):
        os.remove(p)

shutil.move(tmp_data, final_data)
shutil.move(tmp_h, final_h)
shutil.move(tmp_t, final_t)
shutil.move(tmp_snr_single, final_snr_single)
shutil.move(tmp_snr_net, final_snr_net)
