from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from pyopenms import MSSpectrum, PeakPickerHiRes, MSExperiment, InternalCalibration, InternalCalibration_LockMass, CalibrationData
import multiprocessing
try:
    from pyrtms.utils import get_mirror_folder
except ImportError:
    from utils import get_mirror_folder

try:
    from pyrtms.rtmsBrukerMCFReader import RtmsBrukerMCFReader
except ImportError:
    from rtmsBrukerMCFReader import RtmsBrukerMCFReader
import plotly.express as px
import plotly.graph_objects as go
import tqdm
from dataclasses import dataclass

import itertools
import numpy as np
from KDEpy import FFTKDE
from scipy.optimize import curve_fit
import os
import re

@dataclass
class Config:
    n_jobs: int = 8
    max_threads_reading_MCF: int = 10
    peak_width: float = 0.001
    peak_snr: float = 0.5
    sampling_points: int = 102400
    mz_decimal_points: int = 4

    initial_da_tol: float = 0.01
    warning_da_tol = 0.007
    final_da_std = 3

    save_line_spectra: bool = True
    line_spectra_filename: str = "line_spectra.npz"
    xy_spots_filename: str = "xy_spots.npz"
    kde_curve_filename: str = "kde_curve.html"

    create_fake_msexp: bool = True # to generate a pyopenms msexperiment object, used for internal calibration for now
    internal_calibration: bool = True

    mirror_folder: str = r"D:/14TC"


def main(d_path):
    config = Config()

    if not os.path.exists(os.path.join(d_path, config.line_spectra_filename)):
        profile_to_line(d_path, config)

    x_val, y_val = get_kde_curve(d_path, config)

    measured_mz, measured_da_tol = get_target_mz(480, x_val, y_val, config)
    picked_results = get_eic(measured_mz, measured_da_tol, d_path, config)
    return picked_results


def profile_to_line(d_path, config: Config):
    spectra = RtmsBrukerMCFReader.from_dir(d_path)
    xy_spots = spectra.xy
    specs = []
    try:
        raw_mzs = spectra.get_spectrum(0, return_mzs=True)[0]
    except IndexError:
        raise IndexError(f"{d_path} is corrupt")

    CASI_mask = (raw_mzs >= spectra.q1mass - spectra.q1res / 2) & (raw_mzs <= spectra.q1mass + spectra.q1res / 2)

    def process_one(i):
        spec = spectra.get_spectrum(i, return_mzs=False)
        return spec[1][CASI_mask]

    with ThreadPoolExecutor(max_workers=config.max_threads_reading_MCF) as executor:
        specs = list(tqdm.tqdm(
            executor.map(process_one, range(len(spectra))),
            total=len(spectra)
        ))

    mzs = raw_mzs[CASI_mask]

    with multiprocessing.Pool(config.n_jobs) as pool:
        results = pool.starmap(process_spectrum, zip(itertools.repeat(mzs), specs, itertools.repeat(config)))

    if config.create_fake_msexp: # to generate a pyopenms msexperiment object, used for internal calibration for now
        msexp = create_msexp(results)

    if config.internal_calibration:
        calibration = InternalCalibration()

        lockmass = [InternalCalibration_LockMass(1314.2264, 1, 1), InternalCalibration_LockMass(1319.9995, 1, 1),InternalCalibration_LockMass(1324.3046, 1, 1)]

        calibration.fillCalibrants(msexp, lockmass, 50.0, False, False, CalibrationData(), True)

        calibration.calibrate(msexp, [1], 2, 100.0, False, 10,10, '','','','','/usr/local/bin/R')

        new_results = []

        for i in range(len(results)):
            new_results.append(msexp[i].get_peaks())

        results = new_results

    if config.save_line_spectra:
        # get the parent path of d_path:
        try:
            np.savez(os.path.join(d_path,config.line_spectra_filename), *results)
            np.savez_compressed(os.path.join(d_path, config.xy_spots_filename), xy_spots)
            print(f'Line spectra saved to {os.path.join(d_path, config.line_spectra_filename)}')
        except PermissionError:
            np.savez(os.path.join(get_mirror_folder(d_path,config),config.line_spectra_filename), *results)
            np.savez_compressed(os.path.join(get_mirror_folder(d_path,config),config.xy_spots_filename), xy_spots)
            print(f'Line spectra saved to {os.path.join(get_mirror_folder(d_path,config),config.line_spectra_filename)}')

def create_msexp(results: List) -> MSExperiment:
    exp = MSExperiment()
    for i, result in enumerate(results):
        spectrum = MSSpectrum()
        spectrum.setRT(i)
        spectrum.setMSLevel(1)
        spectrum.set_peaks([result[0], result[1]])
        exp.addSpectrum(spectrum)
    return exp

def process_spectrum(profile_mz, profile_intensity, config: Config):
    # sort spec by first column
    spec_obj = MSSpectrum()
    spec_obj.set_peaks([profile_mz, profile_intensity])

    picker = PeakPickerHiRes()
    picker_params = picker.getParameters()
    picker_params.setValue("signal_to_noise", config.peak_snr)
    picker.setParameters(picker_params)

    res_spec = MSSpectrum()

    picker.pick(spec_obj, res_spec)

    mz, intensity = res_spec.get_peaks()

    # only keep the intensity > 100_000
    mask = intensity > 100_000

    return mz[mask], intensity[mask]

def get_kde_curve(d_path, config: Config):
    line_spectra = load_line_spectra(d_path, config)
    mzs_all = [
        mzs
        for key in line_spectra.files
        for mzs in line_spectra[key][0]
    ]
    mzs_all = np.sort(mzs_all)
    np.round(mzs_all, config.mz_decimal_points, out=mzs_all)

    x_val, y_val = FFTKDE(kernel='gaussian', bw=config.peak_width).fit(mzs_all).evaluate(grid_points=config.sampling_points)

    fig = px.line(
        x=x_val,
        y=y_val,
    )

    try:
        fig.write_html(
            os.path.join(d_path, config.kde_curve_filename),
        )
    except:
        fig.write_html(
            os.path.join(get_mirror_folder(d_path,config),config.kde_curve_filename),
        )

    return x_val, y_val

def get_target_mz(target_mz, x_val, y_val, config: Config, plot=False):
    target_mz_mask = (x_val > target_mz - Config.initial_da_tol) & (x_val < target_mz + config.initial_da_tol)
    filtered_y_val = y_val[target_mz_mask]
    filtered_y_val = (filtered_y_val - np.min(filtered_y_val)) / (np.max(filtered_y_val) - np.min(filtered_y_val))
    filtered_x_val = x_val[target_mz_mask]
    popt, pcov = curve_fit(gaussian, filtered_x_val, filtered_y_val, p0=[target_mz, config.initial_da_tol])

    measured_mz = popt[0]
    measured_da_tol = config.final_da_std * abs(popt[1])
    if measured_da_tol >= config.warning_da_tol:
        print("Peak is too wide")
    if plot:
        fig = px.line(
            x=filtered_x_val,
            y=filtered_y_val,
        )
        fig.add_trace(
            go.Scatter(
                x=filtered_x_val,
                y=gaussian(filtered_x_val, *popt),
                mode='lines',  # Specify the mode
                name='Gaussian Fit'  # Assign a name for the legend
            )
        )
        fig.add_vline(
            x=measured_mz,
            line_width=3,
            annotation_text=f"Measured m/z {round(measured_mz, config.mz_decimal_points)}"
        )
        fig.add_vline(
            x=measured_mz - measured_da_tol,
            line_width=3,
            annotation_text=f"Lower m/z boundary {round(measured_mz - measured_da_tol, config.mz_decimal_points)}"
        )
        fig.add_vline(
            x=measured_mz + measured_da_tol,
            line_width=3,
            annotation_text=f"Higher m/z boundary {round(measured_mz + measured_da_tol, config.mz_decimal_points)}"
        )
        return measured_mz, measured_da_tol, fig
    return measured_mz, measured_da_tol

def get_eic(measured_mz, measured_da_tol, d_path, config: Config):
    picked_results = []
    line_spectra = load_line_spectra(d_path, config)
    for key in line_spectra.files:
        result = line_spectra[key]
        mask = (result[0] > measured_mz - measured_da_tol) & (result[0] < measured_mz + measured_da_tol)
        # if there is more than one
        if len(mask[mask]) > 1:
            second_idx = np.argmax(result[1][mask])
            picked_results.append([result[0][mask][second_idx], result[1][mask][second_idx]])
        elif len(mask[mask]) == 1:
            picked_results.append([result[0][mask][0], result[1][mask][0]])
        else:
            picked_results.append([np.nan, np.nan])
    picked_results = np.array(picked_results)
    return picked_results

def gaussian(x, mean, stddev):
    return np.exp(-((x - mean) / stddev)**2 / 2)

def load_line_spectra(d_path, config):
    if os.path.exists(os.path.join(d_path, config.line_spectra_filename)):
        line_spectra = np.load(os.path.join(d_path, config.line_spectra_filename))
    elif os.path.exists(os.path.join(get_mirror_folder(d_path,config),config.line_spectra_filename)):
        line_spectra = np.load(os.path.join(get_mirror_folder(d_path,config),config.line_spectra_filename))
    else:
        raise ValueError('Could not find line spectra')
    return line_spectra

Element = {"C": 12.00000,
           "H" : 1.00783,
           "O" : 15.99491,
           "Na" : 22.98977,
           "NH4": 18.03437,
           'N': 14.00307,
           'S': 31.97207,
           'K': 38.96371,
           'P': 30.97376,
           "-" : 0.00055,
           "+": -0.00055}


def init_mz(formula: str) -> float:
    if formula.endswith("+"):
        mz = Element['+']
    elif formula.endswith("-"):
        mz = Element['-']
    else:
        mz = 0.0
    return mz

def simple_mz_calc(formula: str):

    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    if not matches:
        raise ValueError("Invalid formula")

    mz = init_mz(formula)

    for elem, count in matches:
        if elem not in Element:
            raise ValueError(f"Unknown element: {elem}")
        n = int(count) if count else 1
        mz += Element[elem] * n

    return mz

def load_xy_spots(d_path, config):
    if os.path.exists(os.path.join(d_path, config.xy_spots_filename)):
        xy_spots = np.load(os.path.join(d_path, config.xy_spots_filename))['arr_0']
    elif os.path.exists(os.path.join(get_mirror_folder(d_path,config),config.xy_spots_filename)):
        xy_spots = np.load(os.path.join(get_mirror_folder(d_path,config),config.xy_spots_filename))['arr_0']
    else:
        raise ValueError('Could not find xy_spots')
    return xy_spots

if __name__ == "__main__":
    pass




