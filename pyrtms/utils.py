import os.path

import numpy as np

def peak_at(spectrum: np.ndarray, mz: float, **kwargs) -> np.ndarray:
    """
    Find the highest peak within a given tolerance around a specified m/z value.
    """
    tol = kwargs.get("tol", 10)
    min_intensity = kwargs.get('min_intensity', 0)

    if tol < 1:
        # interpret tol as Daltons when it is less than 1
        mz_tol = tol
    else:
        # interpret tol as ppm when it is greater than 1
        mz_tol = mz * tol / 1e6

    # find the peak within the specified mass tolerance
    mask1 =  (spectrum[:, 0] >= mz - mz_tol/2) & (spectrum[:, 0] <= mz + mz_tol/2)

    # find the peak with intensity greater than min_intensity
    mask2 = spectrum[:, 1] > min_intensity

    # combine the masks
    mask = mask1 & mask2

    # if there are peaks in the specified range, return the one with the highest intensityq
    if np.sum(mask) > 0:
        return spectrum[mask][np.argmax(spectrum[mask][:, 1])]
    else:
        return np.array([np.nan, np.nan])

def get_mirror_folder(d_path, config):
    if os.path.exists(config.mirror_folder):
        d_name = os.path.basename(d_path)
        mirror_folder = os.path.join(config.mirror_folder, d_name)
        if not os.path.exists(mirror_folder):
            os.makedirs(mirror_folder)
        return mirror_folder
    else:
        return d_path