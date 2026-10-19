import mmap
import os
import sqlite3

import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Callable, Any

import struct
import threading
from collections import OrderedDict
import zlib
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pyopenms import MSSpectrum, PeakPickerHiRes
from functools import partial, cached_property

try:
    from pyrtms.utils import peak_at
except ImportError:
    from utils import peak_at
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyopenms")
import matplotlib.pyplot as plt
import h5py


CalibCoefficients: type = tuple[float, float, float, float]
BLOB_CODE_WORD = bytes([0xC0, 0xDE, 0xAF, 0xFE])

# number of distinct m/z axes (calibrations) kept per reader
MZ_CACHE_SIZE = 8
# create lock for caching m/z entries across threads
_MZ_CACHE_LOCK = threading.Lock()


class _MMapCursor:
    """Minimal file-like reader over a shared buffer with its own position."""

    # to reduce memory and don't allow attribute assignment
    __slots__ = ("buf", "pos")

    def __init__(self, buf):
        self.buf = buf
        self.pos: int = 0

    def read(self, n=-1):
        """Read n bytes fromt the buffer, starting at the current position and moving the position forward."""
        end = len(self.buf) if n < 0 else self.pos + n
        data = self.buf[self.pos:end]
        self.pos += len(data)
        return data

    def seek(self, offset: int, whence=0):
        """Move the position to the given offeset (absolute if whence == 0, relative if whence == 1, or from the end if whence == 2)."""
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = len(self.buf) + offset
        return self.pos

    def tell(self) -> int:
        """Return the current position."""
        return self.pos


def write_intensities_to_hdf5(data:List[Tuple[np.array, np.array]], path, **kwargs):
    if os.path.exists(path):
        raise FileExistsError(path)
    with h5py.File(path, "w") as f:
        dset = f.create_dataset(
            'intensities',
            shape=(len(data), len(data[0][0])),
            dtype='float64',
            chunks=(1, len(data[0][0])),
            kwargs=kwargs,
        )
        for i, spectrum in enumerate(data):
            dset[i, :] = spectrum[1]


# reading is limited by disk access, more threads than this do not help
MAX_READ_THREADS = 16

# reader instance of a worker process, set once by the pool initializer instead of pickling it with every task
_worker_reader: 'RtmsBrukerMCFReader' = None


def _init_worker(reader: 'RtmsBrukerMCFReader'):
    """Initialize a worker"""
    global _worker_reader
    _worker_reader = reader


def _pick_spectrum_worker(args):
    index, line_spec, kwargs = args
    return _worker_reader.pick_spectrum(index, line_spec=line_spec, **kwargs)


class BatchProcessor:
    def __init__(self, reader, **kwargs):
        self.reader = reader
        self.n_jobs = min(kwargs.get("n_jobs", mp.cpu_count()), 60)
        self.show_progress = kwargs.get("show_progress", True)

    def _map(self, func, args, initializer=None, initargs=()):
        """pool.imap in chunks, with optional progress bar"""
        args = list(args)
        chunksize = max(1, len(args) // (self.n_jobs * 4))
        with mp.Pool(self.n_jobs, initializer=initializer, initargs=initargs) as pool:
            results = pool.imap(func, args, chunksize=chunksize)
            if self.show_progress:
                results = tqdm(results, total=len(args))
            return list(results)

    def _pick_map(self, args):
        return self._map(_pick_spectrum_worker, args, initializer=_init_worker, initargs=(self.reader,))

    def calib_all_spectra(self, **kwargs) -> list:
        """
        calibrate all spectra in the spotTable
        """
        uncalib_line_spectra = self.pick_mul_spectra(self.reader.spotTable.index, **kwargs)
        f = partial(simple_calibration, **kwargs, return_shift=True)
        shifts = self._map(f, uncalib_line_spectra)
        self.reader.shifts = np.array(shifts)
        self.reader.mean_shift = np.nanmean(self.reader.shifts)
        return uncalib_line_spectra

    def get_mul_spectra(self, indices, **kwargs):
        """
        Read spectra with threads: reading is I/O bound and releases the GIL, and unlike
        processes no spectra have to be pickled and the cached m/z axes are shared.
        """
        indices = np.asarray(indices)
        results = [None] * len(indices)

        def read(k):
            results[k] = self.reader.get_spectrum(indices[k], **kwargs)

        # read in file order for mostly sequential disk access, results keep the requested order
        order = np.argsort(self.reader._raw_positions[indices], kind="stable")
        with ThreadPoolExecutor(min(self.n_jobs, MAX_READ_THREADS)) as executor:
            done = executor.map(read, order)
            if self.show_progress:
                done = tqdm(done, total=len(order))
            for _ in done:
                pass
        return results

    def pick_mul_spectra(self, indices, **kwargs):
        return self._pick_map((index, None, kwargs) for index in indices)

    def pick_mul_from_uncalib_line_spectra(self, uncalib_line_spectra, **kwargs):

        indices = self.reader.spotTable.index

        # calibrate the line spectra
        if hasattr(self.reader, "mean_shift"):
            for i in range(len(uncalib_line_spectra)):
                if np.isnan(self.reader.shifts[i]):
                    uncalib_line_spectra[i][:, 0] -= self.reader.mean_shift
                else:
                    uncalib_line_spectra[i][:, 0] -= self.reader.shifts[i]

        return self._pick_map((None, uncalib_line_spectra[i], kwargs) for i in indices)


class RtmsBrukerMCFReader:
    def __init__(self, mcfdir: str, files: list[str], spotTable: pd.DataFrame, offsetTable: pd.DataFrame, metadata: dict) -> None:
        assert isinstance(mcfdir, str), "dir must be a string"
        assert isinstance(files, list), "files must be a dictionary"
        assert isinstance(spotTable, pd.DataFrame), "spotTable must be a DataFrame"
        assert isinstance(offsetTable, pd.DataFrame), "offsetTable must be a DataFrame"
        assert isinstance(metadata, dict), "metadata must be a dictionary"

        self.mcfdir: str = mcfdir
        self.files: list[str] = files
        self.spotTable: pd.DataFrame = spotTable
        self.offsetTable: pd.DataFrame = offsetTable
        self.metadata: dict = metadata
        self.spots = None

    def __len__(self) -> int:
        return len(self.get_spots()['SpotNumber'].values)

    @property
    def indices(self) -> np.ndarray[int]:
        return self.spotTable.index.values

    @cached_property
    def mzs(self) -> np.ndarray[float]:
        cols = ["frequencyHigh", "frequencyWidth", "size", "alpha", "beta"]
        if len(self.spotTable[cols].drop_duplicates()) > 1:
            # check if all calibration parameters are the same, if not mz values will not be the same
            raise ValueError("Encountered spectra with non-equally sampled mz values.")
        return self.get_spectrum(0, return_mzs=True)[0]

    @cached_property
    def str_xy(self) -> np.ndarray:
        return self.get_spots()['SpotNumber'].values

    @cached_property
    def xy(self) -> np.ndarray[int]:
        """Create an array where each row containts the spot indices x,y"""
        spot_x: list[str] = [re.findall(r"X(\d+)", str(spot))[0] for spot in self.str_xy]
        spot_x: list[int] = [int(x) for x in spot_x]
        spot_y: list[str] = [re.findall(r"Y(\d+)", str(spot))[0] for spot in self.str_xy]
        spot_y: list[int] = [int(y) for y in spot_y]
        spot: np.ndarray[int] = np.column_stack((spot_x, spot_y))
        return spot

    @cached_property
    def metadataDF(self):
        return self.get_metadata(1)

    @cached_property
    def q1mass(self):
        return float(self.metadataDF.query("PermanentName == 'Q1Mass'")["Value"].values[0].split(' ')[0])

    @cached_property
    def q1res(self):
        return float(self.metadataDF.query("PermanentName == 'Q1Res'")["Value"].values[0].split(' ')[0])

    @property
    def casi_window(self) -> tuple[float, float]:
        return self.q1mass - self.q1res / 2, self.q1mass + self.q1res / 2

    @cached_property
    def _mmap(self) -> mmap.mmap:
        # memory map of the main data file, created lazily once per process
        with open(os.path.join(self.mcfdir, self.files[0]), "rb") as f:
            return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    @property
    def con(self):
        # independent cursor so that concurrent readers do not share a file position
        return _MMapCursor(self._mmap)

    def close(self):
        mm = self.__dict__.pop("_mmap", None)
        if mm is not None:
            mm.close()

    def __getstate__(self):
        # mmap objects cannot be pickled, workers recreate it on first access
        state = self.__dict__.copy()
        state.pop("_mmap", None)
        state.pop("_index_axes", None)
        state.pop("_mz_cache", None)
        return state

    def _blob_positions(self, blob_res_type: int) -> np.ndarray[np.int64]:
        """Absolute file position of the blob of given type for each row in spotTable (-1 if missing)."""
        sub: pd.DataFrame = self.offsetTable[self.offsetTable["BlobResType"] == blob_res_type]
        sub: pd.DataFrame = sub.drop_duplicates("id").set_index("id").reindex(self.spotTable["id"])
        positions: pd.Series = sub["OffsetPage"] * (2 ** 31) + sub["Offset"]
        return positions.fillna(-1).astype(np.int64).to_numpy()

    @cached_property
    def _raw_positions(self) -> np.ndarray[np.int64]:
        return self._blob_positions(258)

    @cached_property
    def _meta_positions(self) -> np.ndarray[np.int64]:
        return self._blob_positions(259)

    @cached_property
    def _calib_params(self) -> dict[str, np.ndarray]:
        # for faster accessing, convert dataframe to dict
        cols: list[str] = ["frequencyHigh", "frequencyWidth", "size", "alpha", "beta"]
        return {col: self.spotTable[col].to_numpy(dtype=float) for col in cols}

    def get_spots(self) -> pd.DataFrame:
        if self.spots is None:
            main_index = self.files[1]

            with sqlite3.connect(os.path.join(self.mcfdir, main_index)) as conn:
                cur = conn.cursor()
                spot_name = cur.execute("SELECT GuidA, GuidB, MetaDataID, Value from MetaDataString WHERE MetaDataID='64'")
                spot_name = spot_name.fetchall()
                self.spots = pd.DataFrame(spot_name, columns=["GuidA", "GuidB", "MetaDataID", "SpotNumber"])
        return self.spots

    @classmethod
    def from_dir(cls, mcfdir: str):
        """Initialize from .d directory by parsing files"""
        files = os.listdir(mcfdir)
        mainFile = [f for f in files if f.endswith('_1.mcf')]
        if len(mainFile) == 0:
            raise Exception('Main data file not found in directory.')
        else:
            mainFile = mainFile[0]
            mainIndex = mainFile + "_idx"
            if mainIndex not in files:
                raise Exception('Main index file not found in directory.')
            calibFile = mainFile.replace("_1.mcf", "_2.mcf")
            if calibFile not in files:
                raise Exception('Calibration data file not found in directory.')
            calibIndex = calibFile + "_idx"
            if calibIndex not in files:
                raise Exception('Calibration index file not found in directory.')
            files = [mainFile, mainIndex, calibFile, calibIndex]
            offsetTable = readBrukerMCFIndexFile(os.path.join(mcfdir, mainIndex), calibration=False)
            calOffsets = readBrukerMCFIndexFile(os.path.join(mcfdir, calibIndex), calibration=True)

            subset_ids = offsetTable.loc[offsetTable["BlobResType"] == 258, ["id"]]
            spotCalOffsets = calOffsets.merge(subset_ids, left_on="toId", right_on="id", how="inner", sort=False)
            spotCalData = readMCFCalibration(os.path.join(mcfdir, calibFile), spotCalOffsets)

            spotTable = spotCalData.sort_index()
            spotTable = spotTable.rename(columns={"toId": "id"})

            mcfcon = open(os.path.join(mcfdir, mainFile), "rb")

            paramBlobOffset = offsetTable["Offset"].iloc[2]

            metadata = retrieveMCFMetadata(mcfcon, paramBlobOffset, 0)
            return cls(
                mcfdir=mcfdir,
                files=files,
                spotTable=spotTable,
                offsetTable=offsetTable,
                metadata=metadata,
            )

    def pick_spectrum(self, index=None, line_spec=None, **kwargs):
        """Pick peaks from spectrum"""

        if line_spec is not None:
            mz = line_spec[:, 0]
            intensity = line_spec[:, 1]
        else:
            try:
                snr = float(kwargs["min_snr"])
            except KeyError:
                raise KeyError("min_snr must be provided for peak picking")
            spec = self.get_spectrum(index)
            # sort spec by first column
            spec_obj = MSSpectrum()
            # pyopenms needs writable arrays, the cached m/z axis is read-only
            spec_obj.set_peaks([np.require(spec[0], requirements="W"), spec[1]])

            picker = PeakPickerHiRes()
            picker_params = picker.getParameters()
            picker_params.setValue("signal_to_noise", float(snr))
            picker.setParameters(picker_params)

            res_spec = MSSpectrum()

            picker.pick(spec_obj, res_spec)

            mz, intensity = res_spec.get_peaks()


        # if 'shift' in kwargs:
        #     mz = mz - kwargs['shift']
        # elif hasattr(self, "mean_shift"):
        #     mz = mz - self.mean_shift

        target_mzs = kwargs.get("target_mzs", None)
        if target_mzs is None:
            return np.column_stack([mz, intensity])
        else:
            try:
                tol = float(kwargs["tol"])
                min_intensity = float(kwargs["min_intensity"])
            except KeyError:
                raise KeyError("tol and min_intensity must be provided for picking specific peaks")
            result = []
            for target_mz in target_mzs:
                if tol > 0.1:
                    # if tol > 0.1, it is considered as ppm
                    mz_tol = target_mz * tol / 1e6
                else:
                    mz_tol = tol
                mask1 = (mz >= target_mz - mz_tol / 2) & (mz <= target_mz + mz_tol / 2)
                mask2 = intensity > min_intensity
                mask = mask1 & mask2
                # find the peak with the highest intensity within the tolerance
                if np.sum(mask) > 0:
                    result.append(
                        [mz[mask][np.argmax(intensity[mask])],
                         max(intensity[mask])]
                    )
                else:
                    result.append([np.nan, np.nan])
            return np.array(result)

    def get_spectrum(
            self,
            index: int,
            CASI_only: bool = False,
            return_mzs: bool = None,
            intensities_only: bool = None,
            limits: tuple[float, float] = None
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """
        Read the profile spectrum with the given (0-based) index.

        Parameters
        ----------
        index : int
            Position of the spectrum in the spotTable.
        CASI_only : bool, optional
            Only return data points within the CASI window (Q1Mass +/- Q1Res / 2).
        return_mzs, intensities_only : bool, optional
            Whether to return the m/z values (otherwise None is returned instead).
        limits : tuple[float, float], optional
            Only return data points with limits[0] <= mz <= limits[1]. Only this
            part of the spectrum is copied, which is much faster for narrow
            windows. Cannot be combined with CASI_only.

        Returns
        -------
        mz, intensities : tuple[np.ndarray | None, np.ndarray]
        """
        if (return_mzs is not None) and (intensities_only is not None):
            raise ValueError("Only one of return_mzs or intensities_only should be provided")
        if CASI_only and (limits is not None):
            raise ValueError("Only one of CASI_only or limits should be provided")

        if return_mzs is None:
            # if return_mzs and intensities_only are none, default to returning mz values
            return_mzs: bool = (not intensities_only) if (intensities_only is not None) else True

        if index < 0 or index >= len(self.spotTable):
            raise IndexError("Index out of bounds")

        spec_start_position: np.int64 = self._raw_positions[index]
        if spec_start_position < 0:
            raise IndexError(f"No raw data blob found for spot {self.spotTable['id'].iloc[index]}")

        # Seek to raw data blob
        fcon: _MMapCursor = self.con
        fcon.seek(spec_start_position)

        mcf_checkBlobCodeWord(fcon)
        bin_checkBytes(fcon, b"\x01\x01")
        fcon.seek(16, 1)
        _ = bin_readVarInt(fcon)

        name = mcf_readNamedName(fcon)
        if name != "Intensities":
            raise ValueError("First named element of raw data blob should be 'Intensities'")

        bin_checkBytes(fcon, b"\x03", "Raw spectra must be an array.")
        typeByte = fcon.read(1)

        if typeByte == b"\x20":  # uncompressed 32-bit floats
            num_values: int = bin_readVarInt(fcon)
            # view into the mmap, only the returned part is copied below
            raw: np.ndarray = np.frombuffer(self._mmap, dtype="<f4", count=num_values, offset=fcon.tell())
        elif typeByte == b"\x22":  # gzip-compressed
            num_bytes = bin_readVarInt(fcon)
            gzipped_bytes = fcon.read(num_bytes)
            unzipped = zlib.decompress(gzipped_bytes)
            raw: np.ndarray = np.frombuffer(unzipped, dtype="<f4")
            num_values: int = raw.shape[0]
        else:
            raise ValueError("Raw spectra must be an array of 32-bit floats.")

        if CASI_only:
            # calculate mz limits from Q1 mass and Q1 resolution and set limits from CASI window
            limits: tuple[float, float] = (self.q1mass - self.q1res / 2, self.q1mass + self.q1res / 2)

        if limits is None:
            if not return_mzs:
                # if return_mzs is False, return the spectrum only
                return None, raw.copy()
            # m/z axes are shared between spectra with identical calibration and therefore read-only
            calib_coeffs: CalibCoefficients = self._get_calib_coefficients(index)
            # get mz values corresponding to limits and calibration parameters
            mzs: np.ndarray[np.float64] = self._cached_mz(
                ("full", calib_coeffs, num_values),
                lambda: self._mz_axis(calib_coeffs, num_values, 0, num_values)
            )
            return mzs, raw.copy()

        # only copy the data points within the window, the selection depends on the
        # calibration and is cached together with the m/z values
        mz_lower, mz_upper = float(limits[0]), float(limits[1])
        calib_coeffs: CalibCoefficients = self._get_calib_coefficients(index)
        mzs: np.ndarray[np.float64]
        selection, mzs = self._cached_mz(
            ("window", calib_coeffs, num_values, mz_lower, mz_upper),
            lambda: self._casi_mz_axis(calib_coeffs, num_values, mz_lower, mz_upper)
        )
        return (mzs if return_mzs else None), raw[selection].copy()

    def _cached_mz(
            self,
            key: tuple[str, CalibCoefficients, int, float, float] | tuple[str, CalibCoefficients, int],
            callback: Callable
    ):
        """
        Small LRU cache for m/z axes, the returned arrays are made read-only.

        If the specified key is not in the cache, will assign the returned value of callback
        """
        # self.__dict__.setdefault("_mz_cache", OrderedDict())
        if "_mz_cache" not in self.__dict__:
            self._mz_cache: OrderedDict[tuple[str, CalibCoefficients, int, float, float], Any] = OrderedDict()
        mz_cache: OrderedDict[tuple[str, CalibCoefficients, int, float, float], Any] = self._mz_cache

        with _MZ_CACHE_LOCK:  # make sure no other thread operates on the cache at the same time
            value = mz_cache.get(key)
            if value is not None:
                mz_cache.move_to_end(key)
                return value
        value = callback()
        for arr in (value if isinstance(value, tuple) else (value,)):
            if isinstance(arr, np.ndarray):
                arr.flags.writeable = False
        with _MZ_CACHE_LOCK:
            # another thread may have computed the same axis meanwhile, share its result
            value = mz_cache.setdefault(key, value)
            while len(mz_cache) > MZ_CACHE_SIZE:
                mz_cache.popitem(last=False)
        return value

    def _casi_mz_axis(
            self,
            calib_coeffs: CalibCoefficients,
            num_values: int,
            mz_lower: float,
            mz_upper: float
    ) -> tuple[slice | np.ndarray[int], np.ndarray[np.float64]]:
        """Selection of data points with lower <= mz <= upper and their m/z values."""
        window: tuple[int, int] = self._get_mz_index_window(calib_coeffs, num_values, mz_lower, mz_upper)
        if window is not None:
            start, stop = window
            mzs: np.ndarray[np.float64] = self._mz_axis(calib_coeffs, num_values, start, stop)
            # trim the (slightly generous) window to the exact bounds
            lo: int = start + np.searchsorted(mzs, mz_lower, side="left")
            hi: int = start + np.searchsorted(mzs, mz_upper, side="right")
            # make sure the window was wide enough, otherwise fall through to the full axis
            if (lo > start or start == 0) and (hi < stop or stop == num_values):
                return slice(lo, hi), mzs[lo - start:hi - start].copy()

        mzs: np.ndarray[np.float64] = self._mz_axis(calib_coeffs, num_values, 0, num_values)
        # indices where mask is non-zero
        selection: np.ndarray[int] = np.flatnonzero((mzs >= mz_lower) & (mzs <= mz_upper))
        return selection, mzs[selection]

    def _get_calib_coefficients(self, index: int) -> CalibCoefficients:
        """
        Calibrated m/z of data point i is
            fhigh / ((fwidth * (fsize - i) / fsize) - beta) + alpha / fhigh
            = fhigh / (offset - slope * i) + shift
        """
        params: dict[str, np.ndarray] = self._calib_params
        fhigh = params["frequencyHigh"][index]
        fwidth = params["frequencyWidth"][index]
        fsize = params["size"][index]
        alpha = params["alpha"][index]
        beta = params["beta"][index]
        return float(fhigh), float(fwidth / fsize), float(fwidth - beta), float(alpha / fhigh)

    def _index_axis(self, num_values: int) -> np.ndarray[np.float64]:
        """Getter for mz values container with a certain number of values."""
        # float data point indices, cached since np.arange is the slowest step of the m/z calculation
        axes: dict[int, np.ndarray[np.float64]] = self.__dict__.setdefault("_index_axes", {})
        axis: np.ndarray[np.float64] | None = axes.get(num_values)
        if axis is None:
            axis: np.ndarray[np.float64] = np.arange(num_values, dtype=np.float64)
            axes[num_values] = axis
        return axis

    def _mz_axis(
            self,
            coeffs: CalibCoefficients,
            num_values: int,
            mz_index_start: int,
            mz_index_stop: int
    ) -> np.ndarray[np.float64]:
        """m/z values of data points start..stop-1, evaluated in place to avoid temporaries."""
        fhigh, slope, offset, shift = coeffs
        mz: np.ndarray[np.float64] = np.multiply(self._index_axis(num_values)[mz_index_start:mz_index_stop], -slope)
        mz += offset
        np.divide(fhigh, mz, out=mz)
        mz += shift
        return mz

    @staticmethod
    def _get_mz_index_window(calib_coeffs: CalibCoefficients, num_values: int, mz_lower, mz_upper) -> tuple[int, int] | None:
        """
        Index range covering lower <= mz <= upper with a small margin, obtained by inverting the
        calibration. Returns None if the m/z axis is not strictly increasing.
        """
        fhigh, slope, offset, shift = calib_coeffs
        # mz increases with i if fhigh, slope > 0 and the denominator stays positive
        if not (fhigh > 0 and slope > 0 and offset - slope * (num_values - 1) > 0):
            return None

        def index_of(mz: float) -> float:
            if mz <= shift:
                return -np.inf
            return (offset - fhigh / (mz - shift)) / slope

        # add margin to indices to account for post-hoc calibration shifts
        margin: int = 2
        start: int = int(np.clip(np.floor(index_of(mz_lower)) - margin, 0, num_values))
        stop: int = int(np.clip(np.ceil(index_of(mz_upper)) + margin + 1, start, num_values))
        return start, stop

    def get_metadata(self, index) -> pd.DataFrame:
        position = self._meta_positions[index]
        if position < 0:
            raise IndexError(f"No metadata blob found for spot {self.spotTable['id'].iloc[index]}")

        dectable = self.metadata["declarations"]
        reptable = self.metadata["replacements"]

        fcon = self.con
        fcon.seek(position)

        mcf_checkBlobCodeWord(fcon)
        bin_checkBytes(fcon, b"\x01\x01")
        fcon.seek(16, 1)
        numNamedEls = bin_readVarInt(fcon)

        allMeta = []

        for _ in range(numNamedEls):
            name = mcf_readNamedName(fcon)
            if fcon.read(1) == b"\x00":
                continue
            else:
                fcon.seek(-1, 1)

            keyTable = mcf_readKeyValueTable(fcon)

            if name == "IntValues":
                keyTable = keyTable.rename(columns={"Value": "Code"})
                keyTable = keyTable.merge(reptable, how="left", on=["Key", "Code"])
                keyTable["Value"] = keyTable["Value"].fillna(keyTable["Code"].astype(str))
                keyTable = keyTable.drop(columns="Code")
                keyTable = dectable.merge(keyTable, how="right", on="Key")

            elif name == "StringValues":
                keyTable = dectable.merge(keyTable, how="right", on="Key")

            elif name == "DoubleValues":
                keyTable = dectable.merge(keyTable, how="right", on="Key")
                keyTable["Value"] = keyTable.apply(
                    lambda row: row["DisplayFormat"] % row["Value"] if "DisplayFormat" in row else str(
                        row["Value"]),
                    axis=1
                )
            else:
                raise ValueError("Other metadata value types not supported.")

            allMeta.append(keyTable)

        metaTable = pd.concat(allMeta, ignore_index=True)
        metaTable["Index"] = index

        # Append units to values
        rel = metaTable["Unit"] != ""
        metaTable.loc[rel, "Value"] = metaTable.loc[rel, "Value"] + " " + metaTable.loc[rel, "Unit"]

        return metaTable[["Index", "PermanentName", "DisplayName", "GroupName", "Value"]]


def mcf_checkBlobCodeWord(fcon):
    expected = BLOB_CODE_WORD
    bin_checkBytes(fcon, expected, "Invalid code word.")


def bin_checkBytes(fcon, expected, message="Invalid bytes"):
    actual = fcon.read(len(expected))
    if actual != expected:
        raise ValueError(message)


def mcf_readNamedName(fcon):
    first_byte = int.from_bytes(fcon.read(1), byteorder="big")
    if first_byte <= 127:
        raise ValueError("Invalid object name.")

    bin_checkBytes(fcon, b"\xFF\xFF\xFF\x0F", "Invalid object name.")

    namelen = int.from_bytes(fcon.read(1), byteorder="big")
    name_bytes = fcon.read(namelen)
    return name_bytes.decode("ascii")


def bin_readVarInt(fcon, endian='little'):
    nums = []
    for iter in range(9):
        nextByte = int.from_bytes(fcon.read(1), byteorder='big', signed=False)
        if nextByte > 127:
            nextNum = nextByte - 128
        else:
            nextNum = nextByte
        if endian == "little":
            nums.append(nextNum)
        else:
            nums.insert(0, nextNum)
        if nextByte <= 127:
            break
    return sum(num * (128 ** i) for i, num in enumerate(nums))


def mcf_readPrimitive(fcon):
    dataType = int.from_bytes(fcon.read(1), byteorder="big")

    if dataType in (0x27, 0x28):  # 8-byte integer
        return int.from_bytes(fcon.read(8), byteorder="little", signed=True)

    elif dataType in (0x25, 0x26):  # 4-byte integer
        return int.from_bytes(fcon.read(4), byteorder="little", signed=True)

    elif dataType == 0x20:  # 4-byte float
        return struct.unpack("<f", fcon.read(4))[0]

    elif dataType == 0x1F:  # 8-byte float
        return struct.unpack("<d", fcon.read(8))[0]

    elif dataType == 0x00:
        return None

    elif dataType == 0x2A:  # ASCII string with varint length
        stringLength = bin_readVarInt(fcon)
        return fcon.read(stringLength).decode("utf-8", errors="replace")

    else:
        raise ValueError(f"Invalid primitive data type {dataType}.")


def mcf_readNamedTableRow(fcon):
    numRowCells = int.from_bytes(fcon.read(1), byteorder='big')
    output = {}

    for rciter in range(1, numRowCells + 1):
        next_byte = int.from_bytes(fcon.read(1), byteorder='big')

        if next_byte <= 127:
            if next_byte != rciter + 1:
                raise ValueError("Invalid row cell index")
            name = str(next_byte)
        else:
            fcon.seek(-1, 1)  # move back 1 byte
            name = mcf_readNamedName(fcon)

        prim = mcf_readPrimitive(fcon)

        if next_byte <= 127:
            output[rciter - 1] = prim  # use integer index
        else:
            output[name] = prim

    return output


def readMCFCalibration(path, offsetTable):
    with open(path, "rb") as f:
        rawfile = f.read() + b"\x00" * 1000  # pad like R

    from io import BytesIO
    fcon = BytesIO(rawfile)

    caltab = []

    for i, row in offsetTable.iterrows():
        fcon.seek(row["Offset"])

        mcf_checkBlobCodeWord(fcon)
        bin_checkBytes(fcon, b"\x01\x01")
        fcon.seek(16, 1)  # seek forward 16 bytes from current
        numNamedEls = bin_readVarInt(fcon)

        name = mcf_readNamedName(fcon)
        if name != "Transformator":
            raise ValueError("First named element of parameter blob should be 'Transformator'")
        bin_checkBytes(fcon, b"\x01")
        fcon.seek(16, 1)

        calrow = mcf_readNamedTableRow(fcon)
        if calrow.get("calibMode") == 4:
            calrow["alpha"] = 0
            calrow["beta"] = -calrow["beta"]

        calrow["toId"] = row["toId"]
        caltab.append(calrow)

    return pd.DataFrame(caltab)


def getBrukerMCFSpectrum(reader, index):
    return reader.get_spectrum(index)


def getBrukerMCFSpots(reader):
    return reader.get_spots()


def readBrukerMCFIndexFile(path, calibration=False):
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        try:
            containerdf = cur.execute("SELECT * FROM ContainerIndex").fetchall()
        except sqlite3.OperationalError:
            raise ValueError("ContainerIndex table not found.")
        containerdf = pd.DataFrame(containerdf, columns=["GuidA", "GuidB", "BlobResType", "Offset", "BlobSize"])
        containerdf['id'] = containerdf["GuidA"].astype(str) + containerdf["GuidB"].astype(str)
        containerdf = containerdf.drop(columns=["GuidA", "GuidB"])

        offsets = containerdf["Offset"]
        offset_page = np.zeros(len(offsets), dtype=int)

        #TODO: check if the logic aligns with the original code
        is_list = offsets.apply(lambda x: isinstance(x, (list, tuple)) and len(x) > 1)
        if is_list.any():
            offset0 = offsets.where(~is_list, offsets.apply(lambda x: x[0]))
            offset_page = np.where(is_list, offsets.apply(lambda x: x[1] * 2), 0)
            offsets = offset0

        is_negative = offsets < 0
        if is_negative.any():
            offset_page += 1
            offset0 = offsets.where(~is_negative, offsets.apply(lambda x: x + (2 ** 31)))
            offsets = offset0
        containerdf["Offset"] = offsets
        containerdf["OffsetPage"] = offset_page

        if calibration:
            relation_df = cur.execute("SELECT * FROM Relations").fetchall()
            relation_df = pd.DataFrame(relation_df, columns=["GuidA", "GuidB", "ToGuidA", "ToGuidB", "RelationType"])
            relation_df["id"] = relation_df["GuidA"].astype(str) + relation_df["GuidB"].astype(str)
            relation_df["toId"] = relation_df["ToGuidA"].astype(str) + relation_df["ToGuidB"].astype(str)
            relation_df = relation_df.drop(columns=["GuidA", "GuidB", "ToGuidA", "ToGuidB"])

            containerdf = containerdf.merge(relation_df, on="id", how="left")

        containerdf = containerdf.sort_index()

        return containerdf


def newBrukerMCFReader(mcfdir):
    return RtmsBrukerMCFReader.from_dir(mcfdir)


def mcf_blobSeek(fcon, offset, page):
    fcon.seek(0)
    if page > 0:
        fcon.seek((2 ** 31) * page, 1)  # move forward by N * 2^31 bytes
    fcon.seek(offset, 1)  # then move forward by the within-page offset


def mcf_readTableRow(fcon, namevec=None):
    numRowCells = int.from_bytes(fcon.read(1), byteorder="big")

    if namevec is None:
        namevec = [""] * numRowCells
    elif len(namevec) != numRowCells:
        raise ValueError("Name vector must match number of cells")

    output = {}

    for rciter in range(numRowCells):
        nextByte = int.from_bytes(fcon.read(1), byteorder="big")
        if nextByte != (rciter + 2):  # R's rciter starts at 1
            raise ValueError("Invalid row cell index")

        prim = mcf_readPrimitive(fcon)

        if namevec[rciter] == "":
            output[rciter] = prim  # fallback to numeric key
        else:
            output[namevec[rciter]] = prim

    return output


def mcf_readTable(fcon):
    bin_checkBytes(fcon, b"\x03\x01")
    fcon.seek(16, 1)  # skip 16 bytes
    numRows = bin_readVarInt(fcon)

    outTable = []

    for riter in range(1, numRows + 1):
        if riter == 1:
            curRow = mcf_readNamedTableRow(fcon)
            namevec = list(curRow.keys())
        else:
            curRow = mcf_readTableRow(fcon, namevec)
        outTable.append(curRow)

    return pd.DataFrame(outTable)


def getBrukerMCFIndices(reader):
    return reader.spotTable.index


def mcf_readPrimitiveArray(fcon):
    bin_checkBytes(fcon, b"\x03")  # array marker
    dataType = int.from_bytes(fcon.read(1), byteorder="big")
    numEls = bin_readVarInt(fcon)
    if numEls == 0:
        return []

    outVec = []

    for _ in range(numEls):
        if dataType in (39, 40):  # 8-byte integer
            val = int.from_bytes(fcon.read(8), byteorder="little", signed=True)
        elif dataType in (37, 38):  # 4-byte integer
            val = int.from_bytes(fcon.read(4), byteorder="little", signed=True)
        elif dataType == 32:  # 4-byte float
            val = struct.unpack("<f", fcon.read(4))[0]
        elif dataType == 31:  # 8-byte float
            val = struct.unpack("<d", fcon.read(8))[0]
        elif dataType == 0:  # null (?)
            val = None
        elif dataType == 42:  # ASCII string
            strlen = bin_readVarInt(fcon)
            val = fcon.read(strlen).decode("ascii")
        else:
            raise ValueError(f"Invalid primitive data type {dataType}")
        outVec.append(val)

    return outVec


def retrieveMCFMetadata(fcon, offset, offsetPage=0):
    mcf_blobSeek(fcon, offset, offsetPage)

    # Header
    mcf_checkBlobCodeWord(fcon)
    bin_checkBytes(fcon, b"\x01\x01")
    fcon.seek(16, 1)
    numNamedEls = bin_readVarInt(fcon)

    # "Declarations"
    name = mcf_readNamedName(fcon)
    if name != "Declarations":
        raise ValueError("First named element of parameter blob should be 'Declarations'")
    outTable = mcf_readTable(fcon)

    # "Replacements"
    name = mcf_readNamedName(fcon)
    if name != "Replacements":
        raise ValueError("Second named element of parameter blob should be 'Replacements'")
    bin_checkBytes(fcon, b"\x02\x01")
    fcon.seek(16, 1)
    numReps = bin_readVarInt(fcon)

    valTable = []

    for nriter in range(1, numReps + 1):
        bin_checkBytes(fcon, b"\x01")
        fcon.seek(16, 1)
        bin_checkBytes(fcon, b"\x03")

        if nriter == 1:
            nextName = mcf_readNamedName(fcon)
        else:
            bin_checkBytes(fcon, b"\x0B")

        codes = mcf_readPrimitiveArray(fcon)
        bin_checkBytes(fcon, b"\x0A")
        values = mcf_readPrimitiveArray(fcon)
        bin_checkBytes(fcon, b"\x09")
        bin_checkBytes(fcon, b"\x25")
        key = int.from_bytes(fcon.read(4), byteorder='little', signed=True)

        if len(codes) != len(values):
            raise ValueError("Replacement codes must match values in number")

        if len(codes) > 0:
            for code, value in zip(codes, values):
                valTable.append({"Key": key, "Code": code, "Value": str(value)})

    return {"declarations": outTable, "replacements": pd.DataFrame(valTable)}


def getBrukerMCFAllMetadata(reader, index):
    return reader.get_metadata(index)


def mcf_readKeyValueTable(fcon):
    bin_checkBytes(fcon, b"\x03\x01")
    fcon.seek(16, 1)
    numRows = bin_readVarInt(fcon)

    keyTable = []

    for riter in range(numRows):
        if riter == 0:
            curRow = mcf_readNamedKeyValueRow(fcon)
        else:
            curRow = mcf_readKeyValueRow(fcon)
        keyTable.append(curRow)

    return pd.DataFrame(keyTable)


def mcf_readNamedKeyValueRow(fcon):
    bin_checkBytes(fcon, b"\x02", "Key-value row must contain two elements")
    output = {}

    for rciter in range(1, 3):  # 1 to 2
        next_byte = int.from_bytes(fcon.read(1), byteorder="big")

        if next_byte <= 127:
            if next_byte != rciter + 1:
                raise ValueError("Invalid row cell index")
            name = "Key" if rciter == 1 else "Value"
        else:
            fcon.seek(-1, 1)
            name = mcf_readNamedName(fcon)
            if (rciter == 1 and name != "Key") or (rciter == 2 and name != "Value"):
                raise ValueError("Names in a key-value row must be 'Key' and 'Value'")

        prim = mcf_readPrimitive(fcon)

        if rciter == 1:
            if not isinstance(prim, int):
                raise ValueError("Key in key-value row must be an integer.")
            output["Key"] = prim
        else:
            output["Value"] = prim

    return output


def mcf_readKeyValueRow(fcon):
    bin_checkBytes(fcon, b"\x02", "Key-value row must contain two elements")
    output = {}

    for rciter in range(1, 3):  # 1 to 2
        next_byte = int.from_bytes(fcon.read(1), byteorder="big")
        if next_byte != rciter + 1:
            raise ValueError("Invalid row cell index")

        prim = mcf_readPrimitive(fcon)

        if rciter == 1:
            if not isinstance(prim, int):
                raise ValueError("Key in key-value row must be an integer.")
            output["Key"] = prim
        else:
            output["Value"] = prim

    return output


def simple_calibration(spectrum_in: np.ndarray, return_shift=False,
                       **kwargs) -> float or np.ndarray:
    """ simple calibration function that only shifts the spectrum left or right """

    # find the highest peak within the specified mass tolerance around the calibrant m/z
    target_mz, _ = peak_at(spectrum_in, **kwargs)

    # if no peak found, return the original spectrum
    if np.isnan(target_mz):
        if return_shift:
            return np.nan
        else:
            # if no peak found, return the original spectrum
            return spectrum_in
    else:
        # calculate the shift
        shift = target_mz - kwargs["mz"]
        if return_shift:
            return shift
        else:
            # apply the shift to the spectrum
            spectrum_out = spectrum_in.copy()
            spectrum_out[:, 0] -= shift
            return spectrum_out


class Pipeline:
    def __init__(self, reader, **kwargs):
        self.reader = reader
        self.spots = None
        self.spectra = None
        self.calib_params = {
            "mz": None,
            "min_intensity": 1e5,
            "tol": 0.02,
            "min_snr": 0,
        }

        self.after_params = {
            "target_mzs": None,
            "min_intensity": 1e4,
            "tol": 0.01,
            "min_snr": 0,
        }
        self.batch_processor = BatchProcessor(reader, **kwargs)

    # function to set the parameters for calibration
    def set_calib_params(self, **kwargs):
        self.calib_params.update(kwargs)

    # function to set the parameters for after calibration
    def set_after_params(self, **kwargs):
        # if target_mzs is in kwargs, check if it is a list
        if "target_mzs" in kwargs:
            if not isinstance(kwargs["target_mzs"], list):
                try:
                    kwargs["target_mzs"] = list(kwargs["target_mzs"])
                except TypeError:
                    raise ValueError("target_mzs must be a list")
        self.after_params.update(kwargs)

    def process(self, **kwargs):
        self.spots = self.reader.get_spots()['SpotNumber'].values

        if "uncalib_line_spectra" in kwargs:
            uncalib_line_spectra = kwargs["uncalib_line_spectra"]
        else:
            uncalib_line_spectra = self.batch_processor.calib_all_spectra(**self.calib_params)
        self.spectra = self.batch_processor.pick_mul_from_uncalib_line_spectra(uncalib_line_spectra,
                                                                               **self.after_params)
        self.final_result = FinalResult()
        for i, target_mz in enumerate(self.after_params["target_mzs"]):
            mz_intensity = np.array([spectrum[i] for spectrum in self.spectra])
            result = SingleResultContainer(target_mz,
                                            mz_intensity[:, 0],
                                            mz_intensity[:, 1],
                                            self.spots)
            self.final_result.append(result)
        return uncalib_line_spectra

    def plot_calibration(self):
        fig, ax = plt.subplots()
        ax.plot(self.reader.shifts)
        ax.set_xlabel("Spot Number")
        ax.set_ylabel("Mass Shift (Da)")
        ax.set_title("Calibration Result")
        ax.axhline(0, color='r', linestyle='--')
        # add text show mean and std
        mean = np.nanmean(self.reader.shifts)
        std = np.nanstd(self.reader.shifts)
        ax.text(0.5, 0.5, f"Mean: {1000 * mean:.2f} mDa\nStd: { 1000 * std:.2f} mDa", transform=ax.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center')
        return fig


@dataclass
class SingleResultContainer:
    theoretical_mz: Optional[float] = None
    measured_mzs: Optional[np.ndarray[float]] = None
    intensities: Optional[np.ndarray[float]] = None
    spot_numbers: Optional[List[str]] = None


class FinalResult:
    def __init__(self):
        self.results: Dict[float, SingleResultContainer] = {}

    def get_x(self, spot_numbers):
        return [re.findall(r'X(\d+)', sn)[0] for sn in spot_numbers]

    def get_y(self, spot_numbers):
        return [re.findall(r'Y(\d+)', sn)[0] for sn in spot_numbers]

    @property
    def spot_numbers(self):
        return self.results[list(self.results.keys())[0]].spot_numbers

    @classmethod
    def from_single_results(cls, containers: List[SingleResultContainer]) -> "FinalResult":
        instance = cls()
        for container in containers:
            if not isinstance(container, SingleResultContainer):
                raise ValueError("All elements must be SingleResultContainer")
            if container.theoretical_mz is None:
                raise ValueError("theoretical_mz cannot be None")
            instance.results[container.theoretical_mz] = container
        return instance

    def result_for(self, mz: float) -> SingleResultContainer:
        try:
            return self.results[mz]
        except KeyError:
            raise ValueError(f"Result for {mz} not found.")

    def append(self, container: SingleResultContainer):
        if not isinstance(container, SingleResultContainer):
            raise ValueError("container must be SingleResultContainer")
        if container.theoretical_mz is None:
            raise ValueError("theoretical_mz cannot be None")
        self.results[container.theoretical_mz] = container

    def to_df(self):
        theoretical_mz_all = np.concatenate([[r.theoretical_mz] * len(r.measured_mzs)
                                             for r in self.results.values()])
        mz_all = np.concatenate([r.measured_mzs for r in self.results.values()])
        intensity_all = np.concatenate([r.intensities for r in self.results.values()])
        spot_all = np.concatenate([r.spot_numbers for r in self.results.values()])

        data = np.column_stack((theoretical_mz_all, mz_all, intensity_all, spot_all))
        return pd.DataFrame(data, columns=['theoretical_mz', 'mz', 'intensity', 'spot'])

    def to_str(self):
        result = self.to_df()
        result = result.dropna()
        result['snr'] = 0
        result_str = f"{len(result['spot'].unique())}\n"
        for spot in result['spot'].unique().tolist():
            subresult = result.query("spot == @spot")
            n_peaks = subresult['theoretical_mz'].unique().size
            result_str += f"{spot};{n_peaks}"
            # concat the rows to string
            for index, row in subresult.iterrows():
                result_str += f";{row['mz']:.4f};{row['intensity']:.4f};{row['snr']:.4f}"
            result_str += "\n"
        return result_str


    def viz2D(self):
        # add len(target_mzs) columns of subplots
        fig, axes = plt.subplots(1, len(self.results))
        if len(self.results) == 1:
            axes = [axes]
        for i, (mz, container) in enumerate(self.results.items()):
            x = self.get_x(container.spot_numbers)
            x = np.array(x).astype(int)
            y = self.get_y(container.spot_numbers)
            y = np.array(y).astype(int)

            df = pd.DataFrame(
                {'x': x,
                 'y': y,
                 'intensity': container.intensities
                 })

            axes[i].imshow(
                df.pivot(index='x', columns='y', values='intensity').values,
                vmax=df['intensity'].quantile(0.95)
            )
            # turn off all axes
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(f'm/z {mz:.4f}')
        return fig



if __name__ == "__main__":
    pass