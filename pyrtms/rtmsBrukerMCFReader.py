import os
import sqlite3

import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
from typing import List, Optional, Dict

import struct
import zlib
import multiprocessing as mp
from pyopenms import MSSpectrum, PeakPickerHiRes
from functools import partial

from pyrtms.utils import peak_at
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyopenms")
import matplotlib.pyplot as plt
import h5py


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


class BatchProcessor:
    def __init__(self, reader, **kwargs):
        self.reader = reader
        self.n_jobs = min(kwargs.get("n_jobs", mp.cpu_count()), 60)
        self.show_progress = kwargs.get("show_progress", True)

    def calib_all_spectra(self, **kwargs) -> list:
        """
        calibrate all spectra in the spotTable
        """
        uncalib_line_spectra = self.pick_mul_spectra(self.reader.spotTable.index, **kwargs)
        f = partial(simple_calibration, **kwargs, return_shift=True)

        with mp.Pool(self.n_jobs) as pool:
            if self.show_progress:
                shifts = list(tqdm(pool.imap(f, [spectrum for spectrum in uncalib_line_spectra]),
                                   total=len(uncalib_line_spectra)))
            else:
                shifts = pool.map(f, [spectrum for spectrum in uncalib_line_spectra])
        self.reader.shifts = np.array(shifts)
        self.reader.mean_shift = np.nanmean(self.reader.shifts)
        return uncalib_line_spectra

    def get_mul_spectra(self, indices, **kwargs):
        f = partial(self.reader.get_spectrum, **kwargs)
        with mp.Pool(self.n_jobs) as pool:
            if self.show_progress:
                results = list(tqdm(pool.imap(f, indices), total=len(indices)))
            else:
                results = pool.map(f, indices)
        return results

    def pick_mul_spectra(self, indices, **kwargs):

        f = partial(self.reader.pick_spectrum, line_spec=None, **kwargs)

        with mp.Pool(self.n_jobs) as pool:
            if self.show_progress:
                results = list(tqdm(pool.imap(f, indices), total=len(indices)))
            else:
                results = pool.map(f, indices)
        return results

    def pick_mul_from_uncalib_line_spectra(self, uncalib_line_spectra, **kwargs):

        indices = self.reader.spotTable.index
        f = partial(self.reader.pick_spectrum, **kwargs)

        # calibrate the line spectra
        if hasattr(self.reader, "mean_shift"):
            for i in range(len(uncalib_line_spectra)):
                if np.isnan(self.reader.shifts[i]):
                    uncalib_line_spectra[i][:, 0] -= self.reader.mean_shift
                else:
                    uncalib_line_spectra[i][:, 0] -= self.reader.shifts[i]

        line_specs = [[None,uncalib_line_spectra[i] ]for i in indices]
        # zip nones with the indices


        with mp.Pool(self.n_jobs) as pool:
            if self.show_progress:
                results = list(tqdm(pool.starmap(f, line_specs), total=len(line_specs)))
            else:
                results = pool.starmap(f, line_specs)
        return results

class RtmsBrukerMCFReader:
    def __init__(self, mcfdir, files, spotTable, offsetTable, metadata):
        assert isinstance(mcfdir, str), "dir must be a string"
        assert isinstance(files, list), "files must be a dictionary"
        assert isinstance(spotTable, pd.DataFrame), "spotTable must be a DataFrame"
        assert isinstance(offsetTable, pd.DataFrame), "offsetTable must be a DataFrame"
        assert isinstance(metadata, dict), "metadata must be a dictionary"

        self.mcfdir = mcfdir
        self.files = files
        self.spotTable = spotTable
        self.offsetTable = offsetTable
        self.metadata = metadata
        self.spots = None

    @property
    def mzs(self):
        return self.get_spectrum(0, return_mzs=True)[0]

    @property
    def str_xy(self):
        return self.get_spots()['SpotNumber'].values

    @property
    def xy(self):
        spot_x = [re.findall(r"X(\d+)", str(spot))[0] for spot in self.str_xy]
        spot_x = [int(x) for x in spot_x]
        spot_y = [re.findall(r"Y(\d+)", str(spot))[0] for spot in self.str_xy]
        spot_y = [int(y) for y in spot_y]
        spot = np.column_stack((spot_x, spot_y))
        return spot


    @property
    def metadataDF(self):
        return self.get_metadata(1)

    @property
    def q1mass(self):
        return float(self.metadataDF.query("PermanentName == 'Q1Mass'")["Value"].values[0].split(' ')[0])

    @property
    def q1res(self):
        return float(self.metadataDF.query("PermanentName == 'Q1Res'")["Value"].values[0].split(' ')[0])

    @property
    def con(self):
        return open(os.path.join(self.mcfdir, self.files[0]), "rb")

    def get_spots(self):
        mainIndex = self.files[1]

        with sqlite3.connect(os.path.join(self.mcfdir, mainIndex)) as conn:
            cur = conn.cursor()
            spot_name = cur.execute("SELECT GuidA, GuidB, MetaDataID, Value from MetaDataString WHERE MetaDataID='64'")
            spot_name = spot_name.fetchall()
            self.spots = pd.DataFrame(spot_name, columns=["GuidA", "GuidB", "MetaDataID", "SpotNumber"])
        return self.spots

    @classmethod
    def from_dir(cls, mcfdir):
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
            spec_obj.set_peaks([spec[0], spec[1]])

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

    def get_spectrum(self, index, CASI_only=False, return_mzs=True, intensities_only=False):
        if intensities_only:
            return_mzs = False # to be compatible with maspim

        if index < 0 or index >= len(self.spotTable):
            raise IndexError("Index out of bounds")

        fcon = self.con
        row = self.spotTable.iloc[index]
        spotId = row["id"]

        # Seek to raw data blob
        blob_row = self.offsetTable[(self.offsetTable["id"] == spotId) &
                                    (self.offsetTable["BlobResType"] == 258)].iloc[0]
        mcf_blobSeek(fcon, blob_row["Offset"], blob_row["OffsetPage"])

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
            numValues = bin_readVarInt(fcon)
            spectrum = np.frombuffer(fcon.read(numValues * 4), dtype="<f4")
        elif typeByte == b"\x22":  # gzip-compressed
            numBytes = bin_readVarInt(fcon)
            gzippedBytes = fcon.read(numBytes)
            unzipped = zlib.decompress(gzippedBytes)
            spectrum = np.frombuffer(unzipped, dtype="<f4")
            numValues = len(spectrum)
        else:
            raise ValueError("Raw spectra must be an array of 32-bit floats.")


        if return_mzs:
            # the following parameters derived from spotTable are the calibrated parameters.
            fhigh = row["frequencyHigh"]
            fwidth = row["frequencyWidth"]
            fsize = row["size"]
            alpha = row["alpha"]
            beta = row["beta"]


            mzindex = np.arange(numValues)
            mz = fhigh / ((fwidth * (fsize - mzindex) / fsize) - beta) + alpha / fhigh

            if CASI_only:
                lower = self.q1mass - self.q1res / 2
                upper = self.q1mass + self.q1res / 2
                mask = (mz >= lower) & (mz <= upper)
                spectrum = spectrum[mask]
                mz = mz[mask]
        else:
            # if return_mzs is False, return the spectrum only
            mz = np.zeros(numValues)
        # return np array of m/z and intensity combined
        return mz, spectrum

    def get_metadata(self, index):
        blobRow = self.offsetTable[
            (self.offsetTable["id"] == self.spotTable.iloc[index]["id"]) &
            (self.offsetTable["BlobResType"] == 259)
            ].index[0]

        dectable = self.metadata["declarations"]
        reptable = self.metadata["replacements"]

        fcon = self.con
        offset = self.offsetTable.at[blobRow, "Offset"]
        offsetPage = self.offsetTable.at[blobRow, "OffsetPage"]
        mcf_blobSeek(fcon, offset, offsetPage)

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
    expected = bytes([0xC0, 0xDE, 0xAF, 0xFE])
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