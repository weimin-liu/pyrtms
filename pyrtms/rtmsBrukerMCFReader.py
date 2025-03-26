import os
import pandas as pd
import numpy as np
import io
import sys
import math
import struct
import zlib

class RtmsBrukerMCFReader:
    def __init__(self, dir, files, spotTable, offsetTable, metadata, con=None):
        assert isinstance(dir, str), "dir must be a string"
        assert isinstance(files, list), "files must be a dictionary"
        assert isinstance(spotTable, pd.DataFrame), "spotTable must be a DataFrame"
        assert isinstance(offsetTable, pd.DataFrame), "offsetTable must be a DataFrame"
        assert isinstance(metadata, dict), "metadata must be a dictionary"
        if con is not None:
            assert hasattr(con, "read"), "con must be a file-like object"

        self.dir = dir
        self.files = files
        self.spotTable = spotTable
        self.offsetTable = offsetTable
        self.metadata = metadata
        self.con = con

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

def sqlt_getPage(scon, page):
    assert page == int(page)
    page = int(page)
    scon.seek((page - 1) * 1024)
    rawpage = scon.read(1024)
    rawcon = io.BytesIO(rawpage)
    return rawcon

def sqlt_parseRecordValue(scon, type):
    if type == 0:
        return np.nan
    elif type < 0 or not isinstance(type, int):
        raise ValueError("Unsupported record data type")
    elif type == 8:
        return 0
    elif type == 9:
        return 1
    elif type == 12:
        return bytes()
    elif type == 13:
        return ""
    elif type == 1:
        return int.from_bytes(scon.read(1), byteorder=sys.byteorder, signed=True)
    elif type == 2:
        return int.from_bytes(scon.read(2), byteorder='big', signed=True)
    elif type == 3:
        tempraw = b'\x00' + scon.read(3)
        return int.from_bytes(tempraw, byteorder='big', signed=True)
    elif type == 4:
        return int.from_bytes(scon.read(4), byteorder='big', signed=True)
    elif type == 5:
        tempraw = b'\x00\x00' + scon.read(6)
        # unpack two big-endian 4-byte signed integers
        ints = struct.unpack(">ii", tempraw)
        return ints[::-1]  # reverse the tuple
    elif type == 6:
        ints = struct.unpack(">ii", scon.read(8))  # read 2 x 4-byte signed ints
        return ints[::-1]  # reverse the order
    elif type == 7:
        return struct.unpack(">d", scon.read(8))[0]
    elif type == 10 or type == 11:
        raise TypeError("Unsupported record data type")
    elif type % 2 == 0:
        length = (type - 12) // 2
        return scon.read(length)
    else:
        length = (type - 13) // 2
        return scon.read(length).decode("utf-8")

def getBrukerMCFSpectrum(reader, index):
    if not isinstance(reader, RtmsBrukerMCFReader):
        raise ValueError("reader must be an instance of RtmsBrukerMCFReader")
    if index < 0 or index > len(reader.spotTable):
        raise IndexError("Index out of bounds")

    fcon = reader.con
    row = reader.spotTable.iloc[index - 1]

    spotId = row["id"]
    fhigh = row["frequencyHigh"]
    fwidth = row["frequencyWidth"]
    fsize = row["size"]
    alpha = row["alpha"]
    beta = row["beta"]

    def massToIndex(p):
        return fsize * (fwidth - (fhigh / (p - alpha / fhigh) + beta)) / fwidth

    def indexToMass(i):
        return fhigh / ((fwidth * (fsize - i) / fsize) - beta) + alpha / fhigh

    # Seek to raw data blob
    blob_row = reader.offsetTable[(reader.offsetTable["id"] == spotId) &
                                  (reader.offsetTable["BlobResType"] == 258)].iloc[0]
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
        compressed = False
        numValues = bin_readVarInt(fcon)
        spectrum = np.frombuffer(fcon.read(numValues * 4), dtype="<f4")
    elif typeByte == b"\x22":  # gzip-compressed
        compressed = True
        numBytes = bin_readVarInt(fcon)
        gzippedBytes = fcon.read(numBytes)
        unzipped = zlib.decompress(gzippedBytes)
        spectrum = np.frombuffer(unzipped, dtype="<f4")
        numValues = len(spectrum)
    else:
        raise ValueError("Raw spectra must be an array of 32-bit floats.")

    mz = np.array([indexToMass(i) for i in range(numValues)])
    return pd.DataFrame({"mz": mz, "intensity": spectrum})

def sqlt_parseRecord(scon):
    headerSize = bin_readVarInt(scon, "big")
    headerRemaining = headerSize - 1
    dataTypes = []
    while headerRemaining > 0:
        nextCode = bin_readVarInt(scon, "big")
        dataTypes.append(nextCode)
        headerRemaining -= max(math.ceil(math.log(nextCode, 128)), 1)
    output = [
        sqlt_parseRecordValue(scon, dataTypes[dtiter])
        for dtiter in range(len(dataTypes))
    ]
    return output

def getBrukerMCFSpots(reader):
    if not isinstance(reader, RtmsBrukerMCFReader):
        raise ValueError("Parameter 'reader' must be of class 'RtmsBrukerMCFReader'")

    mcfdir = reader.dir
    mainIndex = reader.files[1]

    with open(os.path.join(mcfdir, mainIndex), "rb") as scon:
        def mainHandler(values, cellIndex):
            return {"name": values[1], "value": values[3]}

        _, mainTable = sqlt_parseBTreeTable(scon, 1, mainHandler, first=True)

        metaRoot = int(mainTable.loc[mainTable["name"] == "MetaDataString", "value"].values[0])

        def metastringHandler(values, cellIndex):
            valueList = {
                "GuidA": values[0],
                "GuidB": values[1],
                "MetadataId": values[2],
                "Text": values[3]
            }
            valueList["id"] = f"{valueList['GuidA']} {valueList['GuidB']}"
            del valueList["GuidA"]
            del valueList["GuidB"]
            return valueList

        _, metasout = sqlt_parseBTreeTable(scon, metaRoot, metastringHandler, first=False)

    wmeta1 = metasout[metasout["MetadataId"] == 64][["id", "Text"]].rename(columns={"Text": "SpotNumber"})
    wmeta2 = metasout[metasout["MetadataId"] == 34][["id", "Text"]].rename(columns={"Text": "Timestamp"})
    wmeta = pd.merge(wmeta1, wmeta2, on="id", how="outer")

    spots = pd.merge(reader.spotTable[["id", "index"]], wmeta, on="id", how="left")
    spots = spots.sort_values("index").reset_index(drop=True)
    spots = spots.drop(columns=["id"])

    return spots

def sqlt_parseBTreeInteriorPage(pagecon, upper):
    pagecon.seek(2, io.SEEK_CUR)
    numCells = int.from_bytes(pagecon.read(2), byteorder='big', signed=False)
    cellStart = int.from_bytes(pagecon.read(2), byteorder='big', signed=False)
    pagecon.seek(1, io.SEEK_CUR)
    #TODO: signed =True is it correct?
    finalPage = int.from_bytes(pagecon.read(4), byteorder='big', signed=True)
    cellPointers = [
        int.from_bytes(pagecon.read(2), byteorder='big', signed=False)
        for _ in range(numCells)
    ]

    newpagedf = pd.DataFrame({
        "Page": [np.nan] * numCells,
        "Upper": [np.nan] * numCells,
        "Type": ["None"] * numCells
    })

    for citer in range(numCells):
        pagecon.seek(cellPointers[citer])
        newpagedf.at[citer, "Page"] = int.from_bytes(pagecon.read(4), byteorder='big', signed=True)
        newpagedf.at[citer, "Upper"] = bin_readVarInt(pagecon, "big")
    new_row = pd.DataFrame([{"Page": finalPage, "Upper": upper, "Type": "None"}])
    return pd.concat([newpagedf, new_row], ignore_index=True)

def sqlt_parseBTreeLeafPage(pagecon, handler):
    if handler is None:
        return []
    pagecon.seek(2, io.SEEK_CUR)
    numCells = int.from_bytes(pagecon.read(2), byteorder='big', signed=False)
    cellStart = int.from_bytes(pagecon.read(2), byteorder='big', signed=False)
    pagecon.seek(1, io.SEEK_CUR)
    cellPointers = [
        int.from_bytes(pagecon.read(2), byteorder='big', signed=False)
        for _ in range(numCells)
    ]

    rows = [[] for _ in range(numCells)]
    for citer in range(numCells):
        pagecon.seek(cellPointers[citer])
        cellSize = bin_readVarInt(pagecon, "big")
        cellIndex = bin_readVarInt(pagecon, "big")
        valueList = sqlt_parseRecord(pagecon)
        rows[citer] = handler(valueList, cellIndex)
    return pd.DataFrame(rows)



def sqlt_parseBTreeTable(scon, rootpage, handler, first=False):
    pastTheFirst = not first
    pagedf = pd.DataFrame({
        "Page": [rootpage],
        "Upper": [np.inf],
        "Type": ["None"]
    })
    outputdf = pd.DataFrame()

    while True:
        if "None" not in pagedf['Type'].unique():
            break
        curpageind = pagedf[pagedf['Type'] == "None"].index[0]
        curpage = pagedf.iloc[curpageind]
        pagecon = sqlt_getPage(scon, curpage['Page'])
        if not pastTheFirst:
            pagecon.seek(100)
            pastTheFirst = True
        pageType = int.from_bytes(pagecon.read(1), byteorder='big', signed=False)
        if pageType == 5:
            newpages = sqlt_parseBTreeInteriorPage(pagecon, pagedf.at[curpageind, "Upper"])
            pagedf = pd.concat([pagedf.drop(curpageind), newpages], ignore_index=True)
        elif pageType == 13:
            outputdf = pd.concat(
                [outputdf, sqlt_parseBTreeLeafPage(pagecon, handler)],
                ignore_index=True
            )
            pagedf.at[curpageind, "Type"] = "Leaf"
        elif pageType != 5:
            raise ValueError("Page must be b-tree leaf or interior page")
            pagecon.close()
        pagecon.close()

    pagedf = pagedf.sort_values("Upper").reset_index(drop=True)
    return [pagedf, outputdf]

def readBrukerMCFIndexFile(path, calibration=False):
    def mainHandler(valueList, cellIndex):
        def is_whole_number(x):
            return isinstance(x, (int, float)) and float(x).is_integer()

        if valueList[1] == "ContainerIndex":
            if is_whole_number(valueList[3]):
                return {"name": "blobRootPage", "value": int(valueList[3])}
            else:
                raise ValueError("Page record column should be an integer")
        elif valueList[1] == "Relations":
            if is_whole_number(valueList[3]):
                return {"name": "relRootPage", "value": int(valueList[3])}
            else:
                raise ValueError("Page record column should be an integer")
        else:
            return None  # or return {}

    def containerHandler(values, cellIndex):
        # Unpack values assuming fixed order
        valueList = {
            "GuidA": values[0],
            "GuidB": values[1],
            "BlobResType": values[2],
            "Offset": values[3],
            "BlobSize": values[4],
            "Index": cellIndex
        }

        # Construct ID from GuidA and GuidB
        valueList["id"] = f"{valueList['GuidA']} {valueList['GuidB']}"

        # Remove GuidA and GuidB
        del valueList["GuidA"]
        del valueList["GuidB"]

        # Handle OffsetPage
        offset = valueList["Offset"]
        offset_page = 0

        if isinstance(offset, (list, tuple)) and len(offset) > 1:
            offset_page = offset[1] * 2
            offset = offset[0]

        if offset < 0:
            offset_page += 1
            offset = offset + (2 ** 31)

        valueList["OffsetPage"] = offset_page
        valueList["Offset"] = offset

        return valueList

    with open(path, 'rb') as scon:
        _, mainTable = sqlt_parseBTreeTable(scon, 1, mainHandler, first=True)
        mainTable = mainTable.dropna()
        mainTable_dict = {}
        for index, row in mainTable.iterrows():
            mainTable_dict.update({index:row.values[0]})
        mainTable = pd.DataFrame(mainTable_dict).T

        try:
            containerRoot = mainTable.loc[mainTable["name"] == "blobRootPage", "value"].values[0]
        except IndexError:
            raise ValueError("ContainerIndex table not found.")

        _, containerdf = sqlt_parseBTreeTable(scon, containerRoot, containerHandler, first=False)

        if calibration:
            relationRoot = mainTable.query("name == 'relRootPage'")["value"].values[0]

            def relationHandler(values, cellIndex):
                # Map values to named fields
                valueList = {
                    "GuidA": values[0],
                    "GuidB": values[1],
                    "ToGuidA": values[2],
                    "ToGuidB": values[3],
                    "RelationType": values[4],
                }

                # Construct IDs
                valueList["id"] = f"{valueList['GuidA']} {valueList['GuidB']}"
                valueList["toId"] = f"{valueList['ToGuidA']} {valueList['ToGuidB']}"

                # Remove original GUID fields
                del valueList["GuidA"]
                del valueList["GuidB"]
                del valueList["ToGuidA"]
                del valueList["ToGuidB"]

                return valueList
            _, relationdf = sqlt_parseBTreeTable(scon, relationRoot, relationHandler, first=False)
            containerdf = containerdf.merge(relationdf, how="left")
            containerdf = containerdf.sort_values("Index")[[
                "Index", "id", "toId", "RelationType", "BlobResType", "Offset", "OffsetPage", "BlobSize"
            ]]

        else:
            containerdf = containerdf.sort_values("Index")[[
                "Index", "id", "BlobResType", "Offset", "OffsetPage", "BlobSize"
            ]]
        return containerdf



def newBrukerMCFReader(mcfdir):
    # find the MCF file in mcfdir
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
        calOffsets = calOffsets.reset_index()

        subset_ids = offsetTable.loc[offsetTable["BlobResType"] == 258, ["id"]]
        spotCalOffsets = calOffsets.merge(subset_ids, left_on="toId", right_on="id", how="inner", sort=False)
        spotCalOffsets = spotCalOffsets.sort_values("index").reset_index(drop=True)
        spotCalData = readMCFCalibration(os.path.join(mcfdir, calibFile), spotCalOffsets)
        spotCalData = spotCalData.reset_index()

        spotTable = spotCalData.sort_index()
        spotTable = spotTable.rename(columns={"toId": "id"})

        mcfcon = open(os.path.join(mcfdir,mainFile), "rb")

        paramBlobOffset = offsetTable["Offset"].iloc[2]  # 0-based index in Python

        metadata = retrieveMCFMetadata(mcfcon, paramBlobOffset, 0)
        return RtmsBrukerMCFReader(
            dir=mcfdir,
            files=files,
            spotTable=spotTable,
            offsetTable=offsetTable,
            metadata=metadata,
            con=mcfcon
        )


def mcf_blobSeek(fcon, offset, page):
    fcon.seek(0)
    if page > 0:
        fcon.seek((2**31) * page, 1)  # move forward by N * 2^31 bytes
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
    return reader.spotTable.index + 1

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
    import pandas as pd

    blobRow = reader.offsetTable[
        (reader.offsetTable["id"] == reader.spotTable.iloc[index]["id"]) &
        (reader.offsetTable["BlobResType"] == 259)
    ].index[0]

    dectable = reader.metadata["declarations"]
    reptable = reader.metadata["replacements"]

    fcon = reader.con
    offset = reader.offsetTable.at[blobRow, "Offset"]
    offsetPage = reader.offsetTable.at[blobRow, "OffsetPage"]
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
                lambda row: row["DisplayFormat"] % row["Value"] if "DisplayFormat" in row else str(row["Value"]),
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

if __name__ == "__main__":
    pass