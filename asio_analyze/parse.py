import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

rawdata = np.fromfile(
    "/Users/evanwilliams/Desktop/Research/EM Testing/testing-results/Data/FakeTestingData/ShortSample/20260601 1.0x02B4",
    dtype=np.uint8,
)
packet = rawdata[6:954]


def find(targetString, data):
    targetBytes = targetString.encode()
    targetIntegers = np.frombuffer(targetBytes, dtype=np.uint8)
    windows = sliding_window_view(data, len(targetIntegers))
    matches = (windows == targetIntegers).all(axis=1)
    return np.where(matches)[0]


asioMatches = find("ASIO", rawdata)
stopMatches = find("STOP", rawdata)

# ---------Make Packets---------------
packetList = []
for i in range(len(asioMatches)):
    asioIndex = asioMatches[i]
    stopIndex = stopMatches[i]
    newPacket = rawdata[asioIndex : stopIndex + 4]
    packetList.append(newPacket)


def getData(packetList):

    SXR1fields = []
    SXR2fields = []
    SXR3fields = []
    SXR4fields = []
    HXRfields = []
    EUVfields = []

    for packet in packetList:
        data = packet[44:944]
        if len(data) != 900:
            print(
                "Something about the packet is telling me you have more or less than 900 bytes of ASIO data in your packet"
            )
            exit()

        segmentNum = len(data) / 18  # There should be 50 fields
        segments = np.split(data, segmentNum)

        for segment in segments:
            sxr1field = segment[0:3]
            sxr2field = segment[3:6]
            sxr3field = segment[6:9]
            sxr4field = segment[9:12]
            hxrfield = segment[12:15]
            euvfield = segment[15:18]

            SXR1fields.append(sxr1field)
            SXR2fields.append(sxr2field)
            SXR3fields.append(sxr3field)
            SXR4fields.append(sxr4field)
            HXRfields.append(hxrfield)
            EUVfields.append(euvfield)

    SXR1Data = np.concatenate(SXR1fields)
    SXR2Data = np.concatenate(SXR2fields)
    SXR3Data = np.concatenate(SXR3fields)
    SXR4Data = np.concatenate(SXR4fields)
    HXRData = np.concatenate(HXRfields)
    EUVData = np.concatenate(EUVfields)

    dataArrays = np.array([SXR1Data, SXR2Data, SXR3Data, SXR4Data, HXRData, EUVData])

    return dataArrays


arrays = getData(packetList)

for array in arrays:
    print(len(array))

np.savetxt(
    "/Users/evanwilliams/Desktop/Research/EM Testing/testing-results/Data/FakeTestingData/ShortSample/testout.txt",
    arrays,
    delimiter=",",
    fmt="%d",
)
