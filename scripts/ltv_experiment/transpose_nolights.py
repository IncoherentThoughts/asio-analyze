import csv
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

PACKET_SIZE = 950
HEADER_SIZE = 44
ADC_SEGMENT_BYTES = 18
N_SEGMENTS = 50
ADC_DATA_SIZE = N_SEGMENTS * ADC_SEGMENT_BYTES
STOP_OFFSET = HEADER_SIZE + ADC_DATA_SIZE

SRC = "/Users/evanwilliams/Desktop/ltv_experiment/04_20250919_Fe55_SXR3_1.csv"
DST = "/Users/evanwilliams/Desktop/ltv_experiment/04_20250919_Fe55_SXR3_1_Science.csv"

with open(SRC) as f:
    rawdata = np.fromstring(f.read(), dtype=np.uint8, sep=",")

def find(targetString, data):
    target = np.frombuffer(targetString.encode(), dtype=np.uint8)
    windows = sliding_window_view(data, len(target))
    return np.where((windows == target).all(axis=1))[0]

def decode24le(b3):
    return int(b3[0]) | (int(b3[1]) << 8) | (int(b3[2]) << 16)

asioMatches = find("ASIO", rawdata)
packetList = []
for asioIndex in asioMatches:
    end = asioIndex + PACKET_SIZE
    if end > len(rawdata):
        continue
    if rawdata[asioIndex + STOP_OFFSET : asioIndex + STOP_OFFSET + 4].tobytes() != b"STOP":
        continue
    packetList.append(rawdata[asioIndex:end])

print(f"Valid packets: {len(packetList)}")

SXR1, SXR2, SXR3, SXR4, HXR, EUV = [], [], [], [], [], []
for packet in packetList:
    adc = packet[HEADER_SIZE : HEADER_SIZE + ADC_DATA_SIZE]
    for s in range(0, ADC_DATA_SIZE, ADC_SEGMENT_BYTES):
        seg = adc[s : s + ADC_SEGMENT_BYTES]
        SXR1.append(decode24le(seg[0:3]))
        SXR2.append(decode24le(seg[3:6]))
        SXR3.append(decode24le(seg[6:9]))
        SXR4.append(decode24le(seg[9:12]))
        HXR.append(decode24le(seg[12:15]))
        EUV.append(decode24le(seg[15:18]))

mat = np.column_stack((SXR1, SXR2, SXR3, SXR4, HXR, EUV))
print(f"Rows: {mat.shape[0]}  Cols: {mat.shape[1]}")

with open(DST, "w", newline="") as out:
    w = csv.writer(out, delimiter=",", quoting=csv.QUOTE_NONE, escapechar=None)
    for row in mat:
        w.writerow([f"{v:.16e}" for v in row])

print(f"Wrote {DST}")
