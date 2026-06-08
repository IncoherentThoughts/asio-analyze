import numpy as np

def LTV(LPT,DATA):

    means_LPT = np.empty((6))
    means_LTV = np.empty((6))

    for i in range(6):
        means_LPT[i] = np.mean(LPT[i+1])
        means_LTV[i] = np.mean(DATA[i+1])

    R = (means_LPT - means_LTV)/means_LPT

    Relative_differences = {'SXR1':R[0],'SXR2':R[1],'SXR3':R[2],'SXR4':R[3],'HXR':R[4],'EUV':R[5]}

    return Relative_differences

if __name__ == "__main__":
    csv_path = "/Users/evanwilliams/Desktop/ltv_experiment/ASIO_Science_Generated.csv"
    csv_background_path = "/Users/evanwilliams/Desktop/ltv_experiment/06_20250918_background_nolights_3_Science.csv"
    data = np.loadtxt(csv_path, delimiter=",")
    data_background = np.loadtxt(csv_background_path, delimiter=",")

    sensitivity = 3
    result = LTV(data, data)
    print(result)
