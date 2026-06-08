import numpy as np

def LTV(LPT,DATA):

    means_LPT = np.empty((6))
    means_LTV = np.empty((6))

    for i in range(6):
        means_LPT[i] = np.mean(LPT[:,i])
        means_LTV[i] = np.mean(DATA[:,i])

    R = (means_LPT - means_LTV)/means_LPT

    Relative_differences = {'SXR1':f'{R[0]:4}','SXR2':f'{R[1]:4}','SXR3':f'{R[2]:4}','SXR4':f'{R[3]:4}','HXR':f'{R[4]:4}','EUV':f'{R[5]:4}'}

    return Relative_differences

if __name__ == "__main__":
    csv_path = "/Users/evanwilliams/Desktop/ltv_experiment/ASIO_Science_Generated.csv"
    csv_background_path = "/Users/evanwilliams/Desktop/ltv_experiment/06_20250918_background_nolights_3_Science.csv"
    data = np.loadtxt(csv_path, delimiter=",")
    data_background = np.loadtxt(csv_background_path, delimiter=",")

    sensitivity = 3
    result = LTV(data, data_background)
    print(result)
