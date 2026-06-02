
import numpy as np

def LTV(data,sensitivity):
    t_list = [0,0,0,0,0,0]

    means = np.empty((6))
    stds = np.empty_like(means)

    anomalies = []

    for i in range(6):
        zscores = np.empty_like(data[:,i+1])
        mean_i = np.mean(data[:,i+1])
        std_i = np.std(data[:,i+1])

        zscores = (data[:,i+1]-mean_i)/std_i

        means[i],stds[i] = mean_i, std_i

        index = np.where(np.abs(zscores) > sensitivity)

        anomaly_i = data[:,i+1][index]
        time_i = data[:,0][index]

        event = np.column_stack((anomaly_i,time_i))
        anomalies.append(event)
     
        if anomalies[i].shape[0] == 0:
            t_list[i] = 'Pass'
        else:
            t_list[i] = 'Fail'


    passfail = {'Sensitivity':sensitivity,'SXR1':t_list[0],'SXR2':t_list[1],'SXR3':t_list[2],'SXR4':t_list[3],'HXR':t_list[4],'EUV':t_list[5]}
    return passfail

