import numpy as np
import csv
import pandas as pd
import os

def ADC_to_V(val):
    if val > 0x7FFFFF:
        V = (val-(0xFFFFFF+1))/0x800000*2.5 + 2.5
    else:
        V = (val) /0x800000 * 2.5 +2.5
        
    return V

def get_data_dict(file):
    #Open .csv file
    open_file = csv.reader(open(file))
    raw_arr = np.array(list(open_file)[0], dtype=int)
    
    # Find each header
    ind = np.array([], dtype=int)  #index array of headers, specifically the 'A' in 'ASIO'
    for i in np.where( raw_arr==65 )[0]:  #find 'A'
        if raw_arr[i+1]==83:  #is 'S' next?
            if raw_arr[i+2]==73:  #is 'I' next?
                if raw_arr[i+3]==79:  #is 'O' next?
                    ind = np.append(ind, i)  #if so, save index for header 'ASIO'
    
    # Seperate each packet
    packet_arr = np.zeros(950)  #pack zeros for np.vstack
    for i in ind:
        packet = raw_arr[i:i+950]
        if len(packet) == 950:
            packet_arr = np.vstack(( packet_arr, packet ))
        else:
            print(f'Packet removed. Insufficient Packet Size at array[{i}]. Needs 950 Bytes, received {len(packet)}.')
    packet_arr = packet_arr[1:].astype(int)
    
    # Seperate Data
    headers = packet_arr[:,:4]            #header remained the same from old to new
    #health_A = packet_arr[:,4:32]        #this was the old packet structure (not flight code)
    health_A = packet_arr[:,4:44]         #new packet structure, flight code
    #data_arr = packet_arr[:,32:932]      #this was the old packet structure (not flight code)
    data_arr = packet_arr[:,44:944]       #new packet structure, flight code
    #health_B = packet_arr[:,932:]        #this was the old packet structure (not flight code)
    health_B = packet_arr[:,944:]         #new_packet structure, flight code

    # Rearrange bytes into measurements

    bytes_arr = np.reshape(data_arr, (len(headers), 300, 3))

    # Seperate each channel
    channels_arr = np.stack((bytes_arr[:,0::6,:], 
                             bytes_arr[:,1::6,:], 
                             bytes_arr[:,2::6,:], 
                             bytes_arr[:,3::6,:], 
                             bytes_arr[:,4::6,:], 
                             bytes_arr[:,5::6,:]))
    # Print statement to check the MSB of each voltage value in each channel
    #ind = np.array([])
    #for i in channels_arr[:,:,:,0].flatten():
        #if i != 128:
            #ind = np.append(ind, i)
    #print(ind)

    # Combine bytes for each measurement
    ADC_arr = ((2**16) * channels_arr[:,:,:,2] +
               (2**8) * channels_arr[:,:,:,1] + 
               (2**0) * channels_arr[:,:,:,0])
    # Convert bytes into voltages
    V_arr = np.vectorize(ADC_to_V)(ADC_arr)

    # Build Data dictionary
    data_dict = {}
    data_dict['SXR1'] = V_arr[0].flatten()
    data_dict['SXR2'] = V_arr[1].flatten()
    data_dict['SXR3'] = V_arr[2].flatten()
    data_dict['SXR4'] = V_arr[3].flatten()
    data_dict['HXR'] = V_arr[4].flatten()
    data_dict['EUV'] = V_arr[5].flatten()
    data_dict['Health1'] = health_A
    data_dict['Health2'] = health_B

    return data_dict

#converts dictionary to a csv titled output.csv excluding Health1 and Health2 keys
def dictionary_to_csv(data_dict, filename = 'output.csv'):
    exclude_keys = {'Health1', 'Health2'}
    filtered_dictionary = {k: v for k, v in data_dict.items() if k not in exclude_keys}
    df = pd.DataFrame.from_dict(filtered_dictionary)
    df = df.T
    df.to_csv(filename, header = None)
    print(f'Saved {filename}')

#test call
#get_data_dict("../../Data/06_EM_Testing_V2_DB_Training/Test_data/ASIO-2025_09_15-11_48_26_AM-test4.csv")

