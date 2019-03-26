import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from sklearn.preprocessing import normalize

from gwpy.timeseries import TimeSeries

import pywt

LABEL_MAP = {
    'Scattered_Light': 0,
    'Repeating_Blips': 1,
    'Violin_Mode': 2,
    'Power_Line': 3,
    'Whistle': 4,
    'Scratchy': 5,
    'Helix': 6,
    'Light_Modulation': 7,
    'Wandering_Line': 8,
    'Low_Frequency_Burst': 9,
    'Koi_Fish': 10,
    'Low_Frequency_Lines': 11,
    'Blip': 12,
    '1400Ripples': 13,
    'Chirp': 14,
    'Extremely_Loud': 15,
    'None_of_the_Above': 16,
    'Paired_Doves': 17,
    'Tomte': 18,
    'Air_Compressor': 19,
    '1080Lines': 20,
    'No_Glitch': 21,
}

class dataToolkit(object):
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(os.getcwd()),"data")
        self.metadata_filename = "gspy-db-20180813_O1_filtered_t1126400691-1205493119_snr7.5_tr_gspy.csv"
        self.metadata_df = pd.read_csv(os.path.join(data_dir,self.metadata_filename))

    def formatMetaData(self):
        self.train = []
        self.y = []
        #np.asarray(someListOfLists, dtype=np.float32)
        for line in self.metadata_df.values:
            self.train.append(line[1:5].tolist())
            self.y.append(LABEL_MAP[line[8]])

        self.train = np.asarray(self.train)
        self.norm_train = normalize(self.train)
        self.y = np.asarray(self.y)

    def getTimeSeries(self, glitch_name, idx=0):
        found = False
        glitch_df = self.metadata_df.loc[self.metadata_df["label"]==glitch_name]
        if(len(glitch_df) <= idx):
            #glitch_id= glitch_df.iloc[len(glitch_df)-1]["id"]
            #print("Index is greater than max number of glitches")
            return False
        else:
            glitch_id= glitch_df.iloc[idx]["id"]

        data_dir_hdf5 = os.path.join(os.path.dirname(os.getcwd()),"data") + "/hdf5"
        for file in os.listdir(data_dir_hdf5):
            if glitch_id in file:
                self.h5=h5py.File(os.path.join(data_dir_hdf5, file), 'r')
                found = True
        
        if(found):
            self.strain = self.h5["Strain"]["Strain"].value
        else:
            print("Did not update strain!")
        return True

    def plotTimeSeries(self):
        self.GPSstart = self.h5["Strain"]["Strain"].attrs["GPSstart"]
        self.GPSend = self.h5["Strain"]["Strain"].attrs["GPSend"]
        self.Sample_Rate = self.h5["Strain"]["Strain"].attrs["Sample_Rate"]
        ts = 1./self.Sample_Rate
        half = (self.GPSend-self.GPSstart)/2.
        self.time = np.arange(-half,half,ts)

        f = plt.figure()
        sp = f.add_subplot(111)
        sp.plot(self.time, self.strain)
        return f

    def whitenStrain(self):
        strain_gw = TimeSeries(self.strain)
        self.whitened = strain_gw.whiten(4,2).value

    def plotWhitenStrain(self):
        f = plt.figure()
        #f.plot(self.time, self.strain)
        #f.xlabel("Time")
        #f.ylabel("Strain")
        sp = f.add_subplot(111)
        sp.plot(self.time, self.whitened)
        return f

    def getExtraFeatures(self):
        t = True
        i = 0
        count = 0
        cAmean_list = []
        cAstd_list = []
        cDmean_list = []
        cDstd_list = []
        for key in LABEL_MAP:
            i = 0
            t = True
            while(t == True):
                t = self.getTimeSeries(key, i)
                cAmean, cD3mean, cD2mean, cD1mean = self.getWaveletIndicators(self.strain)
                cAmean_list.append(cAmean)
                cAstd_list.append(cD3mean)
                cDmean_list.append(cD2mean)
                cDstd_list.append(cD1mean)
                i+=1
                count += 1
                if(count == 6667):
                    break
            if(count == 6667):
                break
                
        return cAmean_list, cDmean_list, cAstd_list, cDstd_list
    
    def getWaveletIndicators(self, ts):
        normalized=(ts-ts.min())/(ts.max()-ts.min())
        #print(normalized)
        np.nan_to_num(normalized, copy=False)
        cA, cD3, cD2, cD1 = pywt.wavedec(normalized*10000, 'dmey', level=3)
        return np.mean(cA), np.mean(cD3), np.mean(cD2), np.mean(cD1)
    
    def close(self):
        self.h5.close()
        
     
