import os
import biosppy
import parfor

import vitaldb as vdb
import matplotlib.pyplot as plt
import numpy as np
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values,get_time_domain_features

import datetime
import pandas as pd

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    search_dir = '/Users/madamek/Box/taVNS_clinical/'
    vit_subject_dir= os.listdir(search_dir)
    for subj in vit_subject_dir:
        if not subj.isnumeric():
            continue
        vit_files_dir = os.listdir(os.path.join(search_dir, subj))
        complete_data = []
        for vit_dir in vit_files_dir:
            vit_files=os.scandir(os.path.join(search_dir, subj, vit_dir))
            for vital_filename in vit_files:
                #vital_data=vitaldb(vital_file)

                if not vital_filename.name.endswith('.vital'):
                    continue

                print('Loading ', os.path.join(search_dir, subj, vit_dir, vital_filename))

                try:
                    vital_file = os.path.join(search_dir, subj, vit_dir, vital_filename)
                    vital_data = vdb.VitalFile(vital_file)
                    ecg_settings = vital_data.find_track('ECG_I')
                    sampling_rate = ecg_settings['srate']
                    ecg_I = vital_data.get_samples('ECG_I', 1/sampling_rate)
                    ecg_II = vital_data.get_samples('ECG_II', 1/sampling_rate)
                    ecg_III = vital_data.get_samples('ECG_III', 1/sampling_rate)
                    ch1 = vital_data.get_samples('CH1', 1/sampling_rate)
                    #important identifiers CH1, ECG_I, ECG_II, ECG_III
                    #each file starts with nan values, we need to remove them first!
                    ecg=[]
                    if subj == '2020004': # check later, ecg rythm looks abnormal, no R in ECG_I
                        ecg = ecg_II
                    else:
                        ecg = ecg_I
                    nan_time = np.isnan(ecg)
                    ECG_nonan = ecg[~nan_time]
                    #ch1_aligned = ch1[~nan_time]

                    signal, rpeaks = biosppy.signals.ecg.ecg(ECG_nonan, sampling_rate=sampling_rate, show=False)[1:3] #https://pyhrv.readthedocs.io/en/latest/_pages/start.html#r-peak-detection-with-biosppy
                    # rri (ms) =  samplingrate / (bpm / 60)
                    print("Number of rpeaks detected: " + str(len(rpeaks)))
                    rr_intervals = np.diff(rpeaks)
                    rr_intervals = rr_intervals*1e3/sampling_rate # convert to ms
                    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals)

                    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,
                                                                       interpolation_method="linear")
                    nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="malik")
                    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)

                    time_domain_features = get_time_domain_features(interpolated_nn_intervals)
                    #print(time_domain_features)
                    #plot_psd(interpolated_nn_intervals, method="welch")
                    dts = datetime.datetime.fromtimestamp(vital_data.dtstart)
                    time_domain_features['time'] = dts
                    time_domain_features['stim'] = max(ch1)
                    complete_data.append(time_domain_features)
                except Exception as ex:
                    print(ex)
                    pass

        pd_data = pd.DataFrame(complete_data)
        plt.subplot(211)
        plt.scatter([pd.Timestamp(i).to_pydatetime().hour for i in pd_data['time'].values], pd_data['sdnn'].values)
        plt.subplot(212)
        plt.scatter([pd.Timestamp(i).to_pydatetime().hour for i in pd_data['time'].values], pd_data['stim'].values)
        plt.show()
        pd_data.to_pickle(subj)
