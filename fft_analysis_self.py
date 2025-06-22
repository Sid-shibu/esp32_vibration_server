import pandas as pd
import numpy as np
import scipy as sp

#importing file by asking user
filename=input("Enter the filename with extension(like 123.csv)")
df=pd.read_csv(filename)

#extracting values for each aceelerometer
#for accelo1 assigning to thumb
thumb_x= df['acc1_x'].values
thumb_y=df['acc1_y'].values
thumb_z=df['acc1_z'].values

#for accelo2 assigning to forefinger

fore_x=df['acc2_x'].values
fore_y=df['acc2_y'].values
fore_z=df['acc2_z'].values

#for accelo3 assigning to middlefinger

middle_x=df['acc3_x'].values
middle_y=df['acc3_y'].values
middle_z=df['acc3_z'].values

#compute magnitude
accel_mag_1=np.sqrt(thumb_x**2+thumb_y**2+thumb_z**2)
accel_mag_2=np.sqrt(fore_x**2+fore_y**2+fore_z**2)
accel_mag_3=np.sqrt(middle_x**2+middle_y**2+middle_z**2)

#Bandpass filtering
from scipy.signal import butter, filtfilt
def bandpass_filter(data,fs=250,low=4,high=12,order=4):
    nyquist=fs*0.5
    low_norm=low/nyquist
    high_norm=high/nyquist
    b,a=butter(order,[low_norm,high_norm],btype='band')
    return filtfilt(b,a,data)

#computing fft
def compute_fft(filtered_signal,fs=250):
    N=len(filtered_signal)
    window=np.hanning(N)
    windowed_signal=filtered_signal*window
    fft_result=np.fft.fft(windowed_signal)
    freq=np.fft.fftfreq(N,1/fs)[:N//2]
    magnitudes=2.0/N*np.abs(fft_result[:N//2]) #2.0/N is normalisation factor
    return freq,magnitudes

#finding peaks from fft for tremor detection
from scipy.signal import find_peaks
def detect_peaks(freq,magnitudes,min_height=0.1):
    peaks, _=find_peaks(magnitudes,height=min_height,prominence=0.05)
    tremor_peaks = [(freq[i], magnitudes[i]) for i in peaks if 4 <= freq[i] <= 12]
    return tremor_peaks
# Categorizing tremor peaks
def categorize_tremor_peaks(tremor_peaks):
    categories = []
    for freq, mag in tremor_peaks:
        if mag > 0.5:  # Example threshold for amplitude
            if 4 <= freq < 6:
                categories.append((freq, mag, "Low Frequency Tremor"))
            elif 6 <= freq < 8:
                categories.append((freq, mag, "Medium Frequency Tremor"))
            elif 8 <= freq <= 12:
                categories.append((freq, mag, "High Frequency Tremor"))
        else:
            categories.append((freq, mag, "Insignificant Tremor"))
    return categories
# analysing the inputted data using various functions
def analyse_tremors():
    filtered_mag_1=bandpass_filter(accel_mag_1)
    filtered_mag_2=bandpass_filter(accel_mag_2)
    filtered_mag_3=bandpass_filter(accel_mag_3)

    freq1,mag1=compute_fft(filtered_mag_1)
    freq2,mag2=compute_fft(filtered_mag_2)
    freq3,mag3=compute_fft(filtered_mag_3)

    tremor_1=detect_peaks(freq1,mag1)
    tremor_2=detect_peaks(freq2,mag2)
    tremor_3=detect_peaks(freq3,mag3)

    # Categorize detected tremor peaks
    categorized_tremor_1 = categorize_tremor_peaks(tremor_1)
    categorized_tremor_2 = categorize_tremor_peaks(tremor_2)
    categorized_tremor_3 = categorize_tremor_peaks(tremor_3)

    # Print categorized tremor peaks
    print("Categorized Tremor Peaks for Thumb:")
    for peak in categorized_tremor_1:
        print(peak)
    print("\nCategorized Tremor Peaks for Forefinger:")
    for peak in categorized_tremor_2:
        print(peak)
    print("\nCategorized Tremor Peaks for Middle Finger:")
    for peak in categorized_tremor_3:
        print(peak)
# Run the analysis

analyse_tremors()

