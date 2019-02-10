"""
        Data Preprocessing for EEG signal obtained from Bern Barcelona Dataset

        1) Detrending the signal 

        2) Butterworth filter

        3) Wavelet Decomposition

        4) Entropy feature extraction

                i)Wavelet based (time-frequency domain)
                a) Sample Entropy
                b) Shannon Entropy
                c) Permutation Entropy
                d) Renyi Entropy
                e) Kraskov entropy

                ii)Signal time domain features:
                f) Inter Quartile
                g) Activity
                h) Mobility
                i) Complexity
                j) Hurst Exponent
                k) DFA
                l) Petrosian Fractal Dimension

5) Standardize the data
"""
#import all required libraries
import numpy,glob,errno,pandas,pywt
from scipy.signal import butter, lfilter, detrend

#Butterworth bandpass 
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
#Bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

path="Data_NFocal_Full/*.txt"
files=glob.glob(path)
files=sorted(files)

db = pywt.Wavelet('db10')               #Mention the type of Mother Wavelet
a=[];d4=[];d3=[];d2=[];d1=[]

lowcut=0.5;highcut=60                   #EEG signal of range 0.5 Hz to 60 Hz is ideal and preserves informative features others are Noise

for i in range(len(files)):                             
    print(i,end=' ',flush=True)
    X=pandas.read_csv(files[i],sep=",",header=None)
    X[2]=X[0]-X[1]
    x=X[2].values
    x=detrend(x)                                                #Detrend the signal for removing trend in the signal by sensors
    x = butter_bandpass_filter(x, lowcut, highcut,512, order=4)         #Butterworth filter 0.5-60 Hz 4th Order

    cA,cD4,cD3,cD2,cD1= pywt.wavedec(x,db,level=4)                      #Decompose the signal for 4 levels by DWT
    #store the DWT coeffecients
    a.append(cA)
    d4.append(cD4)
    d3.append(cD3)
    d2.append(cD2)
    d1.append(cD1)
#save the coeffecients in .csv file
numpy.savetxt("Db10NFZD1.csv", d1, delimiter=",")
del(d1)
numpy.savetxt("Db10NFZD2.csv", d2, delimiter=",")
del(d2)
numpy.savetxt("Db10NFZD3.csv", d3, delimiter=",")
del(d3)
numpy.savetxt("Db10NFZD4.csv", d4, delimiter=",")
del(d4)
numpy.savetxt("Db10NFZA4.csv", a, delimiter=",")
del(a)
