
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#import uproot
import pandas as pd
import statistics as st
import os
from numba import njit
import time
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from loguru import logger
from datetime import datetime
from numba.typed import List

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

for arg in sys.argv:
    run_num=str(arg).zfill(5)
    # print(run_num)

chan_enab = int(sys.argv[-1])
run_start=str(sys.argv[1]).zfill(5)
run_end=str(sys.argv[2]).zfill(5)
run_num=str(sys.argv[3]).zfill(5)

# run_num = '12034'
# runs_folder = 'runs12034-12363/'
# os.chdir('F:/LANL/')
# datadir = 'sample_data/'
# uniquefolder = 'debug_sample/'+runs_folder
# SFNormFile = 'SF_Norm_files/'+runs_folder+run_num
# statefileloc = 'F:\LANL\SF_Norm_files\TR_R_expected_avgs_stds_afterclip.csv'

# # print(os.getcwd())

os.chdir('F:/LANL/')
datadir = 'D:/LANSCE_FP12_2023/data/' ## add directory of hard drive
uniquefolder = "runs" + str(run_start) + "-" + str(run_end) +"/"
# SFNormFile = 'SF_Norm_files/'+uniquefolder+run_num

# statefileloc = 'F:\LANL\SF_Norm_files\TR_R_expected_avgs_stds_afterclip.csv'
processederrorfolder = '/processed_data/'+uniquefolder+'error_D/'
errorSavename = processederrorfolder+run_num+'_error_D'
logger.add("F:/LANL/processed_data/" + uniquefolder + '1_ErrorLog_'+run_start+'_'+run_end+'_gcount_D.txt', delay = False)
print('processing data: ' + uniquefolder + '/run' + run_num)
print('saving data to: ' + os.getcwd()+errorSavename)

## cannot handle all 24 detectors at once, memory issue... can look into np.empty and deleting variables if needed<br>
# chan_enab = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]) ## all
chan_enab = np.array([0,1,2,3,4,5,6,7,8,9,10,11,24]) ## downstream
# chan_enab = np.array([12,13,14,15,16,17,18,19,20,21,22,23,24]) ## upstream

if not os.path.exists(os.getcwd()+processederrorfolder):
    # Create the directory
    os.makedirs(os.getcwd()+processederrorfolder)
    print("Directory created successfully")
else:
    pass

start = time.time()
fullstart = time.time()

# read_data = np.array([])
# fileLength = np.array([])
read_data = []
fileLength = []

def open_file():
    for el in chan_enab:
        # f = open(datadir + folder + 'run' + run_num + "_ch" + str(el) + ".bin", 'rb')
        f = open(datadir+uniquefolder + 'run' + str(run_num) + "_ch" +str(el) + ".bin", 'rb')
        read_data.append(np.fromfile(file=f, dtype=np.uint16))
        f.close()
        fileLength.append(len(read_data[-1]))
    return read_data, fileLength

open_file()

fileLength = np.asarray(fileLength)
read_data = np.asarray(read_data) ## in detector's case, all are the same size samples, so can do read_data as np array

if chan_enab[-1] != 24:
    emessage = ('last channel is not 6Li detector')
    logger.error(run_num + emessage)
    raise Exception(emessage)

# print('saving processed data to ' + AsymSavename)
print("Channel is " + str(chan_enab))
end = time.time()
# print('file open time: ' + str(end-start))      
# print(read_data)

# Store the big header for each channel in arrays

# In[4]:

BoardID = []
recordLength = []
numSamples = []
eventCounter = []
decFactor = []
chanDec = []
postTrig = []
groupStart = []
groupEnd = []
timestamp= []
sizeFirstEvent = []
TTT = []

targetDict = {0: "La", 1: "Tb2O3", 2: "Yb2O3", 3: "Sm2O3", 4: "Er2O3", 5: "Ho2O3", 6: "other"}
foilDict = {0: "TBD", 1: "TBD", 2: "TBD", 3: "TBD", 4: "TBD", 5: "TBD", 6: "other"}

target=(read_data[0][5]&0x00F0)>>4
foil=read_data[0][5]&0x000F
targetFlag = read_data[0][5]>>8&1
foilFlag = read_data[0][5]>>9&1
spinFiltFlag = read_data[0][5]>>10&1
spinFlipFlag = read_data[0][5]>>11&1
shutterFlag = read_data[0][5]>>12&1
facilityTrigFlag = read_data[0][5]>>13&1

if targetFlag:
    target=targetDict[(read_data[0][5]&0x00F0)>>4]
else:
    target = "empty"
if foilFlag:
    foil=foilDict[read_data[0][5]&0x000F]
else:
    foil = "empty"
for i in range(0,len(chan_enab)):
    BoardID.append(read_data[i][9]>>8)
    recordLength.append(((read_data[i][9]&0x00FF)<<16)+read_data[i][8])
    numSamples.append(((read_data[i][11]&0x00FF)<<16)+read_data[i][10])
    eventCounter.append(read_data[i][6]+(read_data[i][7]<<16))
    BoardID.append(read_data[i][9]>>8)  
    decFactor.append(read_data[i][11]>>8)
    chanDec.append(read_data[i][13]>>8)
    postTrig.append(read_data[i][15]>>8)
    groupStart.append(((read_data[i][13]&0x00FF)<<16)+read_data[i][12])
    groupEnd.append(((read_data[i][15]&0x00FF)<<16)+read_data[i][14])
    timestamp.append(read_data[i][16]+(read_data[i][17]<<16)+(read_data[i][18]<<32)+(read_data[i][19]<<40))  
    sizeFirstEvent.append(read_data[i][0]+(read_data[i][1]<<16))
    TTT.append(read_data[i][2]+(read_data[i][3]<<16)+(read_data[i][4]<<32))
    
#     print("For channel " + str(chan_enab[i]) + ", BoardID is " + str(BoardID[i])
#           + "; record length is " + str(recordLength[i]) + "; num Samples is " 
#           + str(numSamples[i]) + "; event counter is " + str(eventCounter[i]) + "; dec factor is " + str(decFactor[i]) + "; chan dec is " 
#           + str(chanDec[i]) + "; postTrig is " + str(postTrig[i]) + "; group start is " + str(groupStart[i]) + "; group end is " + str(groupEnd[i])
#           + "; epoch time is " + str(timestamp[i]) +  "; first event size is " + str(sizeFirstEvent[i]) + "; and ETTT is " + str(TTT[i]) + "\n")

BoardID = np.asarray(BoardID) 
recordLength = np.asarray(recordLength)
numSamples = np.asarray(numSamples)
eventCounter = np.asarray(eventCounter)
decFactor = np.asarray(decFactor)
chanDec = np.asarray(chanDec)
postTrig = np.asarray(postTrig)
groupStart = np.asarray(groupStart)
groupEnd = np.asarray(groupEnd)
timestamp = np.asarray(timestamp)
sizeFirstEvent = np.asarray(sizeFirstEvent)
TTT = np.asarray(TTT)
print("Target is " + target)
# print("Foil is " + foil)
# print("Shutter is open: " + str(bool(shutterFlag)))
# print("Facility t0 is on: " + str(bool(facilityTrigFlag)))
# print("Spin flipper is on: " + str(bool(spinFlipFlag)))
# print("Spin filter is on: " + str(bool(spinFiltFlag)))
# print("Target is present: " + str(bool(targetFlag)))
# print("Foil is present: " + str(bool(foilFlag)))

# Determine the time axis for each channel

# In[5]:

preTime = []
startTime = []
endTime = []
resolution = []
xs = [] 

for i in range(0,len(chan_enab)):
    preTime.append((100-postTrig[i])*recordLength[i]/100)
    startTime.append((-1*preTime[i]*16*decFactor[i] + groupStart[i]*16*decFactor[i]))
    endTime.append((-1*preTime[i]*16*decFactor[i] + groupEnd[i]*16*decFactor[i]))
    resolution.append(16*chanDec[i]*decFactor[i])
#     print("Pretime for channel", chan_enab[i],"is " + str(preTime[i]) + "; start time is " + str(startTime[i]) + "; end time is " + str(endTime[i]) 
#           + "; resolution is " + str(resolution[i]) + "ns")
    xs.append(np.arange(startTime[i],(numSamples[i])*resolution[i]+startTime[i], resolution[i]))

# np.asarray(preTime) 
# np.asarray(startTime) 
# np.asarray(endTime) 
# np.asarray(resolution)
xs = np.asarray(xs) ## can convert xs to np array here because all detectors same numsamples

# In[6]:

start=time.time()

@njit
def dataread(data, channels, fileLen, numSamps):
    numRuns = int((fileLen[0]-20-numSamps[0])/(numSamps[0]+6)+1)
    ys_arr = np.zeros((len(channels), numRuns,numSamps[0]), dtype=np.uint16)
    ETTT_arr = np.zeros((len(channels), numRuns), dtype=np.intc)
    eventcount_arr = np.zeros((len(channels), numRuns), dtype=np.intc)
    for i in range(0,len(channels)):
        eventCount = 0
        byteCounter = 0
            #byte counter is really 2bytecounter, lol
        while byteCounter < fileLen[i]:
            if byteCounter == 0:
                ETTT_arr[i]=TTT[i]
                #ETTT_arr[i].append(TTT[i])
                eventcount_arr[i]=(eventCounter[i])
                byteCounter = 20
            else:
                ETTT_arr[i]=(data[i][byteCounter]+(data[i][byteCounter+1]<<16)+(data[i][byteCounter+2]<<32))
                eventcount_arr[i]=(data[i][byteCounter+4]+(data[i][byteCounter+5]<<16))
                byteCounter += 6
            for j in range(0, numSamps[i]):
                #if j == 0:
                    #ys_arr[i].append([])
                #print(byteCounter)
                ys_arr[i][eventCount][j]=data[i][byteCounter]
                byteCounter += 1
            eventCount += 1
    return ys_arr, ETTT_arr, eventcount_arr

# start=time.time() 
# ys_arrHe, ETTT_arrHe, eventcount_arrHe  = dataread(read_data, [25], fileLength, numSamples) ##hardcoded channel 25 for He
ys_arr, ETTT_arr, eventcount_arr  = dataread(read_data, chan_enab, fileLength, numSamples) ##hardcoded channels for coils

end = time.time()
print('dataread from binary time: ' + str(end-start))

# In[7]:

timeDif=[]
for i in range(0,len(chan_enab)):
    timeDif.append([])
    for j in range(len(ETTT_arr[i])-1):
        timeDif[i].append((ETTT_arr[i][j+1]-ETTT_arr[i][j])*8)
#     print("Min time difference for channel", chan_enab[i], "is", min(timeDif[i]), "ns")
#     print("Max time difference for channel", chan_enab[i], "is", max(timeDif[i]), "ns \n")
#print(timeDif)

# In[8]:
# basesub and plotting ##

baseL = 0
baseR = int(((preTime[0]-groupStart[0])*0.70)/chanDec[0])  ##70% before the trigger
numRuns = int((fileLength[0]-20-numSamples[0])/(numSamples[0]+6)+1)
legend =  ['NaI', 'R']

start = time.time()

s = 20 ## pulse to look at 
t=s+1

#  dont know why this is so slow ##
def plotter(ys, xs, baseR, numpoints):
    tempys_basesub = np.zeros((len(ys), numRuns,numpoints[0]), dtype=float)
    for i in range((len(ys))):
        for pulse in range((len(eventcount_arr[0]))): ## all have 5000 pulses
            tempys_basesub[i][pulse]=np.subtract(ys[i][pulse], np.mean(ys[i][pulse][baseL:baseR]))
        for j in range(s, t): ## plot only interested pulses
            plt.plot(xs[i], tempys_basesub[i][j]) #label=legend[i]) #+str(sums[1][j])) ## sums[j] will not work for more than just TR   
            plt.axvline(xs[0][baseL], ls = '--')
            plt.axvline(xs[0][baseR], ls = '--')
            #plt.axvline(xs[0][int(((preTime[0]-groupStart[0])*0.70)/chanDec[0])], ls = '--', c ='m')
            plt.axvline(xs[0][baseR+5], ls = '--', c ='r') ## BaseR+5 line marks the beginning of the integral, until the end of samples.
#             plt.title('SF state transition' + transitions[p]) 
#             plt.xlabel("time from trigger (ns)")
#             plt.ylabel("ADC")
#             plt.legend()
            
# plotter(ys_arr[9:], xs[9:], baseR, numSamples) ##plot coils

ys_basesub = np.zeros((len(ys_arr), numRuns,numSamples[0]), dtype=np.float64)
# ys_basesub_norm = np.zeros((len(ys_arr), numRuns,numSamples[0]), dtype=np.float64)

@njit ## jit is faster for large # channels, slower for small # channels
def basesub(ys, baseRight, numpoints):
#     uQ_sec = ((2/4096)/50)*1000000 ## 4096 ADC = 2V, divide by 50Ohm to get I [Q/sec], change to microQ
#     ys = ys*uQ_sec
    tempys_basesub = np.zeros((numRuns,numpoints[0]), dtype=np.float64)
    for pulse in range((len(eventcount_arr[0]))): ## all have 5000 pulses
        tempys_basesub[pulse]=np.subtract(ys[pulse], np.mean(ys[pulse][baseL:baseRight]))
    return tempys_basesub

@njit ## jit is faster for large # channels, slower for small # channels
def basesub_norm(ys, baseRight, numpoints): 
    tempys_basesub = np.zeros((numRuns,numpoints[0]), dtype=np.float64)
#     uQ_sec = ((2/4096)/50)*1000000 ## 4096 ADC = 2V, divide by 50Ohm to get I [Q/sec], change to microQ
#     ys = ys*uQ_sec
    for pulse in range((len(eventcount_arr[0]))): ## all have 5000 pulses
        tempys_basesub[pulse]=np.subtract(ys[pulse], np.mean(ys[pulse][baseL:baseRight]))
        tempys_basesub[pulse]=tempys_basesub[pulse]/HeNorms[pulse] 
    return tempys_basesub

# for peak finding algo., we don't want to use the normalization yet...
for i in range(len(ys_basesub)): ## feeding y arrays into function 1 channel at  a time is faster than all at once
    ys_basesub[i] = basesub(ys_arr[i], baseR, numSamples)
# for i in range(len(ys_basesub)): ## if not using aligning/cutting later, ys should be normalized here
#     ys_basesub[i] = basesub_norm(ys_arr[i], baseR, numSamples)

ys_basesub[-1] = ys_basesub[-1]*-1 ## invert 6Li to positive signal. Comment out if not using

end = time.time()
print('plotting and/or base subtraction time: ' + str(end-start))            

# ### find peaks, integrate

# In[12]:

# peak lengths are defined per pulse after being found; can maybe using numpy after that??
# split into 2 functions to use JIT?

# @njit
def find_peaks_np(ys, pthresh, peakrange): ## ys here is for one channel!
    all_peaks = []
    sum_ranges = []  ## all sum ranges  (i.e. for peak at bin 7, sum from bins 5-9. Save this for every peak)
    for p in range(len(ys)):  ## find peaks for every pulse p in one detector channel
        peaks, _ = sp.signal.find_peaks(ys[p], threshold=param[0], prominence=param[1], height=param[2])#, height  = [1,1000])
        temp_i = np.array(np.where((peaks>=peakrange[0]) & (peaks<=peakrange[1]))[0], copy=True)
        peaks_i = peaks[temp_i]
        p_sum_ranges = np.zeros((len(peaks_i), 2), dtype = int)
        if peaks_i[0]-2<peakrange[0]: ## plus and minus 2 x points (512 ns each) was determined from peak widths below
            peaks_i = peaks_i[1:].copy()
            if peaks_i[-1]+2>peakrange[1]: ## this is in case both peak at beginning at end. Couldn't find a more elegant solution
                peaks_i = peaks_i[:-1].copy()
        elif peaks_i[-1]+2>peakrange[1]:
            peaks_i = peaks_i[:-1].copy()
        else:
            peaks_i = peaks[temp_i]
        p_sum_ranges = np.zeros((len(peaks_i), 2), dtype = int)
        for peak in range(len(peaks_i)):
            p_sum_ranges[peak][0] = peaks_i[peak]-2
            p_sum_ranges[peak][1] = peaks_i[peak]+2#plus minus 2 bin range around found peak
        sum_ranges.append(p_sum_ranges)
    return sum_ranges #, all_peaks ## don't necessarily need all_peaks array, keep it for troubleshoot

# In[14]:

start = time.time()
peakthresh = 2 ## threshold for peaks is estimated
prom = 10
h = [0,750] 
param = [peakthresh, prom, h] ## all peak finding parameters

peak_beg = 1399
peak_end = 7599
peak_range_beg = [0, peak_beg] ## hardcoded for 0-1400 and 7600-9000
peak_range_end = [peak_end, 8999]

sum_ranges_beg = np.zeros((len(ys_basesub[:-1]), len(ys_basesub[0])), dtype=object) ## 13 channels, 13 sequences, added pulses for ON
sum_ranges_end = np.zeros((len(ys_basesub[:-1]), len(ys_basesub[0])), dtype=object) ## 13 channels, 13 sequences, added pulses for ON

for i in range(len(sum_ranges_beg)):
#     print('channel ' + str(chan_enab[i]))
    sum_ranges_beg[i] = find_peaks_np(ys_basesub[i], param, peak_range_beg)
    sum_ranges_end[i] = find_peaks_np(ys_basesub[i], param, peak_range_end)

end = time.time()
print('finding peaks time: ' + str(end-start))

# try modifying add_pulse function to instead add points in same pulse

# In[15]:

def integrate_peaks(ys, int_ranges): ## ranges in which to integrate int_ranges = [[2,6], [6,10]], 2 len array for each point
    peak_integrals = []
    for p in range(len(int_ranges)):  ## find integral of peaks for every pulse p in one detector channel
        peak_ints = np.zeros((len(int_ranges[p])), dtype = np.float64)
        for pk in range(len(int_ranges[p])):  ## size of the peaks for every pulse
            start_point = int_ranges[p][pk][0] ## try this to condense code. Basically, the beginning of where to integrate
            end_point = int_ranges[p][pk][1]
            for point in range(start_point,end_point+1): ##From 20-60 for example. SFarr[2] is the array of start to end pulses to sum
                peak_ints[pk] = np.add(peak_ints[pk],ys[p][point]) ## start with zeros, add to each iteratively
        peak_integrals.append(peak_ints)
    return peak_integrals
    
# In[16]:

start = time.time()

ints_beg = np.zeros((len(sum_ranges_beg), len(sum_ranges_beg[0])), dtype=object) ## 13 channels, 13 sequences, added pulses for ON
ints_end = np.zeros((len(sum_ranges_end), len(sum_ranges_end[0])), dtype=object) ## 13 channels, 13 sequences, added pulses for ON

for i in range(len(sum_ranges_beg)):
    # print('channel ' + str(chan_enab[i]))
    ints_beg[i] = integrate_peaks(ys_basesub[i], sum_ranges_beg[i])
    ints_end[i] = integrate_peaks(ys_basesub[i], sum_ranges_end[i])
    
end = time.time()
print('integrating peaks time: ' + str(end-start))

# In[21]:

plt.ioff()
start = time.time()

Q_sec = ((2/4096)/50) ## 4096 ADC = 2V, divide by 50Ohm to get I [Q/sec]
sec = (512*(10e-9))

ints_beg_all = [] ## all pulse mode integrals for a given channel
ints_end_all = []

for i in range(len(ints_beg)):
    ints_beg_all.append(np.hstack(ints_beg[i]*Q_sec*sec))
    ints_end_all.append(np.hstack(ints_end[i]*Q_sec*sec))
    
histdat_beg = [] ## all pulse mode histograms from the integrals
histdat_end = []
    
for i in range(len(ints_beg_all)):
    beghist = ints_beg_all[i]*10e6
    endhist = ints_end_all[i]*10e6
    beg_binval = plt.hist(beghist, bins = 200, range = [0, 0.60]) ## this is a 2d array of bin y values and their bin locations
    end_binval = plt.hist(endhist, bins = 200, range = [0, 0.60])
    histdat_beg.append(beg_binval)  ## 2d array of [[counts values], [bin locations (1 extra)], [something lol]]
    histdat_end.append(end_binval)

end = time.time()
print('making histograms, changing to [C]: ' + str(end-start))

# In[18]:

## maybe include peak locations? but this would be a ch*pulse*# peaks sized array, very long

cols = ['channel', 'all_begregion_integrals', 'all_endregion_integrals', 'begregion_hist [uQ]', 'endregion_hist [uQ]']
intsData = [chan_enab[:-1], ints_beg_all, ints_end_all, histdat_beg, histdat_end] ## don't include 6Li channel

df_ints = pd.DataFrame({cols[0]: intsData[0],            
                    cols[1]: intsData[1],
                    cols[2]: intsData[2],
                    cols[3]: intsData[3],
                    cols[4]: intsData[4]})

# print(df_ints)

# In[19]:

# df_ints.to_hdf('/processed_data/runs12034-12363/error_D/testingtesting_error_D'+ '.h5', f'df_0', mode='w') ## this "deletes" any previous data in the file name
df_ints.to_hdf(os.getcwd()+errorSavename + '.h5', f'df_0', mode='w') ## this "deletes" any previous data in the file name
fullend = time.time()
print('gamma region analysis done, full time: ' + str(fullend-fullstart))
print('finished ' + str(datetime.now())) 
print('\n')

# In[59]:


# ### testing end historgam - beg histogram

# testarr1 = df_ints['begregion_hist [uQ]'].to_numpy()
# testarr2 = df_ints['endregion_hist [uQ]'].to_numpy()
# # print(testarr1[0])
# # print(len(testarr1[0]))

# ch = 0
# print(len(testarr2[ch]))

# # print(testarr2[0][0])
# testxbeg = testarr1[ch]
# testxend = testarr2[ch]
# testx = testxend

# testend_beg = testxend[0]-testxbeg[0]
# plt.bar(testx[1][:-1], testend_beg, width = max(testx[1])/len(testx[0]))  ## this plots the previous histogram !!!
# plt.xlabel('uQ')
# plt.title('beg region pulse integrals run ' + run_num)
# plt.show() 


# In[ ]:




