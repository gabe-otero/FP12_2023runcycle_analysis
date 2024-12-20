import sys
import numpy as np
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
import h5py
from scipy import odr

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

# run_num = '12036'
# os.chdir('F:/LANL/')
# datadir = 'sample_data/'
# runs_folder = 'runs12034-12363/'
# uniquefolder = 'debug_sample/'+runs_folder
# uniquefolder = 'La_sample/'
# SFNormFile = 'SF_Norm_files/'+uniquefolder+run_num
# SFNormFile = 'SF_Norm_files/'+runs_folder+run_num

# print(os.getcwd())

os.chdir('F:/LANL/')
datadir = 'D:/LANSCE_FP12_2023/data/' ## add directory of hard drive
uniquefolder = "runs" + str(run_start) + "-" + str(run_end) +"/"
SFNormFile = 'SF_Norm_files/'+uniquefolder+run_num

statefileloc = 'F:\LANL\SF_Norm_files\TR_R_expected_avgs_stds_afterclip.csv'
processedONOFFfolder = '/processed_data/'+uniquefolder+'ONOFF_D/'
processedasymfolder = '/processed_data/'+uniquefolder+'asym_D/'
processedasymfolder_bg = '/processed_data/'+uniquefolder+'asym_bg_D/'
ONOFFSavename = os.getcwd()+processedONOFFfolder+run_num+'_ONOFF_D'
AsymSavename = os.getcwd()+processedasymfolder+run_num+'_asym_D'
AsymSavename_bg = os.getcwd()+processedasymfolder_bg+run_num+'_asym_bg_D'
logger.add("F:/LANL/processed_data/" + uniquefolder + '0_ErrorLog_'+run_start+'_'+run_end+'_D.txt', delay = False)

# cannot handle all 24 detectors at once, memory issue... can look into np.empty and deleting variables if needed ##
# chan_enab = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]) ## all
chan_enab = np.array([0,1,2,3,4,5,6,7,8,9,10,11,24]) ## downstream
# chan_enab = np.array([12,13,14,15,16,17,18,19,20,21,22,23,24]) ## upstream

print('processing data: ' + uniquefolder + '/run' + run_num)

# print(os.getcwd()+processedpulsefolder)
if not os.path.exists(os.getcwd()+processedONOFFfolder) or not os.path.exists(os.getcwd()+processedasymfolder) or not os.path.exists(os.getcwd()+processedasymfolder_bg):
    # Create the directory
    os.makedirs(os.getcwd()+processedONOFFfolder)
    os.makedirs(os.getcwd()+processedasymfolder)
    os.makedirs(os.getcwd()+processedasymfolder_bg)
    print("Directory created successfully")
else:
    pass

# print(os.getcwd() + folder)

# get_ipython().run_line_magic('matplotlib', 'qt')

start = time.time()
fullstart = time.time()

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
    logger.error('run '+run_num + emessage)
    raise Exception(emessage)

end = time.time()

# print('file open time: ' + str(end-start))            
print('saving processed data to ' + AsymSavename)
print("Channel is " + str(chan_enab))


# In[2]:


# Store the big header for each channel in arrays

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


# In[3]:


# Determine the time axis for each channel

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

np.asarray(preTime)
np.asarray(startTime)
np.asarray(endTime)
np.asarray(resolution)

xs = np.asarray(xs) ## can convert xs to np array here because all detectors same numsamples


# In[4]:


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

start=time.time()
ys_arrHe, ETTT_arrHe, eventcount_arrHe  = dataread(read_data, [25], fileLength, numSamples) ##hardcoded channel 25 for He
ys_arr, ETTT_arr, eventcount_arr  = dataread(read_data, chan_enab, fileLength, numSamples) ##hardcoded channels for coils

end = time.time()
print('dataread from binary time: ' + str(end-start))


# In[5]:


timeDif=[]
for i in range(0,len(chan_enab)):
    timeDif.append([])
    for j in range(len(ETTT_arr[i])-1):
        timeDif[i].append((ETTT_arr[i][j+1]-ETTT_arr[i][j])*8)
#     print("Min time difference for channel", chan_enab[i], "is", min(timeDif[i]), "ns")
#     print("Max time difference for channel", chan_enab[i], "is", max(timeDif[i]), "ns \n")
#print(timeDif)


# In[6]:


# Load in SF and He normalization information ##
# SFNormFile2 = 'F:/LANL/SF_Norm_files/runs12034-12363/12036.h5' ## change sf norm file here or use default

try:
    df_SF = pd.read_hdf(SFNormFile + '.h5', key='df_0')
#     df_SF = pd.read_hdf(SFNormFile2, key='df_0')
    df_HE = pd.read_hdf(SFNormFile + '.h5', key='df_1')
#     df_HE = pd.read_hdf(SFNormFile2, key='df_1')
except Exception as e:
    logger.error('run '+run_num + ' failed during SFNormFile load')
    logger.exception(e)

SF_Sort_arr = df_SF[['nicknames', 'transition_locations']].to_numpy().T
He_Norm_arr = df_HE[['pulse', 'norms']].to_numpy().T

NormFactor = 1000000  ## He integrals are huge, this normalizes all of those by a constant value for ease of use
HeNorms= (He_Norm_arr[1])/NormFactor

# print(SF_Sort_arr)
# print(He_Norm_arr[1]/NormFactor)


# In[8]:


# basesub and plotting ##
start = time.time()

baseL = 0
baseR = int(((preTime[0]-groupStart[0])*0.70)/chanDec[0])  ##70% before the trigger
numRuns = int((fileLength[0]-20-numSamples[0])/(numSamples[0]+6)+1)
legend =  ['NaI', 'R']

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
            plt.legend()
            
# plotter(ys_arr[9:], xs[9:], baseR, numSamples) ##plot coils

@njit ## jit is faster for large # channels, slower for small # channels
def basesub(ys, baseRight, numpoints): 
    tempys_basesub = np.zeros((numRuns,numpoints[0]), dtype=np.float64)
    for pulse in range((len(eventcount_arr[0]))): ## all have 5000 pulses
        tempys_basesub[pulse]=np.subtract(ys[pulse], np.mean(ys[pulse][baseL:baseRight]))
    return tempys_basesub

@njit ## jit is faster for large # channels, slower for small # channels
def basesub_norm(ys, baseRight, numpoints): 
    tempys_basesub = np.zeros((numRuns,numpoints[0]), dtype=np.float64)
    for pulse in range((len(eventcount_arr[0]))): ## all have 5000 pulses
        tempys_basesub[pulse]=np.subtract(ys[pulse], np.mean(ys[pulse][baseL:baseRight]))
        tempys_basesub[pulse]=tempys_basesub[pulse]/HeNorms[pulse] 
    return tempys_basesub

ys_basesub = np.zeros((len(ys_arr), numRuns,numSamples[0]), dtype=np.float64)
# ys_basesub_norm = np.zeros((len(ys_arr), numRuns,numSamples[0]), dtype=np.float64)

for i in range(len(ys_basesub)): ## feeding y arrays into function 1 channel at  a time is faster than all at once
    ys_basesub[i] = basesub(ys_arr[i], baseR, numSamples)
# for i in range(len(ys_basesub)): ## feeding y arrays into function 1 channel at  a time is faster than all at once
#     ys_basesub[i] = basesub_norm(ys_arr[i], baseR, numSamples)

ys_basesub[-1] = ys_basesub[-1]*-1 ## invert 6Li to positive signal. Comment out if not using
# ys_basesub_norm[-1] = ys_basesub_norm[-1]*-1 ## invert 6Li to positive signal. Comment out if not using

end = time.time()
print('plotting and/or base subtraction time: ' + str(end-start))            


# In[9]:


# use 6Li t0 for all instead of for themselves individually ##

start = time.time()

NaIthresh=2000
Li6thresh=1000
threshold_array = (np.full(len(ys_basesub), NaIthresh))
threshold_array[-1] = Li6thresh

# njit ## numba does not support reversed, but this could be changed if it's slow
def find_offset(ys, thresharr):
    xCrosses = np.zeros((len(ys), numRuns)) #outer array is crossing arrays for given channel, inner array is crossing for each event
    offset = np.zeros((len(ys), numRuns), dtype=np.int32) ##offset in bins for each channel, each pulse
    modeCrosses = np.zeros((len(ys)), dtype=np.float64)
    for i in reversed(range(len(ys))):
        #xValues.append([])
        for p in range(len(ys[i])):
            xing = np.argmax(ys[i][p] > thresharr[i])
            #print(xing)
            xCrosses[i][p] = xing
        modeCrosses[i] = (st.mode(xCrosses[i])) #find the most typical crossing value for each channel
        for p in range(len(xCrosses[i])):
            offset[i][p] = (modeCrosses[-1] - xCrosses[i][p]) ## make sure this is the correct sign!!! 
    if (np.all(xCrosses[-1])) == False:
        emessage = ('ERROR: 6Li threshold was not reached for at least one pulse')
        logger.error('run '+run_num + emessage)
        raise Exception(emessage)
    return offset, xCrosses, modeCrosses
                           
offset, xCrosses, modeCrosses = find_offset(ys_basesub, threshold_array)

end = time.time()
print('finding offset time: ' + str(end-start))  


# In[10]:


# extend all arrays by a value, check that the max number of offset on 6Li is less than that value ##
start = time.time()

extendedRange = 3 ## must be a positive value which to extend ys_arr
if abs(max(offset[-1], key = abs)) > extendedRange: ## if the max offset of 6Li is >extendedRange, something is wrong
    emessage = ('ERROR: largest offset greater than extended range')
    logger.error('run '+run_num + emessage)
    raise Exception(emessage)

try:
    ys_ext = np.zeros((len(ys_basesub), len(ys_basesub[0]), len(ys_basesub[0][0])+extendedRange*2), dtype=np.float64)
    ys_cut = np.zeros((len(ys_basesub), len(ys_basesub[0]), (len(ys_ext[0][0])-((extendedRange*2)+1)*2)))
    xs_cut = np.zeros((len(ys_cut), len(ys_cut[0][0])))
except Exception as e:
    logger.error('run '+run_num + ' failed during ys_cut array creation')
    logger.exception(e)

# cant use jit because np.pad is not supported ##
def align_cut(ys, xs_arr, extendedr):
    tempys_ext = np.zeros((len(ys), len(ys[0])+extendedr*2), dtype=np.float64)
    tempys_cut = np.zeros((len(ys), (len(tempys_ext[0])-((extendedr*2)+1)*2)))
    tempxs_cut = np.zeros(len(tempys_cut[0]))
    for p in range(len(ys)):
        tempys_ext[p] = np.pad(ys[p], extendedr, 'constant', constant_values=(0))
        tempys_ext[p] = np.roll(tempys_ext[p],offset[-1][p]) ## assumes 6Li at -1 position
        tempys_cut[p] = tempys_ext[p][((extendedr*2)+1):-((extendedr*2)+1)].copy() ## cut by 7 (if extRange == 3)
        tempys_cut[p] = tempys_cut[p]/HeNorms[p] ## normalize by 3He integral  ## comment out if using basesub_norm
    x_cut_amt = int((len(ys[0]) - len(tempys_cut[0]))/2)
    tempxs_cut = xs_arr[x_cut_amt:-x_cut_amt].copy()
    return tempys_cut, tempxs_cut

# looping every channel through function is 5x faster ##
try:
    for i in range(len(ys_basesub)):
        ys_cut[i], xs_cut[i] = align_cut(ys_basesub[i], xs[i], extendedRange)
except Exception as e:
    logger.error('run '+run_num + ' failed aligning and cutting')
    logger.exception(e)
    
# checkp = 2053
# print(offset[-1][checkp]) ## checking offset for one example checkpulse
# print('original index for checkpulse: '+str(np.argmax(ys_basesub[0][checkp]> 2000))) ## we can follow the index as it changes with extension/cut
# #print('extended range index for checkpulse: '+str(np.argmax(ys_ext[0][checkp]> 2000)))
# print('cut array index for checkpulse: '+str(np.argmax((ys_cut[0][checkp]*HeNorms[checkp])> 2000)))

del ys_ext ## might help with memory issues
del ys_basesub

end = time.time()
print('aligning and cutting time: ' + str(end-start))            


# ## begin SF organization ##

# In[11]:


def organize_SF(SFsort_info): ## sometimes pulse 0 has the state switch. In that case, need to account by if clauses below
    counter = 0
    seq = 0
    seq_arr = ([[],[],[]])
    smallerseq = []
    smallerstateis = []
    for i in range(len(SFsort_info[1])-(np.mod((len(SFsort_info[1])), 8))):  ##111 mod 8 = 7, so essentially 111-7 = 104
        counter = counter+1
        if counter < 8:
            if (SF_Sort_arr[1][i]) == 0: ## catches state switches at pulse 0
                smallerstateis.append([(SFsort_info[1][i])+5,(SFsort_info[1][i+1])])
                smallerseq.append(SFsort_info[0][i+1])
                seq = seq+1
                continue
            smallerstateis.append([(SFsort_info[1][i])+5,(SFsort_info[1][i+1])])
            smallerseq.append(SFsort_info[0][i+1])
        elif counter == 8:
            if ((SF_Sort_arr[1][i])+45) >= 5000: ## breaks for state switches at pulse 0
                print(((SF_Sort_arr[1][i])+5))
                seq = seq+1
                seq_arr[0].append(seq)
                seq_arr[1].append(smallerseq)   
                seq_arr[2].append(smallerstateis)
                seq_arr[0] = [x-1 for x in seq_arr[0]] ## reset so sequences are 1-14 instead of 2-15
                break
            seq = seq+1 ## otherwise continue regular sorting
            smallerstateis.append([(SFsort_info[1][i])+5,(SFsort_info[1][i+1])])
            smallerseq.append(SFsort_info[0][i+1])
            seq_arr[0].append(seq)
            seq_arr[1].append(smallerseq)   
            seq_arr[2].append(smallerstateis)
            smallerseq = []
            smallerstateis = []
            counter  = 0
    return seq_arr

def find_leftover(SFsort_info, seq_arr): ## in case we want to use the other 6 states left over
    left = [[seq_arr[0][-1]+1],[],[]]
    counter = 0
    for i in range((len(SFsort_info[1])-(np.mod((len(SFsort_info[1])), 8))), len(SFsort_info[1])-1):
        counter = counter+1
        if counter < 8:
            left[1].append(SFsort_info[0][i+1])
            left[2].append([(SFsort_info[1][i])+5,(SFsort_info[1][i+1])])
    return left

try:
    sequence = organize_SF(SF_Sort_arr)
    if len(sequence[0]) == 14: ## catches state switches at pulse 0, leftovers are at the end of the regular sequence
        leftovers = [[sequence[0][-1]],[sequence[1][-1]],[sequence[2][-1]]]
        for i in range(len(sequence)):
            sequence[i].pop(-1) ## deletes the leftovers sequence for state switches at pulse 0
    else:
        leftovers = find_leftover(SF_Sort_arr, sequence) ## otherwise can use normal function
except Exception as e:
    logger.error('run '+run_num + ' failed during sequencing')
    logger.exception(e)

# print('sequences '+str(sequence[0]))
print(str(len(sequence[0]))+' sequences with sequence order: '+str(sequence[1][0]))
# print(leftovers)


# In[12]:


#  add up pulses for their respective state, in each 8 step sequence ##
#  turning into a by-channel function 06.13.24 ##

start = time.time()
sequence = np.asarray(sequence, dtype = object)

ON_sums = np.zeros((len(ys_cut), len(sequence[0]), len(ys_cut[0][0])), dtype=np.float64) ## 13 channels, 13 sequences, added pulses for ON
OFF_sums = np.zeros((len(ys_cut), len(sequence[0]), len(ys_cut[0][0])), dtype=np.float64) ## 13 channels, 13 sequences, added pulses for OFF

# @njit
def add_pulse(ys, SFarr):
    temp_ON = np.zeros((len(SFarr[0]), len(ys[0])), dtype=np.float64)
    temp_OFF = np.zeros((len(SFarr[0]), len(ys[0])), dtype=np.float64)
    for seq in range(0, len(SFarr[0])): ## for every sequence
    #         print('seq:' +str(SFarr[0][seq]))
#         print('seq:' +str(seq))
        for state in range(0, len(SFarr[1][0])): ## for every state in the sequence
    #         print("states loop " + str(range(0, len(SFarr[1][0]))[0]) + ' - ' +  str(range(0, len(SFarr[1][0]))[-1]))
            s = SFarr[1][seq][state] ## try this to condense code. Basically, the state currently at
            if s==0 or s==3 or s==5 or s==6: ## these are ON states
#                 print('ON "s" state '+str(s))
#                 print('"state" ' +str(state) + ' from ' + str(range((SFarr[2][seq][state][0]),(SFarr[2][seq][state][1]))))
#                 print('sums from '+str(range((SFarr[2][seq][state][0]),(SFarr[2][seq][state][1]))[0]) +
#                 ' - ' +str(range((SFarr[2][seq][state][0]),(SFarr[2][seq][state][1]))[-1]) + '\n')
                for p in range((SFarr[2][seq][state][0]),(SFarr[2][seq][state][1])): ##From 20-60 for example. SFarr[2] is the array of start to end pulses to sum
                    temp_ON[seq] = np.add(temp_ON[seq],ys[p]) ## start with zeros, add to each iteratively
            if s==1 or s==2 or s==4 or s==7: ## these are OFF states
#                 print('OFF "s" state '+str(s))
#                 print('"state" ' +str(state) + ' from ' + str(range((SFarr[2][seq][state][0]),(SFarr[2][seq][state][1]))))
#                 print('sums from '+str(range((SFarr[2][seq][state][0]),(SFarr[2][seq][state][1]))[0]) +
#                 ' - ' +str(range((SFarr[2][seq][state][0]),(SFarr[2][seq][state][1]))[-1]) + '\n')
                for p in range((SFarr[2][seq][state][0]),(SFarr[2][seq][state][1])):
                    temp_OFF[seq] = np.add(temp_OFF[seq],ys[p])
    return temp_ON, temp_OFF

for i in range(len(ys_cut)):
#     print('#################### channel: ' + str(i) + ' ##########################')
    ON_sums[i], OFF_sums[i] = add_pulse(ys_cut[i], sequence)
    
# for i in range(len(ys_basesub)-12):
#     print('#################### channel: ' + str(i) + ' ##########################')
#     ON_sums[i], OFF_sums[i], ON_minus_sums[i], ON_plus_sums[i] = add_pulse(ys_basesub[i], sequence)

end = time.time()
print('summing pulses into their states time: ' + str(end-start))  


# In[13]:


start = time.time()

BGregion1_beg = 4180  ## 10.28.24 hardcoded for now (maybe forever?). Seems like a good region for BG/Res
BGregion1_end = 5450
BGregion2_beg = 6250
BGregion2_end = 8991
BGreg1 = [BGregion1_beg,BGregion1_end]
BGreg2 = [BGregion2_beg,BGregion2_end]
fullrange = BGreg2[1]-BGreg1[0]
# print(BGreg1[0])

def BG_fitsubtract(bef_res_reg, aft_res_reg, ys): ## before/after resonance region [start:end] respectively, ys[ch] to fit
    binstot = aft_res_reg[1]-bef_res_reg[0]  ## total number of bins in whole region
    x1 = np.arange(bef_res_reg[0], bef_res_reg[1],1)
    x2 = np.arange(aft_res_reg[0], aft_res_reg[1],1)
    x = np.append(x1,x2) ## the x values that will be used to fit
    fullx = np.arange(bef_res_reg[0], aft_res_reg[1],1) ## an array of every x bin in entire region
    ys_bgsub = []
#     seq_bgsub = []
    for seq in range(0, len(ys)): ## number of sequences, usually 13
        fitdata1 = ys[seq][bef_res_reg[0]: bef_res_reg[1]]
        fitdata2 = ys[seq][aft_res_reg[0]: aft_res_reg[1]]
        datasplice = np.append(fitdata1, fitdata2)
#         x = np.linspace(0, len(datasplice),len(datasplice))
        y = datasplice
        data = odr.Data(x, y)
        poly_model2 = odr.polynomial(2)  # using 2nd order polynomial model (looks to be better than 3...)
        odr_obj = odr.ODR(data, poly_model2)
        output = odr_obj.run()  # running ODR fitting
        poly2 = np.poly1d(output.beta[::-1])
        poly_y2 = poly2(x)
        fullpoly_y2 = poly2(fullx)  ## extends the fit to fullx within the whole region
        bgsubtracted = ys[seq][bef_res_reg[0]:aft_res_reg[1]] - fullpoly_y2  ## subtracts RealData-BackgroundFit
#         seq_bgsub.append(bgsubtracted)
        ys_bgsub.append(bgsubtracted)
#     ys_bgsub.append(seq_bgsub)
    return ys_bgsub

ON_bgsub = np.zeros((len(ON_sums),len(ON_sums[0]),fullrange), dtype = np.float64) ## channels, sequences, range of BG subtraction
OFF_bgsub = np.zeros((len(ON_sums),len(ON_sums[0]),fullrange), dtype = np.float64) ## channels, sequences, range of BG subtraction
fullx = np.arange(BGreg1[0], BGreg2[1],1)

for i in range(0, len(ON_sums)-1):
    if chan_enab[i] == 24:
        emessage = ('BG fit does not work for 6Li yet')
        logger.error('run '+run_num + emessage)
        raise Exception(emessage)
#     print(chan_enab[i])
    ON_bgsub[i] = BG_fitsubtract(BGreg1,BGreg2, ON_sums[i])
    OFF_bgsub[i] = BG_fitsubtract(BGreg1,BGreg2, OFF_sums[i])
    
# for i in range(len(testfunc)):
#     plt.plot(xs_cut[1][BGreg1[0]:BGreg2[1]]*1e-6, testfunc[i], lw = '1.0', label='3rd Order polynomial background subtracted')

end = time.time()
print('BG fitting time: ' + str(end-start))  


# Saving all on/off added pulses, mainly for plotting

# In[14]:


all_ONOFF = np.zeros((len(ON_sums),2, len(ON_sums[0][0])), dtype=np.float64) ## all summed ON[0] and OFF[1] pulses for each channel, not each sequence

def all_onoff(ON_arr, OFF_arr):
    tempallON = np.zeros((len(ON_arr[0])), dtype=np.float64)
    tempallOFF = np.zeros((len(ON_arr[0])), dtype=np.float64)
    for seq in range(0, len(ON_arr)): ## number of sequences
        tempallON = np.add(ON_arr[seq],tempallON)
        tempallOFF = np.add(OFF_arr[seq],tempallOFF)
    return tempallON, tempallOFF

for i in range(len(ON_sums)):
    all_ONOFF[i][0], all_ONOFF[i][1] = all_onoff(ON_sums[i], OFF_sums[i]) ## channel, ON[0] or OFF[1], summed wave
    # print('ch done')


# In[16]:


asym_ch_bg = np.zeros((len(ON_bgsub), len(ON_bgsub[0][0])), dtype=np.float64) ## 1 Asym for each channel, not for each sequence (can change?)
asym_ch_raw = np.zeros((len(ON_sums), len(ON_sums[0][0])), dtype=np.float64) ## 1 Asym for each channel, not for each sequence (can change?)
# asym_pm = np.zeros((len(ONOFF_plus_sums),2, len(ONOFF_plus_sums[0][0][0])), dtype=np.float64) ## 1 Asym for each channel, not for each sequence (can change?)

def asym2(ON_arr, OFF_arr):
    tempasym = np.zeros((len(ON_arr[0])), dtype=np.float64)
    for seq in range(0, len(ON_arr)): ## number of sequences
        seqasym = ((ON_arr[seq]-OFF_arr[seq]) / (ON_arr[seq]+OFF_arr[seq]))
#         if ON_arr[seq]+OFF_arr[seq] == 0: ## add an exception for sum being == 0 that causes div by 0    
        tempasym = np.add(seqasym,tempasym)
    normedasym = tempasym/len(ON_arr)
    return normedasym

for i in range(len(ON_bgsub)):
    asym_ch_bg[i] = asym2(ON_bgsub[i], OFF_bgsub[i])
    asym_ch_raw[i] = asym2(ON_sums[i], OFF_sums[i])
#     asym_pm[i][0] = asym2(ONOFF_plus_sums[i][0], ONOFF_plus_sums[i][1])
#     asym_pm[i][1] = asym2(ONOFF_minus_sums[i][0], ONOFF_minus_sums[i][1])


# In[15]:


start = time.time()
# testch = 0

with h5py.File(ONOFFSavename+'.h5', 'w') as hdf5_file:
#     hdf5_file.create_dataset('ch '+str(np.char.zfill(str(chan_enab[0]), 2)), data=all_ONOFF[0], compression="gzip")
#     hdf5_file.create_dataset('ch '+str(np.char.zfill(str(chan_enab[0]), 2)), data=all_ONOFF[0])
    hdf5_file.create_dataset('xs ', data=xs_cut[0]) ## all xs are the same, even though they are per channel...
    hdf5_file.attrs['sequences'] = len(sequence[0])
    hdf5_file.attrs['rownames'] = ['ON_Row0', 'OFF_Row1']
    for i in range(0,len(ys_cut)):
#         with h5py.File('test_allONOFFs'+'.h5', 'a') as hdf5_file:
#         hdf5_file.create_dataset('ch '+str(np.char.zfill(str(chan_enab[i]), 2)), data=all_ONOFF[i], compression="gzip")
        hdf5_file.create_dataset('ch '+str(np.char.zfill(str(chan_enab[i]), 2)), data=all_ONOFF[i])
#     a = hdf5_file.create_dataset('ch '+str(np.char.zfill(str(chan_enab[0]), 2)), data=all_ONOFF[0])
#     a.attrs['rownames'] = ['ON_Row0', 'OFF_Row1']
#     a.attrs['sequences'] = len(sequence[0])


# In[17]:


start = time.time()

AsymSavename_raw = AsymSavename
with h5py.File(AsymSavename_raw+'.h5', 'w') as hdf5_file:
#     hdf5_file.create_dataset('ch '+str(np.char.zfill(str(chan_enab[0]), 2)), data=all_ONOFF[0], compression="gzip")
    hdf5_file.create_dataset('xs ', data=xs_cut[0])
    hdf5_file.attrs['sequences'] = len(sequence[0])
    for i in range(0,len(ys_cut)):
        hdf5_file.create_dataset('ch '+str(np.char.zfill(str(chan_enab[i]), 2)), data=asym_ch_raw[i])
        
with h5py.File(AsymSavename_bg+'.h5', 'w') as hdf5_file:
#     hdf5_file.create_dataset('ch '+str(np.char.zfill(str(chan_enab[0]), 2)), data=all_ONOFF[0], compression="gzip")
    hdf5_file.create_dataset('xs ', data=fullx) ## different x values for fitted function
    hdf5_file.attrs['sequences'] = len(sequence[0])
    for i in range(0,len(ys_cut)):
        hdf5_file.create_dataset('ch '+str(np.char.zfill(str(chan_enab[i]), 2)), data=asym_ch_bg[i])
        
# f = gzip.GzipFile("testcompressednparr.npy.gz", "w")
# np.save(file=f, arr=ys_cut)
# f.close()
        
end = time.time()
print('saving hdf5: ' + str(end-start))      


# In[ ]:


# ## testing reaad
# with h5py.File(ONOFFSavename+'.h5', 'r') as f:
#     print(f.attrs.keys())
#     print(f.attrs.get('sequences'))
#     print(f.attrs.get('rownames'))
#     f.close()
    
# # f = h5py.File('test_allONOFFs'+'.h5', 'r')
# # # print(f['ch 00'].attrs.get('sequences'))
# # f.close()


# In[ ]:


# ## testing reaad
# with h5py.File(AsymSavename+'.h5', 'r') as f:
#     print(f.attrs.keys())
#     print(f.attrs.get('sequences'))
#     print(f.attrs.get('rownames'))
#     f.close()


# In[ ]:


fullend = time.time()
print('full processing time: ' + str(fullend-fullstart))  
print('finished ' + str(datetime.now())) 
print('\n')


# ## end of data processing ##

# Now Voigt (unfinished)

# In[51]:


# from scipy.special import voigt_profile
# from scipy.optimize import curve_fit
# voigt_profile(2, 1., 1.)
# fig, ax = plt.subplots(figsize=(8, 8))
# x = np.linspace(-10, 10, 500)
# parameters_list = [(1.5, 0., "solid"), (1.3, 0.5, "dashed"),
#                    (0., 1.8, "dotted"), (1., 1., "dashdot")]
# for params in parameters_list:
#     sigma, gamma, linestyle = params
#     voigt = voigt_profile(x, sigma, gamma)
#     ax.plot(x, voigt, label=rf"$\sigma={sigma},\, \gamma={gamma}$",
#             ls=linestyle)
# plt.legend()
# plt.show()


# In[143]:


# from scipy.special import voigt_profile
# from scipy.optimize import curve_fit
# Resregion_beg = 5450
# Resregion_end = 6250
# Resreg = [Resregion_beg,Resregion_end]
# # voigt_profile(2, 1., 1.)
# # fig, ax = plt.subplots(figsize=(8, 8))
# # x = np.linspace(-10, 10, 500)
# # parameters_list = [(1.5, 0., "solid"), (1.3, 0.5, "dashed"),
# #                    (0., 1.8, "dotted"), (1., 1., "dashdot")]
# # for params in parameters_list:
# #     sigma, gamma, linestyle = params
# #     voigt = voigt_profile(x, sigma, gamma)
# #     ax.plot(x, voigt, label=rf"$\sigma={sigma},\, \gamma={gamma}$",
# #             ls=linestyle)
# sigma = 60
# gamma = 30
# beg = 5800
# end = 5900
# print(BGreg1[1]-BGreg1[0])
# print(BGreg2[0]-BGreg1[0])
# print(len(ON_bgsub[1][0]))
# # xdata = xs_cut[1][4180:8991]*1e-6
# # xdata = np.arange(Resreg[0],Resreg[1],1)
# xdata = np.arange(-(Resreg[1]-Resreg[0])/2,(Resreg[1]-Resreg[0])/2,1)
# # print(xdata[380:420])
# print(len(xdata))
# # ydata = ON_sums[1][0][5450:6250]
# # xdata = xs_cut[1][beg:end]
# # ydata = ON_sums[1][0][beg:end]
# ydata = ON_bgsub[1][0][BGreg1[1]-BGreg1[0]:BGreg2[0]-BGreg1[0]]
# print(len(ydata))

# voigt = voigt_profile(xdata, sigma, gamma)*30000
# # print(voigt)
# plt.plot(xdata, voigt)
# plt.plot(xdata,ydata)
# plt.show()
# popt, pcov = curve_fit(voigt_profile, xdata, ydata)
# # plt.plot(xs_cut[1][5450:6250], ON_sums[1][0][5450:6250], label="input data")
# # plt.plot(xdata, voigt)
# # popt, pcov = curve_fit(voigt, xdata, ydata)

# # popt
# # array([2.56274217, 1.37268521, 0.47427475])

# # plt.plot(xdata, func(xdata, *popt), 'r-',

# #          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

