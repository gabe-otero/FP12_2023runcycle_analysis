
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

# run_num = '11143'
# os.chdir('F:/LANL/')
# datadir = 'sample_data/'
# uniquefolder = 'La_sample/'
# SFNormFile = 'SF_Norm_files/'+uniquefolder+run_num

# # print(os.getcwd())

os.chdir('F:/LANL/')
datadir = 'D:/LANSCE_FP12_2023/data/' ## add directory of hard drive
uniquefolder = "runs" + str(run_start) + "-" + str(run_end) +"/"
SFNormFile = 'SF_Norm_files/'+uniquefolder+run_num

statefileloc = 'F:\LANL\SF_Norm_files\TR_R_expected_avgs_stds_afterclip.csv'
processedpulsefolder = '/processed_data/'+uniquefolder+'pulses_added_U/'
processedasymfolder = '/processed_data/'+uniquefolder+'asym_U/'
AddedPulseSavename = processedpulsefolder+run_num+'_pulsesadded_U'
AsymSavename = processedasymfolder+run_num+'_asym_U'
logger.add("F:/LANL/processed_data/" + uniquefolder + '0_ErrorLog_'+run_start+'_'+run_end+'_U.txt', delay = False)

print('processing data: ' + uniquefolder + '/run' + run_num)

# print(os.getcwd()+processedpulsefolder)
if not os.path.exists(os.getcwd()+processedpulsefolder) or not os.path.exists(os.getcwd()+processedasymfolder):
    # Create the directory
    os.makedirs(os.getcwd()+processedpulsefolder)
    os.makedirs(os.getcwd()+processedasymfolder)
    print("Directory created successfully")
else:
    pass

# print(os.getcwd() + folder)

# get_ipython().run_line_magic('matplotlib', 'qt')

start = time.time()
fullstart = time.time()

## cannot handle all 24 detectors at once, memory issue... can look into np.empty and deleting variables if needed
#chan_enab = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]) ## all
#chan_enab = np.array([0,1,2,3,4,5,6,7,8,9,10,11,24]) ## downstream
chan_enab = np.array([12,13,14,15,16,17,18,19,20,21,22,23,24]) ## upstream

#@jit(nopython = True)
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

end = time.time()
# print('file open time: ' + str(end-start))            

print('saving processed data to ' + AsymSavename)
print("Channel is " + str(chan_enab))
end = time.time()
# print(end-start)
# print(read_data)

# Store the big header for each channel in arrays


# In[2]:


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

# In[3]:

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

#np.asarray(preTime)
#np.asarray(startTime)
#np.asarray(endTime)
#np.asarray(resolution)
xs = np.asarray(xs) ## can convert xs to np array here because all detectors same numsamples

# In[16]:

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

#start=time.time()
#ys_arrHe, ETTT_arrHe, eventcount_arrHe  = dataread(read_data, [25], fileLength, numSamples) ##hardcoded channel 25 for He
ys_arr, ETTT_arr, eventcount_arr  = dataread(read_data, chan_enab, fileLength, numSamples) ##hardcoded channels for coils

end = time.time()
print('dataread from binary time: ' + str(end-start))

# In[4]:

timeDif=[]
for i in range(0,len(chan_enab)):
    timeDif.append([])
    for j in range(len(ETTT_arr[i])-1):
        timeDif[i].append((ETTT_arr[i][j+1]-ETTT_arr[i][j])*8)
#     print("Min time difference for channel", chan_enab[i], "is", min(timeDif[i]), "ns")
#     print("Max time difference for channel", chan_enab[i], "is", max(timeDif[i]), "ns \n")
#print(timeDif)

# In[18]:

## basesub and plotting ##
baseL = 0
baseR = int(((preTime[0]-groupStart[0])*0.70)/chanDec[0])
numRuns = int((fileLength[0]-20-numSamples[0])/(numSamples[0]+6)+1)
legend =  ['NaI', 'R']

start = time.time()

s = 20 ## pulse to look at
t=s+1

## dont know why this is so slow ##
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
            
#plotter(ys_arr[9:], xs[9:], baseR, numSamples) ##plot coils

ys_basesub = np.zeros((len(ys_arr), numRuns,numSamples[0]), dtype=np.float64)

@njit ## jit is faster for large # channels, slower for small # channels
def basesub(ys, baseRight, numpoints): 
    tempys_basesub = np.zeros((numRuns,numpoints[0]), dtype=np.float64)
    for pulse in range((len(eventcount_arr[0]))): ## all have 5000 pulses
        tempys_basesub[pulse]=np.subtract(ys[pulse], np.mean(ys[pulse][baseL:baseRight]))
    return tempys_basesub

## got rid of sums here, should be done after aligning and cutting
## got rid of xs in basesub, don't think we need them as an input 06.10.24

for i in range(len(ys_basesub)): ## feeding y arrays into function 1 channel at  a time is faster than all at once
    ys_basesub[i] = basesub(ys_arr[i], baseR, numSamples)

ys_basesub[-1] = ys_basesub[-1]*-1 ## invert 6Li to positive signal. Comment out if not using

end = time.time()
# print('plotting and/or base subtraction time: ' + str(end-start))            

# In[6]:

## Load in SF and He normalization information 
try:
    df_SF = pd.read_hdf(SFNormFile + '.h5', key='df_0')
    df_HE = pd.read_hdf(SFNormFile + '.h5', key='df_1')
except Exception as e:
    logger.error(run_num + ' failed during SFNormFile load')
    logger.exception(e)

SF_Sort_arr = df_SF[['nicknames', 'transition_locations']].to_numpy().T
He_Norm_arr = df_HE[['pulse', 'norms']].to_numpy().T

NormFactor = 100000  ## He integrals are huge, this normalizes all of those by a constant value for ease of use
HeNorms= (He_Norm_arr[1])/NormFactor

# print(df_SF)


# In[7]:

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
    logger.error(run_num + ' failed during sequencing')
    logger.exception(e)

# print('sequences '+str(sequence[0]))
print(str(len(sequence[0]))+' sequences with sequence order: '+str(sequence[1][0]))
# print(leftovers)


# In[10]:

## use 6Li t0 for all instead of for themselves individually

start = time.time()

NaIthresh=2000
Li6thresh=1000

threshold_array = (np.full(len(ys_basesub), NaIthresh))
threshold_array[-1] = Li6thresh

#@njit ## numba does not support reversed, but this could be changed if it's slow
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
        logger.error(run_num + emessage)
        raise Exception(emessage)
    return offset, xCrosses, modeCrosses
                           
offset, xCrosses, modeCrosses = find_offset(ys_basesub, threshold_array)

end = time.time()
# print('finding offset time: ' + str(end-start))  


# In[11]:

## this cell loops through every channel as opposed to inputting all channels at once. 5x faster

start = time.time()

## extend all arrays by a value, check that the max number of offset on 6Li is less than that value ##
extendedRange = 3 ## must be a positive value which to extend ys_arr
if abs(max(offset[-1], key = abs)) > extendedRange: ## if the max offset of 6Li is >extendedRange, something is wrong
    emessage = ('ERROR: largest offset greater than extended range')
    logger.error(run_num + emessage)
    raise Exception(emessage)

# ys_ext = np.zeros((len(ys_basesub), len(ys_basesub[0]), len(ys_basesub[0][0])+extendedRange*2), dtype=np.float64)
# ys_cut = np.zeros((len(ys_basesub), len(ys_basesub[0]), (len(ys_ext[0][0])-((extendedRange*2)+1)*2)))
try:
    ys_ext = np.empty((len(ys_basesub), len(ys_basesub[0]), len(ys_basesub[0][0])+extendedRange*2), dtype=np.float64)
    ys_cut = np.empty((len(ys_basesub), len(ys_basesub[0]), (len(ys_ext[0][0])-((extendedRange*2)+1)*2)))
    xs_cut = np.zeros((len(ys_cut), len(ys_cut[0][0])))
except Exception as e:
    logger.error(run_num + ' failed during ys_cut array creation')
    logger.exception(e)

# cant use jit because np.pad is not supported
def align_cut_norm(ys, xs_arr, extendedr):
    tempys_ext = np.zeros((len(ys), len(ys[0])+extendedr*2), dtype=np.float64)
    tempys_cut = np.zeros((len(ys), (len(tempys_ext[0])-((extendedr*2)+1)*2)))
    tempxs_cut = np.zeros(len(tempys_cut[0]))
    for p in range(len(ys)):
        tempys_ext[p] = np.pad(ys[p], extendedr, 'constant', constant_values=(0))
        tempys_ext[p] = np.roll(tempys_ext[p],offset[-1][p]) ## assumes 6Li at -1 position
        tempys_cut[p] = tempys_ext[p][((extendedr*2)+1):-((extendedr*2)+1)].copy() ## cut by 7 (if extRange == 3)
        tempys_cut[p] = tempys_cut[p]/HeNorms[p] ## normalize by 3He integral
    x_cut_amt = int((len(ys[0]) - len(tempys_cut[0]))/2)
    tempxs_cut = xs_arr[x_cut_amt:-x_cut_amt].copy()
    return tempys_cut, tempxs_cut

try:
    for i in range(len(ys_basesub)):
        ys_cut[i], xs_cut[i] = align_cut_norm(ys_basesub[i], xs[i], extendedRange)
except Exception as e:
    logger.error(run_num + ' failed aligning and cutting')
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


# In[46]:

## add up pulses for their respective state, in each 8 step sequence
## turning into a by-channel function 06.13.24

start = time.time()

legend = ['NaI5 (downstream)','NaI4R (upstream)','6Li']

# added_pulses = np.zeros((len(ys_cut), len(sequence[0]), 8, len(ys_cut[0][0])), dtype=np.float64) ## 13 sequences, 8 stages each works?
## i channels, 13 sequences each, 8 states each sequence, 8992 num points

ON_OFF_sums = np.zeros((len(ys_cut), len(sequence[0]), 2, len(ys_cut[0][0])), dtype=np.float64) ## 13 sequences, 2 for ON or OFF for each sequence

# @njit
def add_pulse(ys, SFarr):
#     tempadded_p = np.zeros((len(SFarr[0]), 8, len(ys[0])), dtype=np.float64)    
    temp_ONOFF = np.zeros((len(SFarr[0]), 2, len(ys[0])), dtype=np.float64)
    for seq in range(0, len(SFarr[0])): ## for every sequence
#         print('seq:' +str(SFarr[0][seq]))
        for state in range(0, len(SFarr[1][0])): ## for every state in the sequence
#             print(state)
#             print('seq:' +str(SFarr[0][seq]) +', state: ' + str(SFarr[1][seq][state]))
            s = SFarr[1][seq][state] ## try this to condense code. Basically, the state currently at
#             for p in range((SFarr[2][seq][state][0]),(SFarr[2][seq][state][1])+1): ##From 20-60 for example. SFarr[2] is the array of start to end pulses to sum
#                 tempadded_pulses[i] = added_pulses[i]+np.array(ys_cut[i][j])
#                 tempadded_p[seq][state] = np.add(tempadded_p[seq][state],(ys[p])) ## start with zeros, add to each iteratively
            if s==2 or s==4 or s==5 or s==7: ## these are ON states
                for p in range((SFarr[2][seq][state][0]),(SFarr[2][seq][state][1])+1): ##From 20-60 for example. SFarr[2] is the array of start to end pulses to sum
#                 tempadded_pulses[i] = added_pulses[i]+np.array(ys_cut[i][j])
#                 tempadded_p[seq][state] = np.add(tempadded_p[seq][state],(ys[p])) ## start with zeros, add to each iteratively
#                 print('ON sum: ' + str(temp_ONOFF[seq][0]))
                    temp_ONOFF[seq][0] = np.add(temp_ONOFF[seq][0],ys[p]) ## start with zeros, add to each iteratively
#                 print('ON state: ' + str(s))
            if s==0 or s==1 or s==3 or s==6: ## these are OFF states
#                 print('OFF state: ' + str(s))
                for p in range((SFarr[2][seq][state][0]),(SFarr[2][seq][state][1])+1): ##From 20-60 for example. SFarr[2] is the array of start to end pulses to sum
#                 tempadded_pulses[i] = added_pulses[i]+np.array(ys_cut[i][j])
#                 tempadded_p[seq][state] = np.add(tempadded_p[seq][state],(ys[p])) ## start with zeros, add to each iteratively
#                 print('ON sum: ' + str(temp_ONOFF[seq][0]))
                    temp_ONOFF[seq][1] = np.add(temp_ONOFF[seq][1],ys[p]) ## start with zeros, add to each iteratively
    return temp_ONOFF

for i in range(len(ys_cut)):
    ON_OFF_sums[i] = add_pulse(ys_cut[i], sequence)
                
## plotting examples
# plt.plot(xs_cut[i], added_pulses[0][0][0] , label=legend[0] +', sequence 1 state 1, 40 pulses added')
# plt.plot(xs_cut[i], added_pulses[0][0][1] , label=legend[0] +', sequence 1 state 2, 40 pulses added')
# plt.plot(xs_cut[i], added_pulses[0][1][0] , label=legend[0] +', sequence 2 state 1, 40 pulses added') 
# plt.plot(xs_cut[i], added_pulses[0][1][1] , label=legend[0] +', sequence 2 state 2, 40 pulses added') 
    
# plt.title('Detector signals') 
# plt.xlabel("time from trigger (ns)")
# plt.ylabel("ADC")

# # plt.axvline(xs[0][baseL], ls = '--')
# # plt.axvline(xs[0][baseR], ls = '--')
# #plt.axvline(xs[1][intgrL], ls = '--', c ='g')
# #plt.axvline(xs[1][intgrR], ls = '--', c ='g')
# #plt.axvline(xs[2][HeintgrL], ls = '--', c ='r')
# #plt.axvline(xs[2][HeintgrR], ls = '--', c ='r')

# plt.legend()
# plt.show()

# end = time.time()
# print('summing pulses into their states time: ' + str(end-start))  


# In[48]:

start = time.time()

Asym = np.zeros((len(ON_OFF_sums), len(ON_OFF_sums[0][0][0])), dtype=np.float64) ## 1 Asym for each channel, not for each sequence (can change)

def asym(ON_OFF_arr):
    tempasym = np.zeros((len(ON_OFF_arr[0][0])), dtype=np.float64)
    for seq in range(len(ON_OFF_arr[0])): ## number of sequences
        asymform = ((ON_OFF_arr[seq][0]-ON_OFF_arr[seq][1]) / (ON_OFF_arr[seq][0]+ON_OFF_arr[seq][1]))
        tempasym = np.add(asymform,tempasym)
    normedasym = tempasym/len(ON_OFF_sums[0])
    return normedasym

for i in range(len(ON_OFF_sums)):
    Asym[i] = asym(ON_OFF_sums[i])

end = time.time()
# print('calculate asymmetry time: ' + str(end-start)) 


# In[13]:

# np.save(os.getcwd() + AddedPulseSavename, added_pulses)
np.save(os.getcwd() + AsymSavename, Asym)

fullend = time.time()
print('full processing time: ' + str(fullend-fullstart))  
print('finished ' + str(datetime.now())) 
print('\n')

# ## end of data processing ##


# In[57]:
