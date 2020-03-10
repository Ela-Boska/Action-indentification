import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import stft

def loadcsv(fname:str, epcs:list, dB2Amp=True, begining=None, ending=None): # max time = 11870 ms
     with open(fname,encoding='utf-8') as file:
          lines = file.readlines()
     ans = {x:[] for x in epcs}
     real_ans = {x:[] for x in epcs}
     start_times = []
     end_times = []
     lines[0] = lines[0][1:]
     for line in lines:
          items = line.split(',')
          if int(items[0]) in ans:
               ans[int(items[0])].append([float(x) for x in items[3:6]])
     for key in ans:
          ans[key] = np.array(ans[key])
          start_times.append(np.min(ans[key][:,0]))
          end_times.append(np.max(ans[key][:,0]))
          # 解缠绕相位

          for i in range(len(ans[key])-1):
               if ans[key][i+1,2]-ans[key][i,2] > 3.14:
                    ans[key][i+1:,2] -= 2*np.pi
               elif (ans[key][i+1,2] - ans[key][i,2] < -3.14):
                    ans[key][i+1:,2] += 2*np.pi
     start = max(start_times)
     end = min(end_times)
     time = np.arange(start, end, 10000)
     if ending:
          ending = 1000*ending - (start-min(start_times))//1000
     if begining:
          begining = 1000*begining - (start-min(start_times))//1000
     for key in ans:
          real_ans[key] = np.zeros([time.shape[0],3])
          f = interp1d(ans[key][:,0],ans[key][:,1])
          real_ans[key][:,0] = (time-time[0])//1000
          real_ans[key][:,1] = f(time)
          real_ans[key][:,1] -= (real_ans[key][0,1])
          if dB2Amp:
               real_ans[key][:,1] = 10**(real_ans[key][:,1]/10)
          f = interp1d(ans[key][:,0],ans[key][:,2])
          real_ans[key][:,2] = f(time)
          real_ans[key] = real_ans[key][:,1:]
          if ending:
               real_ans[key] = real_ans[key][:int(ending)//10]
          if begining:
               real_ans[key] = real_ans[key][int(begining)//10:]
     return real_ans

def showcsv(fname:str, epcs:list, dB2Amp=True):
     plt.figure(figsize=[20,14])
     with open(fname,encoding='utf-8') as file:
          lines = file.readlines()
     ans = {x:[] for x in epcs}
     real_ans = {x:[] for x in epcs}
     start_times = []
     end_times = []
     lines[0] = lines[0][1:]
     for line in lines:
          items = line.split(',')
          if int(items[0]) in ans:
               ans[int(items[0])].append([float(x) for x in items[3:6]])
     for key in ans:
          ans[key] = np.array(ans[key])
          start_times.append(np.min(ans[key][:,0]))
          end_times.append(np.max(ans[key][:,0]))
          # 解缠绕相位

          for i in range(len(ans[key])-1):
               if ans[key][i+1,2]-ans[key][i,2] > 3.14:
                    ans[key][i+1:,2] -= 2*np.pi
               elif (ans[key][i+1,2] - ans[key][i,2] < -3.14):
                    ans[key][i+1:,2] += 2*np.pi
     start = min(start_times)
     for key in ans:
          ans[key][:,0] -= start
          ans[key][:,0] /= 1e6
     plt.subplot(121)
     for key in ans:
          plt.plot(ans[key][:,0],ans[key][:,1], label=key)
     plt.xlabel(r"time/s")
     plt.ylabel("RSSI/dB")
     plt.subplot(122)
     for key in ans:
          plt.plot(ans[key][:,0],ans[key][:,2], label=key)
     plt.ylabel("Phase/rad")
     plt.xlabel(r"time/s")
     plt.legend()
     plt.show()

def show_rssi(data, key:int, nfft, nperseg)->None:
     plt.subplot(221)
     plt.plot(np.arange(0,12.0,0.01),data[key][:,0],label=str(key))
     plt.legend()
     plt.xlabel('time/ms')
     plt.ylabel('RSSI/dB')
     plt.subplot(222)
     plt.plot(np.arange(0,12.0,0.01),data[key][:,1],label=str(key))
     plt.legend()
     plt.xlabel('time/ms')
     plt.ylabel('phase')
     plt.subplot(223)
     for key in data:
          data[key][:,1] = 10**(data[key][:,1]/10)
     f,t,dfs = to_DFS(data, nfft, nperseg=nperseg)
     t = t
     plt.pcolormesh(t, f, dfs[key].T, cmap='gray')
     plt.xlabel('time/s')
     plt.ylabel('DFS')
     plt.show()

def to_DFS(data:dict, fs, nperseg, cut = None):
     ans = {}
     for key in data.keys():
          ans[key] = data[key][:,0]*np.exp(1j*data[key][:,1])
          f, t, zxx = stft(ans[key],fs=fs,nperseg=nperseg)
          zxx = np.abs(zxx)
          ans[key] = np.zeros_like(zxx)
          ans[key][:nperseg//2] = zxx[nperseg//2:]
          ans[key][nperseg//2:] = zxx[:nperseg//2]
          ans[key] = ans[key].T
          if cut is not None:
               ans[key] = ans[key][:,int(cut[0]//(fs/nperseg)+nperseg//2):int(cut[1]//(fs/nperseg)+nperseg//2)]
     F = np.zeros_like(f)
     F[:nperseg//2] = f[nperseg//2:]
     F[nperseg//2:] = f[:nperseg//2]
     if cut is not None:
          F = F[int(cut[0]//(fs/nperseg)+nperseg//2):int(cut[1]//(fs/nperseg)+nperseg//2)]
     return F,t,ans

def aggregate(data:dict, order:list):
     # input: {epc1:array(T,K), ... , ecpN:array(T,K)}
     # output:array(T,K,N)
     return np.stack([data[key] for key in order], -1)