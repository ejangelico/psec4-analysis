import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys
import pandas as pd 
import scipy.io as spio
import Waveform
import cPickle as pickle 
import time
import math

class DataSet:
	#Creates an class filled with all the events out of filename
	#data is only filled into event_array on a function-by-function
	#basis. If a function only needs 1 event, it will only load 1 event
	def __init__(self, filename, acdc_clock=40, calibration_file=None):

		self.data = pd.read_csv(filename,delimiter='\s+', index_col=[0, 1, 2])
		#peddata not used yet
		#self.peddata = pd.read_csv(self.filename[:-4]+"ped",delimiter='\s+', index_col=[0,1])
		self.metadata = pd.read_csv(filename[:-4]+"meta",delimiter='\s+', index_col=[0,1])

		self.nsamples = None
		self.nsamples = self.get_nsamples()

		self.timestep = 1.0/(self.nsamples*acdc_clock*1e6) #nanoseconds
		
		#calfile holds info about individual LAPPD
		#stripline velocities and others...
		#*not yet implemented
		self.cal_file = calibration_file 




	def get_waveform(self, event, board, channel):
		ADCcounts_to_mv = 1.2*1000.0/4096.0
		if((event, board, channel) in self.data.index):
			mv_values = self.data.loc[(event, board, channel)].tolist()
			mv_values = [_*ADCcounts_to_mv for _ in mv_values]
			times = [i*self.timestep for i in range(self.get_nsamples())]
			return Waveform.Waveform(mv_values, times)

		else:
			#print "Could not find that channel/event/board in dataset"
			return None


	#get the max number of events in dataset. 
	#this does not ensure that every board 
	#is recorded in each event.
	def get_max_events(self):
		return max(self.data.index.levels[0])

	def get_chs(self):
		return list(self.data.index.levels[2])

	def get_boards(self):
		return list(self.data.index.levels[1])

	def get_nsamples(self):
		if(self.nsamples is None):
			return len(self.data.iloc[0])
		else:
			return self.nsamples

	def get_timestep(self):
		return self.timestep

	#Writes the dataset object to a pickle file
	def write_pickle(self, filename):
		pickle.dump(self, open(filename, 'wb'))
				
	#return a list of all voltages from all samples
	def get_all_sample_values(self):
		all_samples = []
		for i, row in self.data.iterrows():
			for sm in row:
				all_samples.append(sm)

		return all_samples


	def plot_waveforms_separated(self, event, boards, channels):

		if(len(channels) < 5):
			ncols = 2
			nrows = 2
		else:
			ncols = 6 #for six channels per board
			nrows = int(math.ceil(float(len(channels))/ncols))

		fig, ax = plt.subplots(figsize=(12,7),ncols=ncols, nrows=nrows)
		ax_1d = []
		for _ in ax:
			for i in range(len(_)):
				ax_1d.append(_[i])

		colors = ['b', 'g', 'r']
		for j, bo in enumerate(boards):
			for i, ch in enumerate(channels):
				thewav = self.get_waveform(event, bo, ch)
				if(thewav is None):
					continue
				thewav.plot(ax_1d[i])
				ax_1d[i].get_lines()[-1].set_color(colors[bo])
				ax_1d[i].set_title("CH = " + str(ch))
				ax_1d[i].set_xlabel('ns')
				ax_1d[i].set_ylabel('mv')

		return ax

	def plot_event_heatmap(self, event, board, ax=None):
		if(ax is None):
			fig, ax = plt.subplots()
		#requires that all channels have the
		#same time values 
		x = []
		y = []
		z=[]
		times = np.arange(0, self.get_nsamples()*self.get_timestep()*1e9, self.get_timestep()*1e9)
		channels = self.get_chs()
		for j, ch in enumerate(channels):
			wfm = self.get_waveform(event, board, ch)
			sig = wfm.get_signal()
			if(sig is None):
				continue
			for i, t in enumerate(times):
				voltage = sig[i]
				x.append(t)
				y.append(ch)
				z.append(voltage)
			

		xbins = times
		ybins = channels
		h = ax.hist2d(x, y, bins=[xbins, ybins], weights=z, cmap=plt.inferno(), cmax=20, cmin=-50)
		plt.colorbar(h[3],ax=ax)
		plt.show()



	#Plots numEvents random events from the dataset
	#using 
	def plot_random_events(self, nevts, boards=None, channels=None):
		maxev = self.get_max_events()
		if(channels is None):
			channels = self.get_chs()
		if(boards is None):
			boards = self.get_boards()

		
		for i in range(nevts):
			fig, ax = plt.subplots(nrows = len(boards))
			ev = np.random.randint(maxev)
			#self.plot_waveforms_separated(ev, boards, channels)
			if(len(boards) == 1):
				self.plot_event_heatmap(ev, boards[0], ax)
			else:
				for j, a in ax:
					self.plot_event_heatmap(ev, boards[j], a)

			plt.show()

	def plot_all_events(self, boards=None, channels=None):
		maxev = self.get_max_events()
		if(channels is None):
			channels = self.get_chs()
		if(boards is None):
			boards = self.get_boards()

		
		for ev in range(maxev):
			fig, ax = plt.subplots(nrows = len(boards))
			for b in boards:
				print self.which_channels_triggered(ev, b)

			self.plot_waveforms_separated(ev, boards, channels)
			plt.show()
			continue
			if(len(boards) == 1):
				self.plot_event_heatmap(ev, boards[0], ax)
			else:
				for j, a in ax:
					self.plot_event_heatmap(ev, boards[j], ax)
				
			plt.show()




	#sums all of the pulse power spectral densities
	#for all events, keeps channels separate
	def get_summed_psds(self):
		summed_chs = [[[] for _ in self.get_chs()] for _ in range(max(self.get_boards()))]
		freqs = [] 	

		for ev in range(self.get_max_events()):
			for bo in self.get_boards():
				for ch in self.get_chs():
					wf = self.get_waveform(ev, bo, ch)
					if(wf is None):
						continue

					psd, fs = wf.get_power_spectral_density()
					if(len(summed_chs[bo][ch]) == 0):
						summed_chs[bo][ch] = psd
						freqs = fs
						continue
					else:
						for j in range(len(psd)):
							summed_chs[bo][ch] += psd[j]

		#keeping it un-normalized, arbitrary units

		return (freqs, summed_chs)

	def plot_summed_psd_separated_channels(self, board):
		freqs, summed = self.get_summed_psds()
		fig, ax = plt.subplots(nrows=5, ncols=6)
		#flatten ax array to 1D
		ax_1d = []
		for _ in ax:
			for i in range(len(_)):
				ax_1d.append(_[i])

		for ch in self.get_chs():
			ax_1d[ch].plot(freqs, summed[board][ch])
			ax_1d[ch].set_title("CH = " + str(ch + 1))
			ax_1d[ch].locator_params('x', nbins=10)

		fig.suptitle("Board " + str(board) + " PSD, x-axis = freq. (GHz), y-axis = PSD (arb units)")
		plt.show()

	def plot_summed_psd_overlayed(self, channels, board):
		fig, ax = plt.subplots()
		freqs, summed = self.get_summed_psds()
		for ch in self.get_chs():
			ax.plot(freqs, summed[board][ch])
			ax.set_title("Power spectral density of CH = " + str(ch + 1))
			ax.set_xlabel("Freq (GHz)")
			ax.set_ylabel("PSD arb. units")

		plt.show()

	#looks at all voltages in all events
	#in the dataset and finds the
	#minimum difference between two voltage samples
	#to infer the resolution of the voltage measurement
	#on the waveform
	def get_minimum_voltage_binning(self):
		samples = self.get_all_sample_values()
		a, size = sorted(samples), len(samples)
		res = [a[i + 1] - a[i] for i in xrange(size) if i+1 < size]
		res = [_ for _ in res if _ > 1e-6]
		return min(res) #mV




#---place holder on updating code

	#takes the dataset and takes every sample 
	#from every event and every channel 
	#and plots a histogram of the mV values. 
	def plot_noise_dataset(self, outfilename):

		#vbinwidth = self.get_minimum_voltage_binning()
		vbinwidth = 1#mv
		fig, ax = plt.subplots(nrows=5, ncols=6, figsize=(60, 30))
		#flatten ax array to 1D
		ax_1d = []
		for _ in ax:
			for i in range(len(_)):
				ax_1d.append(_[i])
		for ch in range(self.nch):
			chsamples = []
			for ev in self.event_array:
				tempsamps = ev.get_all_samples([ch])
				chsamples += tempsamps

			bin_edges = np.arange(min(chsamples), max(chsamples), vbinwidth)
			n, bins, patches = ax_1d[ch].hist(chsamples, bin_edges, lw=3, fc=None)
			ax_1d[ch].set_title("CHANNEL = " + str(ch + 1))

			#find full width half max of dataset
			maxidx = list(n).index(max(n))
			maxn = max(n)
			bin_low = None
			bin_high = None
			#iterate forward from maxidx
			for i, b_e in enumerate(bins[maxidx:]):
				if(n[maxidx + i] <= 0.5*maxn):
					bin_high = b_e
					break
			#iterate backwards from maxidx
			for j, b_e in enumerate(bins[maxidx::-1]):
				if(n[maxidx - j] <= 0.5*maxn):
					bin_low = b_e
					break
			if(bin_low is None or bin_high is None):
				print "failed finding FWHM"
				continue

			ax_1d[ch].axvspan(bin_low, bin_high, facecolor='y', alpha=0.3, label="FWHM: " + str(bin_high - bin_low) + "\nLower bound (mV): " + str(bin_low) + "\nLower bound (adc counts): " + str(bin_low*4096.0/1200.0))
			ax_1d[ch].legend()
			ax_1d[ch].set_xlabel("mV")
			

		plt.savefig(outfilename, bbox_inches='tight')


	#takes the dataset and takes every sample 
	#from every event and a select channel
	#and plots a histogram of the mV values. 
	def plotNoiseDatasetChs(self, outfilename, channel):

		vbinwidth = self.getVoltageBinning()
		fig, ax = plt.subplots(figsize=(15, 11))
		#flatten ax array to 1D
		ch = channel

		chsamples = []
		for ev in self.eventArray:
			t, chwave = ev.getPulseWaveform(ch)
			#t, chwave = ev.getPulseWaveformDCRemoved(ch)
			for sample in chwave:
				chsamples.append(sample)

		bin_edges = np.arange(min(chsamples), max(chsamples), vbinwidth)
		n, bins, patches = ax.hist(chsamples, bin_edges, lw=3, fc=None)
		ax.set_title("CHANNEL = " + str(ch + 1))

		#find full width half max of dataset
		maxidx = list(n).index(max(n))
		maxn = max(n)
		bin_low = None
		bin_high = None
		#iterate forward from maxidx
		for i, b_e in enumerate(bins[maxidx:]):
			if(n[maxidx + i] <= 0.5*maxn):
				bin_high = b_e
				break
		#iterate backwards from maxidx
		for j, b_e in enumerate(bins[maxidx::-1]):
			if(n[maxidx - j] <= 0.5*maxn):
				bin_low = b_e
				break
		if(bin_low is None or bin_high is None):
			print "failed finding FWHM"

		ax.axvspan(bin_low, bin_high, facecolor='y', alpha=0.3, label="FWHM: " + str(bin_high - bin_low) + "\nLower bound (mV): " + str(bin_low) + "\nLower bound (adc counts): " + str(bin_low*4096.0/1200.0))
		ax.legend()
		ax.set_xlabel("mV")
		

		plt.savefig(outfilename, bbox_inches='tight')


	

	

	def plot_overlay_nevents(self, nevts=None, channels=None):
		if(nevts is None and self.num_events==0):
			self.load_all_events()
		elif(nevts is None):
			pass
		else:
			self.load_random_events(nevts)

		if(channels is None):
			channels = range(self.nch)

		fig, ax = plt.subplots()
		for ev in self.event_array:
			ev.plot_waveforms_overlayed(channels, ax)

		plt.show()





	#-----------transit time spread for tektronix data--------#


	#looks for a pulse in each channel. 
	#find it's 10% of max. find it's time
	#relative to a time where the laser trigger
	#passes a constant threshold. return a list of times
	#for all event in the loaded buffer
	def get_tts_rel_trigger(self, trigchan, sig_chans, trig_thresh, sig_thresh, pulse_loc=None):
		if(self.num_events == 0):
			self.load_all_events()

		skipped_events = 0
		reltts = [[] for _ in sig_chans]
		for ev in self.event_array:
			#constant threshold discrimination on square laser trigger
			trig_time = ev.get_laser_trigger_time(trigchan, trig_thresh)
			if(trig_time is None):
				#skip the event
				skipped_events += 1
				continue
			
			arrival_times = ev.get_pulse_arrival_times(sig_chans, sig_thresh, pulse_loc)
			for i, at in enumerate(arrival_times):
				#didn't find a pulse in this channel
				#or thresholding didnt trigger
				if(at is None):
					continue
				else:
					reltts[i].append(at - trig_time)

		return reltts




#------------ metadata functions -----------------#


	def print_metadata(self, event, board, key=None):
		cols = list(self.metadata.columns.values)
		if((event, board) in self.metadata.index):
			if(key is None):
				for c in cols:
					print c + " \t\t\t " + str(self.metadata[c][(event, board)])
			else:
				print key + " \t\t\t" + str(self.metadata[key][(event, board)])
		else:
			print "Could not find metadata for event " + str(event) + " and board " + str(board)


	def get_all_metadata(self, data_key):
		boards = []
		events = []
		values = []
		for i,row in self.metadata.iterrows():
			boards.append(i[1])
			events.append(i[0])
			values.append(int(row[data_key]))

		return events, boards, values

	#returns the trigger time 
	#in units of "relative clock counts"
	def get_event_time(self, event, board):
		if((event, board) in self.metadata.index):
			hi = self.metadata["dig_timestamp_hi"][(event, board)]
			mid = self.metadata["dig_timestamp_mid"][(event, board)]
			lo = self.metadata["dig_timestamp_lo"][(event, board)]
			return hi*mid*lo 
		else:
			return None

	#returns a list of which channels triggered 
	#from self trigger
	def which_channels_triggered(self, event, board):
		if((event, board) in self.metadata.index):
			selftrig_dec = self.metadata["reg_self_trig"][(event, board)]
			selftrig_binstr = str(bin(selftrig_dec))
			chlist = []
			for i in range(len(self.get_chs()) + 1):
				j = -1*i
				if(selftrig_binstr[j] == 'b'):
					break

				if(selftrig_binstr[j] == '1'):
					chlist.append(i)
			return chlist

		else:
			return None

	#
	def find_closest_events(self, event, board):
		pass



