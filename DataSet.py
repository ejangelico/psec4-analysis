import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys
import pandas as pd 
import scipy.io as spio
from itertools import groupby
import Waveform
import pickle 
from scipy import optimize
import time
import math

class DataSet:
	#A class that is a CSV reader for ACDC ascii data. 
	#The class turns the input file into a pandas dataframe.
	#Every time an event is requested, the class uses that
	#dataframes random access attributes to pull data. 

	#filename: .acdc filename. assumes metadata file is the same name with .meta tag
	#acdc_clock: the onboard ACDC write clock frequency in MHz. used to calculate avg timestep
	def __init__(self, filename, acdc_clock=25):

		self.data = pd.read_csv(filename+".acdc", sep=" ", index_col=[0, 1, 2])

		#peddata not used yet
		#self.peddata = pd.read_csv(self.filename[:-4]+"ped",delimiter='\s+', index_col=[0,1])
		self.metadata = pd.read_csv(filename+".meta",sep=" ", index_col=[0,1])

		self.nsamples = 256

		self.timestep = 1.0e9/(self.nsamples*acdc_clock*1e6) #nanoseconds


	#this function returns a waveform object
	#by accessing the CSV pandas dataframe 
	#and assuming a constant sampling time. 
	def get_waveform(self, event, board, channel):
		ADCcounts_to_mv = 1200./4096.
		if((event, board, channel) in self.data.index):
			mv_values = self.data.loc[(event, board, channel)].values.tolist()
			#cludge because chained data somehow ends up different from unchained data
			if(len(mv_values) == 1):
				mv_values = mv_values[0]
			mv_values = [_*ADCcounts_to_mv for _ in mv_values]
			times = [i*self.timestep for i in range(self.get_nsamples())]
			thewfm = Waveform.Waveform(mv_values, times)
			#remove spikes
			#thewfm.remove_spikes()
			return thewfm

		else:
			#print "Could not find that channel/event/board in dataset"
			return None


	#get the max number of events in dataset. 
	#this does not ensure that every board 
	#is recorded in each event.
	def get_max_events(self):
		return max(self.data.index.levels[0])

	#list of channel numbers. e.g. range(1, 31)
	def get_chs(self):
		return list(self.data.index.levels[2])

	#list of board indices. all boards that appear
	#as having triggered in the dataset.
	def get_boards(self):
		return list(self.data.index.levels[1])

	#check which board triggered for a given event
	def which_boards_triggered(self, event):
		all_possible_boards = self.get_boards()
		triggered_boards = []
		ex_channel = self.get_chs()[0] #an example channel as a filler
		for b in all_possible_boards:
			if((event, b, ex_channel) in self.data.index):
				triggered_boards.append(b)

		return triggered_boards


	#assuming that every event
	#has the same number of samples
	#for each waveform channel, return 
	#that number of samples. 
	def get_nsamples(self):
		if(self.nsamples is None):
			return len(self.data.iloc[0])
		else:
			return self.nsamples

	#timestep based on input to dataset object
	def get_timestep(self):
		return self.timestep

	#plots all channels for a list of boards and event on the same plot
	def plot_waveforms_overlayed(self, event, boards, channels):
		fig, ax = plt.subplots(figsize=(13, 9))
		for j, bo in enumerate(boards):
			for i, ch in enumerate(channels):
				thewav = self.get_waveform(event, bo, ch)
				if(thewav is None):
					continue
				thewav.plot(ax)
				ax.get_lines()[-1].set_color('k')
				ax.get_lines()[-1].set_linewidth(3)
				ax.set_xlim([5, 25])
				#ax.set_title("CH = " + str(ch))
				ax.set_xlabel('sample time (ns)', fontsize=16)
				ax.set_ylabel('sampled voltage (mV)', fontsize=16)
				ax.get_xaxis().set_tick_params(labelsize=15, length=14, width=2, which='major')
				ax.get_xaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
				ax.get_yaxis().set_tick_params(labelsize=15,length=14, width=2, which='major')
				ax.get_yaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')

		return ax


	#plots all waveforms for a list of boards on individual
	#subplots per channel. 
	def plot_waveforms_separated(self, event, boards, channels):

		if(len(channels) < 5):
			ncols = 2
			nrows = 2
		else:
			ncols = 6 #for six channels per board
			nrows = int(math.ceil(float(len(channels))/ncols))

		fig, ax = plt.subplots(figsize=(12,7),ncols=ncols, nrows=nrows)
		fig.suptitle("Event " + str(event))
		#flatten ax array to 1D
		ax_1d = []
		for _ in ax:
			for i in range(len(_)):
				ax_1d.append(_[i])

		colors = ['r','g', 'b', 'm', 'c']
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

	#plots a single board's event as a heatmap with time
	#on the x axis and channel number on y axis, color = mV
	def plot_event_heatmap(self, event, board, ax=None):
		if(ax is None):
			fig, ax = plt.subplots()
		#requires that all channels have the
		#same time values 
		x = []
		y = []
		z=[]
		channels = self.get_chs()
		for j, ch in enumerate(channels):
			wfm = self.get_waveform(event, board, ch)
			if(wfm is None):
				return
			times = wfm.get_times()
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
		h = ax.hist2d(x, y, bins=[xbins, ybins], weights=z, cmap=plt.inferno())#, cmax=20, cmin=-70)
		plt.colorbar(h[3],ax=ax)


	#Plots numEvents random events from the dataset
	def plot_random_events(self, nevts, boards=None, channels=None):
		maxev = self.get_max_events()
		if(channels is None):
			channels = self.get_chs()
		if(boards is None):
			boards = self.get_boards()

		
		for i in range(nevts):
			fig, ax = plt.subplots(nrows = len(boards))
			ev = np.random.randint(maxev)
			for b in boards:
				print(self.which_channels_triggered(ev, b))

			if(len(boards) == 1):
				self.plot_event_heatmap(ev, boards[0], ax)
				ax.set_title("event " + str(ev))
			else:
				for j, a in enumerate(ax):
					self.plot_event_heatmap(ev, boards[j], a)
					a.set_title("event " + str(ev))

			plt.show()

	#loops through each event, plotting each time. 
	def plot_all_events(self, boards=None, channels=None):
		maxev = self.get_max_events()
		if(channels is None):
			channels = self.get_chs()
		if(boards is None):
			boards = self.get_boards()

		for ev in range(maxev):
			fig, ax = plt.subplots(nrows = len(boards))
			for b in boards:
				print(self.which_channels_triggered(ev, b))

			if(len(boards) == 1):
				self.plot_event_heatmap(ev, boards[0], ax)
				ax.set_title("event " + str(ev))
			else:
				for j, a in enumerate(ax):
					self.plot_event_heatmap(ev, boards[j], a)
					a.set_title("event " + str(ev))

				
			plt.show()



	def plot_all_events_separated(self, boards=None, channels=None):
		maxev = self.get_max_events()
		if(channels is None):
			channels = self.get_chs()
		if(boards is None):
			boards = self.get_boards()


		for ev in range(maxev):
			for b in boards:
				print(self.which_channels_triggered(ev, b))

			self.plot_waveforms_separated(ev, boards, channels)

				
			plt.show()


#------------ metadata functions -----------------#
	#return a single metadata element
	def get_metadata_element(self, event, board, key):
		cols = list(self.metadata.columns.values)
		if((event, board) in self.metadata.index):
			return self.metadata[key][(event, board)]
		else:
			print("Could not find metadata for event " + str(event) + " and board " + str(board))

	#this method is used to fill the metadata class. 
	#it returns a dictionary of keys and values for metadata. 
	def get_metadata_structure(self, event, board):
		cols = list(self.metadata.columns.values)
		metadict = {}
		if((event, board) in self.metadata.index):
			for c in cols:
				metadict[c] = self.metadata[c][(event, board)]
			return metadict
		else:
			print("Could not find metadata for event " + str(event) + " and board " + str(board))
			return None


	#print a table of statistics on
	#which boards triggered on which events
	#and what multiplicity throughout the dataset
	def print_board_statistics(self):
		maxev = self.get_max_events()
		boards_allevs = [] #list of all event board numbers, including empty []
		for ev in range(maxev):
			boards_allevs.append(self.which_boards_triggered(ev))


		keys = []
		for el in boards_allevs:
			if(el in keys):
				continue
			else:
				keys.append(el)

		freqs = [0 for _ in keys]

		for el in boards_allevs:
			for i, k in enumerate(keys):
				if(el == k):
					freqs[i] += 1

		print("-----Board multiplicities-----")
		for i, k in enumerate(keys):
			print("Boards: ", end='')
			if(len(k) == 0):
				print("no-board-trigs,", end='')
			else:
				for b in k:
					if(b == k[-1]):
						print(str(b) + ",", end='')
					else:
						print(str(b) + "&", end='')

			print("\t\t", end='')
			print(str(freqs[i]))



	#board_combs is a list of board combinations that
	#are allowed to pass the cut. = [[0,2],[1,3],[1,2,3,4]...]
	def get_event_nos_with_boards(self, board_combs):
		maxev = self.get_max_events()
		pass_events = []
		for ev in range(maxev):
			bs = self.which_boards_triggered(ev) #which ones are in this event
			for bc in board_combs:
				bc.sort()
				bs.sort()
				if(bc == bs):
					#todo: check if they are within a clock cut
					pass_events.append(ev)
					break

		return pass_events









			










