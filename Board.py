import Event
import Waveform
import numpy as np 
import matplotlib.pyplot as plt 
import time
import scipy.optimize as optimize
import sys
import math
import copy 
import pickle
import random
import Pulse
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import copy


class Board:
	def __init__(self, waveform_list, chs, board_number, metadata, sync_chs = None, endedness = "single"):
		self.nch = len(waveform_list)
		self.wfms = waveform_list
		self.no = board_number
		self.chs = chs #list of channel numbers ordered with wfm list. goes from 1 to 30
		self.bad_chs = [] #channels to be ignored in some analyses

		#sync info
		self.syncs = sync_chs #channels that have synchronization channels
		self.sync_offsets = None #time to shift each sync channel. set after construction or in analysis
		self.sync_wave_params = [] #the list of sine wave parameters from fit for each sync channel


		#strip "endedness" is whether it is
		#single ended or double ended readout. 
		self.endedness = endedness
		self.channel_pairings = [] #a list of pairs of channels that correspond to [left, right]
		self.adjust_pairings()


		#position reco
		self.ch_velocities = [119 for _ in self.chs] #mm/ns
		self.strip_lengths = [1.9 for _ in self.chs] #ns
		self.strip_positions = {}
		self.strip_lengths = {}
		self.strip_velocities = {}
		self.set_default_strip_properties()

		

		self.board_loc = None #set after construction, used for plots

		#these dictionaries indexed by channel number
		self.nnlswavs = {}
		self.template_wav = None
		self.components = {}  #[[time, scale], [time, scale], ...] where time is the raw waveform time index location

		#output from the pulse finder algorithms
		self.pulses = {} #indexed by channel

		self.metadata = metadata # a dictionary of metadata for this board and event.

		self.trigger_time_guess = None

		
	




	def get_board_no(self):
		return self.no

	def get_wfms(self):
		return self.wfms

	def get_sync_chs(self):
		return self.syncs

	def get_waveform(self, channel):
		idx = self.chs.index(channel)
		return self.wfms[idx]

	def get_chs(self):
		return self.chs

	def get_loc(self):
		return self.board_loc

	def get_position_forms(self):
		return self.position_forms


	def set_strip_properties(self, strip_vel, strip_len, strip_pos):
		self.strip_velocities = strip_vel 
		self.strip_lengths = strip_len
		self.strip_positions = strip_pos


	def set_default_strip_properties(self):
		strip_pitch = 6.9 #mm

		self.strip_positions = {}
		self.strip_lengths = {}
		self.strip_velocities = {}

		if(self.endedness == "double"):
			#all indexed by strip number, not channel. 
			for number in range(1, 29):
				self.strip_lengths[number] = 1.9 #ns
				self.strip_velocities[number] = 119 #mm/ns
				#positions defined relative to the center of the tile. 
				#center splits two strips. numbers 1-14 are top half, 15-28 are bottom. 
				self.strip_positions[number] = (15 - number - 0.5)*strip_pitch

		elif(self.endedness == "single"):
			#all indexed by strip number, not channel. 
			for number in range(1, 31):
				self.strip_lengths[number] = 1.9 #ns
				self.strip_velocities[number] = 119 #mm/ns
				#positions defined relative to the center of the tile. 
				#center splits two strips. numbers 1-14 are top half, 15-28 are bottom. 
				self.strip_positions[number] = (16 - number - 0.5)*strip_pitch

	#Location string mapping this board to an LAPPD
	#and location. 
	def set_board_loc(self, loc):
		self.board_loc = loc



	def median_subtract(self):
		for w in self.wfms:
			w.median_subtract()

	def get_trigger_time_guess(self):
		return self.trigger_time_guess

	#set list of sync channel numbers
	def set_syncs(self, syncs):
		self.syncs = syncs 


	def get_pulses(self):
		return self.pulses
		
	#channels to be ignored in some analyses
	def set_bad_chs(self, chs):
		self.bad_chs = chs
		#remove pulses from bad channels
		#if they were already loaded/found
		for bch in self.bad_chs:
			if(bch in self.pulses):
				self.pulses.pop(bch)

	def get_strip_position(self, num):
		return self.strip_positions[num]


	def remove_spikes(self):
		for w in self.wfms:
			w.remove_spikes()


	#this function is somewhat specific use, 
	#removing times and signals from a wave
	#as specified in nanoseconds in cut_ranges
	#only use at the moment is to remove
	#trigger regions in software triggered time
	#calibration data. 
	def cut_waveform_times(self, cut_ranges):
		for w in self.wfms:
			for cr in cut_ranges:
				w.trim_range_out(cr) #modification of waveform. 



	#stores a list of channel "pairings"
	#so that we know which channel is connected to 
	#eachother. Based on a constructor variable "endedness"
	def adjust_pairings(self):
		self.channel_pairings = []
		if(self.endedness == "single"):
			if(self.syncs is not None):
				self.channel_pairings = [[_, _] for _ in self.chs if _ not in self.syncs]
			else:
				self.channel_pairings = [[_, _] for _ in self.chs]
		elif(self.endedness == "double"):
			if(self.syncs is not None):
				for ch in self.chs:
					if((29 - ch) in self.syncs):
						self.channel_pairings.append([ch, ch])
						continue
					if(15 <= ch <= 30):
						continue
					if(ch in self.syncs):
						continue
					else:
						self.channel_pairings.append([ch, 29-ch])
			else:
				for ch in self.chs:
					if(15 <= ch <= 28):
						continue
					self.channel_pairings.append([ch, 29-ch])

		#sort by first channel
		self.channel_pairings = sorted(self.channel_pairings, key=lambda x: x[0])



	def print_pairings(self):
		print(self.channel_pairings)


	#this function returns the pair channel
	#to the input channel. 
	def get_channel_pair(self, ch):
		for chp in self.channel_pairings:
			if(ch in chp):
				temppair = [_ for _ in chp]
				temppair.remove(ch)
				return temppair[0]

		return None


	#this function returns which chip the channel
	#is on. This is constant regardless of readout, 
	#and is a property of the ACDC boards. 
	def get_chip_number(self, ch):
		if(ch in [1,2,3,4,5,6]):
			return 0
		if(ch in [7,8,9,10,11,12]):
			return 1
		if(ch in [13,14,15,16,17,18]):
			return 2
		if(ch in [19,20,21,22,23,24]):
			return 3
		if(ch in [25,26,27,28,29,30]):
			return 4



	#this function calculates the strip number 
	#given a channel number and specific endedness
	def get_strip_number(self, ch):
		if(ch in self.syncs):
			return None

		#this is the "annie readout board"
		if(self.endedness == "double"):
			#NOT CODED IN YET: board loc will give
			#you whether it is part of block 1-14
			#or 15-28. Presently, just assuming strip
			#1-14

			if(ch < 15):
				return (ch)
			elif(15 <= ch <= 28):
				return (29 - ch)
			else:
				return None #not right


		elif(self.endedness == "single"):
			return ch 

	#tells you which side of a strip the channel
	#is looking at. 
	#Returns 
	#1 if on 'right side' closer to bottom MCP SHV inputs
	#-1 if on 'left side' closer to top MCP SHV inputs
	def get_strip_side(self, ch):
		if(ch in self.syncs or ch in self.bad_chs):
			return None

		if(len(self.channel_pairings) == 0):
			print("no channel pairings on request of strip side")
			return None

		#get the channel's pair
		chpair = None
		for pair in self.channel_pairings:
			if(ch in pair):
				chpair = pair 
				break 
		if(chpair is None):
			print("Channel is not in pairings on request of strip side")
			return None

		
		#if the BOARD endedness is single ended, then 
		#the channels are all on one strip side. For 
		#FTBF run, it was right side. 
		if(self.endedness == "single"):
			return 1 

		else:
			#use the mapping of the annie readout board. 	
			if(ch >= 15):
				return 1
			else:
				return -1


	#occasionally, like in poorly coupled SAMTECS of
	#the ANNIE board, one side of a strip is badly terminated
	#or open-ended. This is stored as a relation between
	#the pairing and bad channels. I set channels as being
	#bad if their mate sees reflected pulses. But, then i need
	#to know if it is on the left side and if it is single ended. 
	#return True if single ended. 
	def is_channel_single(self, ch):
		if(ch in self.syncs or ch in self.bad_chs):
			return None

		if(len(self.channel_pairings) == 0):
			print("no channel pairings on request of strip side")
			return None

		#get the channel's pair
		chpair = None
		for pair in self.channel_pairings:
			if(ch in pair):
				chpair = pair 
				break 

		#if it is its own pair, it is single ended
		if(chpair[0] == chpair[1]):
			return True

		#if it is paired with a bad channel, return single ended
		if(chpair[0] in self.bad_chs or chpair[1] in self.bad_chs):
			return True 

		else:
			return False 


	#gets the transverse position of the strip number. 
	def get_strip_position(self, number):
		return self.strip_positions[number]




	def set_endedness(self, e):
		if(e == "single" or e == "double"):
			self.endedness = e
			self.adjust_pairings()
		else:
			print("only supports 'single' and 'double' endedness. setting to default of single")
			self.endedness = "single"
		

	#time offsets for each sync channel to synchronize
	#chips. list should match sync channel list
	def set_sync_offsets(self, offsets):
		self.sync_offsets = offsets
		if(self.syncs is not None):
			if(len(offsets) != len(self.syncs)):
				print("Length of sync channels doesn't match sync offsets")


	#return a list of relative time offsets
	#of synchronization signals relative to the
	#first sync channel. 
	def get_sync_wave_offsets(self):
		if(len(self.sync_wave_params) == 0):
			return self.sync_wave_params #empty list...

		def sin_fitfunc(x, amp, freq, phas, off):
			return (amp*np.sin(x*freq*2*np.pi - phas) + off)

		delays = {}
		reference = None #the reference time in ns for reference channel
		for i, params in enumerate(self.sync_wave_params):
			phase = params[2] #radians
			amp = params[0] #mV
			freq = params[1] #GHz
			tau = (1.0/freq)*(phase/(2*np.pi)) #ns - check fitfunction

			delays[self.syncs[i]] = (tau) #ns



		return delays


	#this function does specific computations
	#related to doing timing and correlated noise
	#calibrations from sync waves. 
	def get_calibration_data(self, good_times):

		def sin_fitfunc(x, amp, freq, phas, off):
				return (amp*np.sin(x*freq*2*np.pi - phas) + off)

		#deriv
		def deriv_fitfunc(x, amp, freq, phas, off):
				return (amp*freq*2*np.pi*np.cos(x*freq*2*np.pi - phas))


		result_dict = {}
		for ch in self.syncs:
			result_dict[ch] = {'resids': {}, 'deriv': {}, 'fits': {}, 'popts': []}
		#this data is assumed to have a bit cut out of
		#it where the software trigger triggered the
		#SCAs (see def cut_waveform_times). 
		for ch in self.syncs:
			#first step is to find the location where there is
			#a big gap in the waveform times. 
			sw = self.get_waveform(ch)
			times = sw.get_times()
			signal = sw.get_signal()
			ppts = []

			for block in good_times:
				#fit sine waves on either end of these bad samples. 
				popt, resid, _ = sw.get_sin_wave_information(block)
				

				for i, t in enumerate(times):
					if(min(block) < t < max(block)):
						result_dict[ch]['resids'][i] = sin_fitfunc(t, *popt) - signal[i]
						result_dict[ch]['deriv'][i] = deriv_fitfunc(t, *popt)
						result_dict[ch]['fits'][i] = sin_fitfunc(t, *popt)

				ppts.append(popt)

			result_dict[ch]['popts'] = ppts


		return result_dict


	#this uses the knowledge of the annie board
	#in combination with a 6 way sync situation
	#to return the synchronization constant offset
	#given a channel number and an assumed chip
	#that that channel is associated with. Only the
	#test-beam board configurations are considered here. 
	def get_sync_offset_from_channel(self, ch):
		if(len(self.sync_wave_params) == 0):
			self.find_sync_waves()
			#if it couldnt find them...
			if(len(self.sync_wave_params) == 0):
				print("Couldn't find sync waves")
				return None

		#this is an annie readout board in combination with
		#a 5 strip sync splitter, one for each chip. Sometimes
		#sync channels are bad, in that case just use the
		#offset from the nearest channel. Also works for the
		#FTBF config of capacitively coupled readout, single ended. 
		if((self.endedness == "double" and len(self.syncs) > 2) \
			or (self.endedness == "single" and len(self.syncs) > 2)):
			channel_associations = {6: [1,2,3,4,5], 12: [7,8,9,10,11], \
									18: [13,14,15,16,17], 24: [19,20,21,22,23],\
									30: [25,26,27,28,29]}
			ch_offs = self.get_sync_wave_offsets()
			for i, schan in enumerate(ch_offs):
				#if the input channel is on a chip
				#with this chan as the sync chan 
				if(ch in channel_associations[schan]):
					#we have found the desired sync chan
					desired_sync_chan = schan
					if(schan in ch_offs):
						return ch_offs[schan]

			#if it came out of this loop, this channel isnt
			#on a chip with a sync chan active, or a sync chan 
			#that was fitted. 

			#Priority is if someone has set a calibration from calibration
			#data. 
			if(self.sync_offsets is not None):
				#use the desired sync chan found above
				if(desired_sync_chan in self.sync_offsets):
					return self.sync_offsets[desired_sync_chan]

			#if we made it here, it isnt in calibrated offsets. Just
			#use the one of the sync chans in sync offsets calculated. 
			if(len(ch_offs) != 0):
				for ch in ch_offs:
					return ch_offs[ch]

			#otherwise, I'm at a loss
			return None

		#This is the same, but if you are only using the ch 29 or 30 as sync
		#channels. 
		elif((self.endedness == "double" and 1 <= len(self.syncs) <= 2) \
			or (self.endedness == "single" and 1 <= len(self.syncs) <= 2)):
			#just use the average of all sync offsets. 
			ch_offs = self.get_sync_wave_offsets()
			offset_list = [ch_offs[_] for _ in ch_offs]
			return np.mean(offset_list)




	#returns a list of all samples on all channels
	#ignoring the ch_reject list elements
	def get_all_samples(self, ch_reject=[]):
		sam_dict = {}
		#initialize dictionary
		for ch in self.chs:
			if(ch in ch_reject):
				continue
			sam_dict[ch] = []

		#fill with all samples
		for ch in self.chs:
			if(ch in ch_reject):
				continue
			w = self.get_waveform(ch)
			sam_dict[ch] += w.get_signal()

		return sam_dict


	#shifts all waveforms in the board
	#by constant amount "shift_val" assuming
	#that the wave is a circular buffer
	def shift_board(self, shift_val):
		for w in self.wfms:
			w.time_shift(shift_val)



	def find_sync_waves(self):

		#a normal analysis
		
		self.find_trigger_time() 
		#if no guess for trigger time, the fit will be wrong.
		if(self.trigger_time_guess is None):
			return

		sync_wfms = [self.get_waveform(_) for _ in self.syncs]


		#the trigger time guess is used to 
		#ignore a region of corrupt samples from
		#the dataset. There are also timing offsets
		#associated with these samples, so we only
		#fit on one side of the event buffer, the side
		#with the most number of samples outside of the 
		#trigger region. 
		trigger_region_ignore_width = 2 #ns
		maxtime = max(sync_wfms[0].get_times())
		mintime = min(sync_wfms[0].get_times())
		#find which side of the event buffer to fit. 
		if(abs(self.trigger_time_guess - maxtime) > abs(self.trigger_time_guess - mintime)):
			#do the left side side
			fit_timerange = [self.trigger_time_guess + trigger_region_ignore_width, maxtime]
		else:
			#do the right
			fit_timerange = [mintime, self.trigger_time_guess - trigger_region_ignore_width]
		
		for sw in sync_wfms:
			popt, res, err = sw.get_sin_wave_information(fit_timerange, plot=False)
			if(popt is not None):
				self.sync_wave_params.append(popt)

	

	#this function looks at sync waveforms
	#and performs a "rolling fit". I.e., 
	#it tries to fit a sine wave in small chunks
	#and defines the wraparound point as the point
	#where all of the fits start having a large residual. 
	def find_trigger_time(self):
		if(self.syncs is None):
			return
		sync_wfms = [self.get_waveform(_) for _ in self.syncs]


		timeblock_width = 10 #ns
		residual_threshold = 1000 #squared residual #350 is 5% finding losses, 4% error
		base_range = [0, timeblock_width] #slide this base range in 1 ns increments
		timeranges = [[base_range[0] + i, base_range[1] + i] for i in range(26)]

		guess_times = []
		#first is a coarse scan
		for w in sync_wfms:
			#at each time range, find the summed residual. 
			summed_res = [] 
			for tr in timeranges:
				#if the timerange leaves the region of the waveform
				if(max(tr) > max(sync_wfms[0].get_times())):
					break

				popt, res, err = w.get_sin_wave_information(tr)
				if(popt is None):
					summed_res.append(1e6) #need to keep indexing correct, so placing a very large value
					continue
				ressquared = [_**2 for _ in res]
				summed_res.append(np.sum(ressquared)) 

			#find the index of the minimum summed residual fit. 
			#this is a "golden fit" that we will use. The peak
			#residual of this fit over the whole waveform range
			#will represent where the samples become corrupt. 
			minidx = summed_res.index(min(summed_res))
			best_timerange = timeranges[minidx]
			#get fit information for the best time range
			popt, _, _ = w.get_sin_wave_information(best_timerange)
			#calculate residual over the WHOLE range. 
			res = w.get_sin_residuals(popt, None) #None makes it over whole range. 
			ress = [_**2 for _ in res]


			#do a threshold test to find where this
			#residual starts blowing up. search around
			#the middle of the best timerange. 
			startidx = w.get_sample_at(np.mean(best_timerange))
			guess_idx = None
			for i in range(len(ress)):
				ihi = startidx + i
				ilo = startidx - i
				#if upper search is in range
				if(ihi < len(ress)):
					#is it over threshold
					if(ress[ihi] > residual_threshold):
						guess_idx = ihi
						break
				#if lower search is in range
				if(ilo >= 0):
					if(ress[ilo] > residual_threshold):
						guess_idx = ilo
						break
			
			#nothing passed threshold
			if(guess_idx is None):
				continue

			guess_times.append(w.get_times()[guess_idx])

		if(len(guess_times) != 0):
			self.trigger_time_guess = np.median(guess_times)


		"""
		fig, ax = plt.subplots()
		self.plot_sync_channels(ax)
		if(len(guess_times) == 0):
			print("No trigger location was found")
		else:
			self.trigger_time_guess = np.median(guess_times)
			ax.axvline(np.median(guess_times), color='k', linewidth=3)

		ax.set_xlabel("time (ns)")
		ax.set_ylabel("mV")
		plt.show()
		#plt.savefig("plots/wraparound_finder/"+str(random.randint(0, 100))+".png", bbox_inches='tight')
		"""





	def plot_waveforms_offset(self, channels=None, ax=None):
		if(channels is None):
			channels = range(1, 31)
		if(ax is None):
			fig, ax = plt.subplots(figsize=(13, 7.8))


		voltage_offset = 25 #mv offset per channel on y-axis

		colors = ['r','g','b','m','k','y']
		for i, ch in enumerate(channels):
			wfm = self.wfms[ch-1]
			col = colors[int(i % len(colors))]
			newsig = [_+i*voltage_offset for _ in wfm.get_signal()]
			ax.plot(wfm.get_times(), newsig, col)

		ax.set_ylabel("shifted signal (a.u.)", fontsize=15)
		ax.set_xlabel("time (ns)", fontsize=15)

		ax.get_xaxis().set_tick_params(labelsize=15, length=14, width=2, which='major')
		ax.get_xaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
		ax.get_yaxis().set_tick_params(labelsize=15,length=14, width=2, which='major')
		ax.get_yaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
		majorLocatorX = MultipleLocator(5)
		minorLocatorX = MultipleLocator(1)
		ax.get_xaxis().set_major_locator(majorLocatorX)
		ax.get_xaxis().set_minor_locator(minorLocatorX)






	#assumes 
	def plot_board_separated(self, ax):
		#flatten ax array to 1D
		ax_1d = []
		for _ in ax:
			for i in range(len(_)):
				ax_1d.append(_[i])

		colors = ['r','g','b','c']
		for i, ch in enumerate(self.chs):
			thewav = self.wfms[i]
			if(thewav is None):
				continue

			thewav.plot(ax_1d[i], colors[int(self.no)])
			ax_1d[i].set_title(str(ch))


			#plot nnls waves if they exist
			if(ch in self.nnlswavs):
				wav = self.nnlswavs[ch]
				wav.plot(ax_1d[i])
				ax_1d[i].get_lines()[-1].set_color(colors[int(self.no)])
				ax_1d[i].get_lines()[-1].set_alpha(0.4)


			#plot pulse regions if they exist
			if(ch in self.pulses):
				ps = self.pulses[ch]
				for p in ps:
					arr = p.get_arrival_time()
					rise = p.get_rise_time()
					peak_time = p.get_peak_time()
					peak = -1*p.get_amplitude()

					ax_1d[i].axvspan(arr, arr+rise, color=colors[int(self.no)], alpha=0.2)
					ax_1d[i].scatter([peak_time], [peak], color=colors[int(self.no)], s=5)


		

	def plot_board_ended(self, fig = None, ax = None):
		#create set of subplots that matches the
		#number of paired channels. 
		l = len(self.channel_pairings)
		if(l <= 12):
			ncols = 4
			nrows = 3
		elif(12 < l <= 30):
			ncols = 6
			nrows = 5
		else:
			ncols = 6
			nrows = 6

		if(ax is None):
			fig, ax = plt.subplots(figsize=(12, 6), ncols=ncols, nrows=nrows)

			gs1 = gridspec.GridSpec(ncols, nrows)
			gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

		#flatten ax array to 1D
		ax_1d = []
		for _ in ax:
			for i in range(len(_)):
				ax_1d.append(_[i])

		if(len(ax_1d) < l):
			print("not enough subplots to plot the pairs on board " + str(self.no)) 
			return ax

		colors = ['r','g','b','c']
		for i, chp in enumerate(self.channel_pairings):
			for j, ch in enumerate(chp):
				chindex = self.chs.index(ch)
				thewav = self.wfms[chindex]
				if(thewav is None):
					continue

				thewav.plot(ax_1d[i], colors[int(self.no)])
				if(j == 1):
					ax_1d[i].get_lines()[-1].set_alpha(0.5)
				if(j == 0):
					ax_1d[i].set_title(str(chp))

				#plot pulse regions if they exist
				if(ch in self.pulses):
					ps = self.pulses[ch]
					for p in ps:
						arr = p.get_arrival_time()
						rise = p.get_rise_time()
						peak_time = p.get_peak_time()
						peak = -1*p.get_amplitude()

						ax_1d[i].axvspan(arr, arr+rise, color=colors[int(self.no)], alpha=0.2)
						ax_1d[i].scatter([peak_time], [peak], color=colors[int(self.no)], s=5)



		
	def plot_sync_channels(self, ax):
		colors = ['r','g','b','c', 'm', 'k']
		for i, ch in enumerate(self.syncs):
			thewav = self.get_waveform(ch)
			if(thewav is None):
				continue

			thewav.plot(ax, colors[i], label=str(ch))

		ax.set_xlabel("time (ns)", fontsize=15)
		ax.set_ylabel("mV", fontsize=15)
		ax.get_xaxis().set_tick_params(labelsize=15, length=14, width=2, which='major')
		ax.get_xaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
		ax.get_yaxis().set_tick_params(labelsize=15,length=14, width=2, which='major')
		ax.get_yaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
		majorLocatorY = MultipleLocator(10)
		minorLocatorY = MultipleLocator(5)
		majorLocatorX = MultipleLocator(4)
		minorLocatorX = MultipleLocator(2)
		ax.get_xaxis().set_major_locator(majorLocatorX)
		ax.get_xaxis().set_minor_locator(minorLocatorX)
		ax.get_yaxis().set_major_locator(majorLocatorY)
		ax.get_yaxis().set_minor_locator(minorLocatorY)
		ax.title.set_text("board number " + str(self.no))




	#this function presently
	#finds time indices of peaks
	#in the nnls full fit waveform
	#using the peakutils library.
	#Threshold in mV
	def get_peak_indices(self, threshold, mindist):
		if(len(self.nnlswavs) == 0):
				self.nnls_all()

		pi_dict = {} #indexed by channel
		for ch in self.chs:
			if(ch in self.bad_chs):
				continue
			if(ch in self.syncs):
				continue

			wfm = self.nnlswavs[ch]
			wfmtimes = wfm.get_times()
			peak_indices = wfm.find_peaks_peakutils(threshold, mindist)
			#no peaks found. 
			if(len(peak_indices) == 0):
				continue

			pi_dict[ch] = peak_indices

			"""
			peak_times = [wfmtimes[_] for _ in peak_indices]
			fig, ax = plt.subplots(figsize=(10, 7))
			self.get_waveform(ch).plot(ax)
			wfm.plot(ax, 'b-')
			for pt in peak_times:
				ax.axvline(pt)

			ax.set_ylabel("mV", fontsize=15)
			ax.set_xlabel("time (ns)", fontsize=15)
			ax.get_xaxis().set_tick_params(labelsize=15, length=14, width=2, which='major')
			ax.get_xaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
			ax.get_yaxis().set_tick_params(labelsize=15,length=14, width=2, which='major')
			ax.get_yaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
			majorLocatorX = MultipleLocator(4)
			minorLocatorX = MultipleLocator(1)
			ax.get_xaxis().set_major_locator(majorLocatorX)
			ax.get_xaxis().set_minor_locator(minorLocatorX)

			plt.show()
			"""


		return pi_dict





	#assumes that it is a waveform objection from a pickle file
	def set_template_waveform(self, templatefile):
		wfm = pickle.load(open(templatefile, "rb"))
		wfm.zero_times()

		#make it so that this waveform has
		#the same sampling rate as the data
		#waveforms.
		datatimes = self.wfms[0].get_times() #times for data waveforms
		template_sigs = []
		template_times = []
		for t in datatimes:
			s = wfm.get_signal_at(t)
			if(s is None):
				continue
			else:
				template_sigs.append(s)
				template_times.append(t)

		self.template_wav = Waveform.Waveform(template_sigs, template_times)
		return self.template_wav



	def get_template_waveform(self):
		if(self.template_wav is None):
			print("No template file set and attempted to get template waveform")
			sys.exit()

		else:
			return self.template_wav


	#Assumes the existence of nnls 
	#waveforms. Uses a peak finding algorithm
	#to find peaks. Treats each peak location like a pulse
	#and calculates/retains its 
	#(1) 90-10 risetime
	#(2) 10% CFD arrival time
	#(3) Charge integrated from 10% to first drop below 10%
	#(4) Amplitude
	#in the form of a pulse object
	def find_pulses(self):

		#a constant fraction discriminator
		#assuming negative polar pulses, i.e.
		#assuming amp is negative, and monotonic
		#decrease in signal. i.e. only negative slope
		#the whole way down the edge. direction is
		#-1 for backward, +1 for forward
		def CFD(wave, amp, f, peakidx, direction):
			#first find the two samples
			#in between which the signal goes
			#to f*amp. 
			hi_samp = None
			lo_samp = None
			sig = wave.get_signal()
			ts = wave.get_times()
			i = peakidx

			#this loop allows for a circular buffer wraparound. 
			#but, I want an explicit break condition in case one
			#never finds a f*amp point even after wrapping all the way 
			#around. 
			looped_around = False
			while True:
				if(looped_around and i == peakidx):
					break
				if(i == -1):
					i = len(ts) - 1 #last index
					looped_around = True
				if(i == len(sig)):
					i = 0
					looped_around = True
				#print("Sig: " + str(sig[i]) + ", time: " + str(ts[i]))
				if(sig[i] == amp*f):
					#exact! rare
					return ts[i]
				elif(sig[i] > amp*f):
					#if it just passes threshold
					hi_samp = i
					lo_samp = i - np.sign(direction) #protected by break statement above
					break

				i += np.sign(direction)

			#if it reaches the end without finding the point. 
			if(hi_samp is None or lo_samp is None):
				return None

			#if the last sample was a wraparound one
			if(lo_samp == len(sig)):
				lo_samp = 0

			#otherwise, interpolate between the two samples linearly
			x0 = ts[lo_samp]
			x1 = ts[hi_samp]
			y0 = sig[lo_samp]
			y1 = sig[hi_samp]
			m = (y1 - y0)/(x1 - x0)
			#if m is 0, take the hi samp
			if(abs(m) < 1e-6):
				return ts[hi_samp]
			b = y1 - m*x1
			#solve for the desired signal value
			x_t = (f*amp - b)/m
			#print(",".join([str(_) for _ in [x0, x1, y0, y1, m, b, x_t]]))
			#print("Cought, interpolated to " + str(x_t))
			return x_t



		#coarse peak finder
		thresh = 20
		mindist = 20
		peak_dict = self.get_peak_indices(thresh, mindist) #dictionary indexed by channel

		#ns around the calculated peak index
		#for which to find the actual peak in 
		#raw waveform, and for which to spline
		#fit to find the amplitude. 
		peak_range = 1 
		peak_spline_sampling = 5 #times the sampling rate
		peak_spline_smoothing = 3 #smoothed edges is higher number

		#based on observed bugs
		allowed_rise_times = [0.5, 1.2] #if a pulse has rise-time outside this range, something went wrong.
		allowed_widths = [0.6, 8] #peak-to-10 width


		for ch in peak_dict:
			peak_indices = peak_dict[ch]
			raw_wfm = self.get_waveform(ch)
			nnls_wav = self.nnlswavs[ch]
			for pi in peak_indices:

				#the nnls finder is not perfect at 
				#fitting to peak amplitudes. From this
				#point forward we want to work with the
				#raw waveform. 

				#Start by spline fitting about a window
				#around the peak sample to get the amplitude. 
				peak_time = nnls_wav.get_times()[pi]
				trim_wfm = Waveform.Waveform(raw_wfm.get_signal(), raw_wfm.get_times())
				trim_wfm.trim_on_range([peak_time-0.5*peak_range, peak_time+0.5*peak_range])

				#sometimes the peaks found are on a falling edge of another pulse. 
				#make sure that this peak has a minimum that is not at an endpoint
				#of the range. 
				_, raw_peak_index = trim_wfm.find_absmax_sample()
				#if it is at an endpoint (i.e. on a falling edge)
				if(raw_peak_index == 0 or raw_peak_index == trim_wfm.get_num_samples() - 1):
					continue

				#spline fit around this peak
				spl_peak = trim_wfm.get_spline_waveform(peak_spline_sampling, peak_spline_smoothing)
				p_amp, p_amp_idx = spl_peak.find_absmax_sample() #returns p_amp as abs(peak). 

				#now do CFD on the *NNLS wave* to reduce noise
				#it seems to fit the rising edge incredibly well
				t_10 = CFD(raw_wfm, -1*p_amp, 0.1, pi, -1) #10% rise point
				t_90 = CFD(raw_wfm, -1*p_amp, 0.9, pi, -1) #90% rise point
				#if for some reason it cant find either of these points
				if(t_10 is None or t_90 is None):
					print("Had trouble finding 90% and 10% points")
					continue

				#if the rise is splitting an event
				#window edge. 
				if(t_10 > t_90):
					trise = (t_90 + max(raw_wfm.get_times())) - t_10
				else:
					trise = t_90 - t_10

				#sometimes happens if the 
				#particular threshold crossing point
				#is a bad place where slope is irrepresentative
				#of the pulse shape. 
				if(not(min(allowed_rise_times) <= trise <= max(allowed_rise_times))):
					continue

				tarrival = t_10

				#find the charge by finding the FORWARD 10% point
				#and integrating between backward t_10 and forward t_10
				t_10_forward = CFD(raw_wfm, -1*p_amp, 0.1, pi, +1)
				if(t_10_forward is None):
					print("Had trouble finding forward 10% point")
					continue

				#if the fall is splitting an event window edge, 
				#the integrate ranged function will handle it. 
				charge = raw_wfm.integrate_ranged([t_10, t_10_forward]) #mV*ns
				if(charge is None):
					continue

				if(not(min(allowed_widths) <= (t_10_forward - peak_time) <= max(allowed_widths))):
					#two things, either a bug or a wraparound issue. Quick temporarily
					#correct the wraparound issue. 
					if(t_10_forward < peak_time):
						temp10forward = t_10_forward + max(raw_wfm.get_times())/2.0
						temppeaktime = peak_time - max(raw_wfm.get_times())/2.0
						#check again. 
						if(not(min(allowed_widths) <= (temp10forward- temppeaktime) <= max(allowed_widths))):
							continue
					else:
						continue

				#add a pulse to the list
				if(ch in self.pulses):
					self.pulses[ch].append(Pulse.Pulse(p_amp, peak_time, tarrival, trise, charge))
				else:
					self.pulses[ch] = [Pulse.Pulse(p_amp, peak_time, tarrival, trise, charge)]

				#debugging
				"""
				fig, ax = plt.subplots()
				raw_wfm.plot(ax)
				ax.get_lines()[-1].set_linewidth(2)
				ax.get_lines()[-1].set_markersize(3)
				#nnls_wav.plot(ax, 'b-')
				ax.axvspan(t_10, t_10_forward, color='yellow', alpha=0.2)
				ax.axvspan(t_10, t_90, color='green', alpha=0.3)
				ax.set_ylabel("mV", fontsize=15)
				ax.set_xlabel("time (ns)", fontsize=15)
				ax.get_xaxis().set_tick_params(labelsize=15, length=14, width=2, which='major')
				ax.get_xaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
				ax.get_yaxis().set_tick_params(labelsize=15,length=14, width=2, which='major')
				ax.get_yaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
				majorLocatorX = MultipleLocator(4)
				minorLocatorX = MultipleLocator(1)
				ax.get_xaxis().set_major_locator(majorLocatorX)
				ax.get_xaxis().set_minor_locator(minorLocatorX)
				plt.show()
				"""


	#Identical to the find_pulses function
	#but instead only looks at a specific channel
	#in a specific range and provides a threshold input. 
	#threshold in mV and mindist in nanoseconds
	def find_pulses_specific(self, ch, rang, thresh, mindist):

		#a constant fraction discriminator
		#assuming negative polar pulses, i.e.
		#assuming amp is negative, and monotonic
		#decrease in signal. i.e. only negative slope
		#the whole way down the edge. direction is
		#-1 for backward, +1 for forward
		def CFD(wave, amp, f, peakidx, direction):
			#first find the two samples
			#in between which the signal goes
			#to f*amp. 
			hi_samp = None
			lo_samp = None
			sig = wave.get_signal()
			ts = wave.get_times()
			i = peakidx

			#this loop allows for a circular buffer wraparound. 
			#but, I want an explicit break condition in case one
			#never finds a f*amp point even after wrapping all the way 
			#around. 
			looped_around = False
			while True:
				if(looped_around and i == peakidx):
					break
				if(i == -1):
					i = len(ts) - 1 #last index
					looped_around = True
				if(i == len(sig)):
					i = 0
					looped_around = True
				#print("Sig: " + str(sig[i]) + ", time: " + str(ts[i]))
				if(sig[i] == amp*f):
					#exact! rare
					return ts[i]
				elif(sig[i] > amp*f):
					#if it just passes threshold
					hi_samp = i
					lo_samp = i - np.sign(direction) #protected by break statement above
					break

				i += np.sign(direction)

			#if it reaches the end without finding the point. 
			if(hi_samp is None or lo_samp is None):
				return None

			#if the last sample was a wraparound one
			if(lo_samp == len(sig)):
				lo_samp = 0

			#otherwise, interpolate between the two samples linearly
			x0 = ts[lo_samp]
			x1 = ts[hi_samp]
			y0 = sig[lo_samp]
			y1 = sig[hi_samp]
			m = (y1 - y0)/(x1 - x0)
			#if m is 0, take the hi samp
			if(abs(m) < 1e-6):
				return ts[hi_samp]
			b = y1 - m*x1
			#solve for the desired signal value
			x_t = (f*amp - b)/m
			#print(",".join([str(_) for _ in [x0, x1, y0, y1, m, b, x_t]]))
			#print("Cought, interpolated to " + str(x_t))
			return x_t



		#the waveform in question. 
		wfm = self.get_waveform(ch).trim_on_range_soft(rang)
		nnls_wfm = self.nnlswavs[ch].trim_on_range_soft(rang)
		#convert input mindist in ns to samples 
		nnls_times = nnls_wfm.get_times()
		dt = abs(nnls_times[0] - nnls_times[1])
		mindist = mindist/dt #samples
		#find peak indices in the nnls_wave.
		peak_indices = nnls_wfm.find_peaks_peakutils(thresh, mindist)

		#find amplitude for CFD via
		#a spline fit of the peak. 
		peak_range = 1 #ns around peak in nnls wave. 
		peak_spline_sampling = 5 #times the sampling rate
		peak_spline_smoothing = 3 #smoothed edges is higher number

		#based on observed bugs
		allowed_rise_times = [0.5, 1.2] #if a pulse has rise-time outside this range, something went wrong.
		allowed_widths = [0.6, 8] #peak-to-10 width

		
		pulses = []
		for pi in peak_indices:
			#Start by spline fitting about a window
			#around the peak sample to get the amplitude. 
			peak_time = nnls_times[pi]
			peak_trim = wfm.trim_on_range_soft([peak_time-0.5*peak_range, peak_time+0.5*peak_range])

			#sometimes the peaks found are on a falling edge of another pulse. 
			#make sure that this peak has a minimum that is not at an endpoint
			#of the range. 
			_, raw_peak_index = peak_trim.find_absmax_sample()
			#if it is at an endpoint (i.e. on a falling edge)
			if(raw_peak_index == 0 or raw_peak_index == trim_wfm.get_num_samples() - 1):
				continue

			#spline fit around this peak
			spl_peak = peak_trim.get_spline_waveform(peak_spline_sampling, peak_spline_smoothing)
			p_amp, p_amp_idx = spl_peak.find_absmax_sample() #returns p_amp as abs(peak). 

			#now do CFD on the raw wave
			t_10 = CFD(raw_wfm, -1*p_amp, 0.1, pi, -1) #10% rise point
			t_90 = CFD(raw_wfm, -1*p_amp, 0.9, pi, -1) #90% rise point
			#if for some reason it cant find either of these points
			if(t_10 is None or t_90 is None):
				print("Had trouble finding 90% and 10% points")
				continue

			#if the rise is splitting an event
			#window edge. 
			if(t_10 > t_90):
				trise = (t_90 + max(raw_wfm.get_times())) - t_10
			else:
				trise = t_90 - t_10

			#sometimes happens if the 
			#particular threshold crossing point
			#is a bad place where slope is irrepresentative
			#of the pulse shape. 
			if(not(min(allowed_rise_times) <= trise <= max(allowed_rise_times))):
				continue

			tarrival = t_10

			#find the charge by finding the FORWARD 10% point
			#and integrating between backward t_10 and forward t_10
			t_10_forward = CFD(raw_wfm, -1*p_amp, 0.1, pi, +1)
			if(t_10_forward is None):
				print("Had trouble finding forward 10% point")
				continue

			#if the fall is splitting an event window edge, 
			#the integrate ranged function will handle it. 
			charge = raw_wfm.integrate_ranged([t_10, t_10_forward]) #mV*ns
			if(charge is None):
				continue

			if(not(min(allowed_widths) <= (t_10_forward - peak_time) <= max(allowed_widths))):
				#two things, either a bug or a wraparound issue. Quick temporarily
				#correct the wraparound issue. 
				if(t_10_forward < peak_time):
					temp10forward = t_10_forward + max(raw_wfm.get_times())/2.0
					temppeaktime = peak_time - max(raw_wfm.get_times())/2.0
					#check again. 
					if(not(min(allowed_widths) <= (temp10forward- temppeaktime) <= max(allowed_widths))):
						continue
				else:
					continue

			pulses.append(Pulse.Pulse(p_amp, peak_time, tarrival, trise, charge))

			#debugging
			"""
			fig, ax = plt.subplots()
			raw_wfm.plot(ax)
			ax.get_lines()[-1].set_linewidth(2)
			ax.get_lines()[-1].set_markersize(3)
			#nnls_wav.plot(ax, 'b-')
			ax.axvspan(t_10, t_10_forward, color='yellow', alpha=0.2)
			ax.axvspan(t_10, t_90, color='green', alpha=0.3)
			ax.set_ylabel("mV", fontsize=15)
			ax.set_xlabel("time (ns)", fontsize=15)
			ax.get_xaxis().set_tick_params(labelsize=15, length=14, width=2, which='major')
			ax.get_xaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
			ax.get_yaxis().set_tick_params(labelsize=15,length=14, width=2, which='major')
			ax.get_yaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
			majorLocatorX = MultipleLocator(4)
			minorLocatorX = MultipleLocator(1)
			ax.get_xaxis().set_major_locator(majorLocatorX)
			ax.get_xaxis().set_minor_locator(minorLocatorX)
			plt.show()
			"""
		return pulses



	#this function is used as a coarse analysis
	#of when the particle arrived at the detector. 
	#it uses found pulses and looks for the "primary"
	#pulse arrival. Presntly, the primary pulse
	#is defined as having the highest absolute charge. 
	#I tried using a different method that related
	#the soonest arriving pulse relative to the
	#trigger time as inferred by the blip in the SCAs
	#on the sync channels. It was highly inconsistent,
	#complicated, and tended not to work. 
	def get_primary_pulse_arrival(self):
		self.find_trigger_time()
		if(self.trigger_time_guess is None):
			print("lost an event due to inability to find trigger time ")
			return None

		#find pulse closest AND after the
		#trigger time. 
		arrivals = []
		charges = []
		for ch in self.pulses:
			for p in self.pulses[ch]:
				arrivals.append(p.get_arrival_time())
				charges.append(abs(p.get_charge()))

		highest_idx = charges.index(max(charges))
		return arrivals[highest_idx]

	
	#finds the largest amplitude pulse
	#using the self.pulses list and returns
	#the pulse object and the channel. 
	def get_largest_pulse_info(self):

		max_pulse = None
		max_amp = 0
		max_ch = None
		allowed_region = [0, 400] #mV, to reject weird spikes and positive upswings. 
		for ch in self.pulses:
			for p in self.pulses[ch]:
				if(not(min(allowed_region) < p.get_amplitude() < max(allowed_region))):
					continue
				if(max_pulse is None):
					max_pulse = p
					max_amp = p.get_amplitude()
					max_ch = ch 
					continue 

				thisamp = p.get_amplitude()
				if(np.abs(thisamp) > max_amp):
					max_amp = np.abs(thisamp)
					max_ch = ch 
					max_pulse = p 

		return max_pulse, max_ch


	#this function finds the primary pulse
	#in a similar fashion as in the transverse 
	#pulse profile function. It then looks at neighboring strips
	#and finds all side-to-side time differences. 
	#for each successfully reconstructed time difference, 
	#amplitudes are also reconstructed. 
	def get_longitudinal_data(self):
		#strips: strip number for the pulse pair
		#delays: time delay, positive means photon is closer to the right side
		#amps: amplitude of the prompt pulse
		#atts: attenuation, secondary amplitude/primary amplitude
		long_data = {'strips': [], 'delays': [], 'amps': [], 'atts': []} #properties of a pulse pair.
		maxpulse, maxch = self.get_largest_pulse_info()
		if(maxpulse is None):
			#this happens if no pulses in the event
			return long_data

		maxstrip = self.get_strip_number(maxch)

		#define a window
		#that is sure to contain the daughter pulse charges. 
		#This needs to be somewhat large ~ 1ns to account for
		#uncalibrated chip-to-chip constant timing offsets. 
		window = 3 #ns 
		center_time = maxpulse.get_peak_time() #because we are looking for other peaks. 
		strip_window = 1 #one strip on left and right of the primary strip to look at for longitudinal reconstruction. 

		allowed_amps = [0, 400] #mV allowed signals, to avoid spikes. 
		for ch in self.chs:
			if(ch in self.bad_chs or ch in self.syncs):
				continue

			this_strip_number = self.get_strip_number(ch)
			#if this strip has already been processed by the dual channel
			if(this_strip_number in long_data['strips']):
				continue

			#only look at strips within the "golden" window around the primary strip. 
			if(not(maxstrip - strip_window <= this_strip_number <= maxstrip + strip_window)):
				continue

			#if its single ended, use the autocorrelation
			#method on the nnls full fit wave to find time difference. 
			if(self.is_channel_single(ch)):
				#for the moment on these datasets, the ends of
				#single ended channels were not extended far enough, a simple
				#reconstruction will be biased towards pulses near the end point. 
				continue
				#trim the pulse around a larger window
				trimwindow = [center_time - 2*window, center_time + 2*window]
				nnls_wfm = self.nnlswavs[ch] #get the full nnls fit for noise reduction and negative constraint.
				raw_wfm = self.get_waveform(ch)
				if(nnls_wfm is None):
					continue

				corr_wfm = raw_wfm.trim_on_range_soft(trimwindow)
				#cross correlation with lag to find autocorrelation time difference. 
				corr, corr_dt = corr_wfm.lag_cross_correlate(corr_wfm, plot=True)
				#more to do, but debug here. 

				
			#otherwise it is double ended.
			else:
				#get the waveform and its strip-side pair.
				dual_channel = self.get_channel_pair(ch)
				#if for some reason it has no pair...
				if(dual_channel is None):
					continue

				#if there are no pulses on the channels
				if(not(ch in self.pulses)):
					continue
				if(not(dual_channel in self.pulses)):
					continue


				#loop through the pulse dictionary and see
				#if any exist in this time window and on these
				#channels. 
				acceptance_window = [center_time - window, center_time + window]
				pulses_in_range = {ch: [], dual_channel: []} #collect pulses on both sides. 

				#for some reason these cuts didn't make it in the last dataset. 
				allowed_rise_times = [0.5, 1.2] #if a pulse has rise-time outside this range, something went wrong.
				for p in self.pulses[ch]:
					if(min(acceptance_window) <= p.get_peak_time() <= max(acceptance_window) \
						and min(allowed_rise_times) <= p.get_rise_time() <= max(allowed_rise_times)):
						pulses_in_range[ch].append(p)
				for p in self.pulses[dual_channel]:
					if(min(acceptance_window) <= p.get_peak_time() <= max(acceptance_window) \
						and min(allowed_rise_times) <= p.get_rise_time() <= max(allowed_rise_times)):
						pulses_in_range[dual_channel].append(p)

				#if there is not at least one pulse on 
				#each side, move on. 
				if(len(pulses_in_range[ch]) == 0 or len(pulses_in_range[dual_channel]) == 0):
					continue
				#if there are more than one pulses in range (pretty rare for golden events)
				#take the biggest pulse
				if(len(pulses_in_range[ch]) > 1):
					pulses_in_range[ch] = sorted(pulses_in_range[ch], key=lambda x: abs(x.get_amplitude()))
					pulses_in_range[ch] = [pulses_in_range[ch][-1]] #take the highest amp only
				if(len(pulses_in_range[dual_channel]) > 1):
					pulses_in_range[dual_channel] = sorted(pulses_in_range[dual_channel], key=lambda x: abs(x.get_amplitude()))
					pulses_in_range[dual_channel] = [pulses_in_range[dual_channel][-1]] #take the highest amp only


				#at this point, the pulses_in_range dict has
				#a length 1 list for each channel. The pulse attributes
				#are valid and can be used to calculate the longitudinal data. 
				#this is, however, where one would implement adjustments due to 
				#chip constant offsets. 
				arrival_times = [pulses_in_range[ch][0].get_arrival_time(), pulses_in_range[dual_channel][0].get_arrival_time()]
				#if the arrival times split the DLL wraparound point (255 to 0), then 
				#forget about this event because that needs to be calibrated on a channel
				#to channel basis to avoid ~100s of picoseconds error ~5-10 mm 
				#at this level in the analysis, the pulses will have arrival times
				#that are off by greater than a strip width (really, close to an event window of ~40ns). 
				#factor of 2*striptime_uncalibrated is just to cover any errors in calibration of strip length. 
				striptime_uncalibrated = self.strip_lengths[this_strip_number] #ns
				if(abs(min(arrival_times) - max(arrival_times)) > 2*striptime_uncalibrated):
					continue


				primary_side = self.get_strip_side(ch) #1 for right, -1 for left. 
				#convention of positive time delay indicates closer to right side.
				#this may seem confusing, because lower values of arrival_times is earlier arrival.  
				time_delay = primary_side*(arrival_times[1] - arrival_times[0])

				#amplitude and amplitude attenuation
				amplitudes = {}
				for _ in pulses_in_range:
					if(self.get_strip_side(_) == 1):
						amplitudes['right'] = pulses_in_range[_][0].get_amplitude()
					else:
						amplitudes['left'] = pulses_in_range[_][0].get_amplitude()

				atten = amplitudes['right']/amplitudes['left'] #voltage attenuation right over left. 
				left_amp = amplitudes['left']


				if(not(min(allowed_amps) <= amplitudes['right'] <= max(allowed_amps) \
					and min(allowed_amps) <= amplitudes['left'] <= max(allowed_amps))):
					continue

				#add this component to the mix. 
				long_data['strips'].append(this_strip_number) #calculating for all neighboring strips. 
				long_data['delays'].append(time_delay) #photon is closer to right side if positive delay
				long_data['amps'].append(left_amp) #left side amp. 
				long_data['atts'].append(atten) #voltage attenuation

				
				#debugging
				"""
				fig, ax = plt.subplots(ncols = 2)
				self.get_waveform(ch).plot(ax[0])
				self.get_waveform(dual_channel).plot(ax[1])
				colors = ['r','g','b','c']
				for p in pulses_in_range[ch]:
					arr = p.get_arrival_time()
					rise = p.get_rise_time()
					peak_time = p.get_peak_time()
					peak = -1*p.get_amplitude()

					ax[0].axvspan(arr, arr+rise, color=colors[int(self.no)], alpha=0.2)
					ax[0].scatter([peak_time], [peak], color=colors[int(self.no)], s=5)
				for p in pulses_in_range[dual_channel]:
					arr = p.get_arrival_time()
					rise = p.get_rise_time()
					peak_time = p.get_peak_time()
					peak = -1*p.get_amplitude()

					ax[1].axvspan(arr, arr+rise, color=colors[int(self.no)], alpha=0.2)
					ax[1].scatter([peak_time], [peak], color=colors[int(self.no)], s=5)

				fig.suptitle("Strip: " + str(this_strip_number) + " , Delay: " + str(time_delay) + "\nAmp: " + str(left_amp) + " , Att: " + str(atten))

				plt.show()
				"""

				
		#at this point, long_data contains properties
		#of pulse pairs that are adjacent to the largest
		#amplitude pulse in the event. Or it contains none.
		#there shouldn't be massive dissagreement, but there
		#could be disagreement due to chip-to-chip offsets, so
		#I am keeping all of the data to be assessed as a global dataset. 


		return long_data




	
	#returns a transverse amplitude profile
	#of a photon based on the largest pulse in the event
	#and the amplitudes on all channels at times very close
	#to that main pulse arrival. 
	def get_transverse_pulse_profile(self):

		maxpulse, maxch = self.get_largest_pulse_info()
		if(maxpulse is None):
			#this happens if no pulses in the event
			return None

		#define a window
		#that is sure to contain the daughter pulse charges. 
		#This needs to be somewhat large ~ 1ns to account for
		#uncalibrated chip-to-chip constant timing offsets. 
		window = 1 #ns 
		center_time = maxpulse.get_peak_time() #because we are looking for other peaks. 

		#loop through all channels on both sides of the strip
		#as this one. If it is single ended, the largest pulse
		#always arrives on the side which is closest to the digitizing
		#channel, i.e. the side from get_strip_side. 
		maxside = self.get_strip_side(maxch)

		#centroid_dict[side][strip numbers, amplitudes for those strips]
		allowed_region = [-400, 0] #mV allowed signals, to avoid spikes. 
		transverse_dict = {1:{}, -1:{}}
		#look at both sides. 
		for side in transverse_dict:
			strip_numbers = []
			max_amps_in_window = [] 
			for ch in self.chs:
				#if its single ended, find its side
				#and look for a pair of pulses. 
				if(self.is_channel_single(ch)):
					#trim the pulse around a larger window
					trimwindow = [center_time - window, center_time + 3*window]
					raw_wfm = self.get_waveform(ch)
					trm_wfm = raw_wfm.trim_on_range_soft(trimwindow)
					#spline fit, we will find peaks. 
					trm_wfm = trm_wfm.get_spline_waveform(3, 40) #very smooth as the peakfinder will use low thresholds
					#find peaks
					thresh = 5 #mv, very low. 
					mindist = 9 #samples of the splined waveform
					peak_indices = trm_wfm.find_peaks_peakutils(thresh, mindist)
					if(len(peak_indices) == 0):
						#no peak here... 
						#just use the minimum voltage in the window. 
						if(min(allowed_region) < trm_wfm.find_min_sample()[0] < max(allowed_region)):
							strip_numbers.append(self.get_strip_number(ch))
							max_amps_in_window.append(trm_wfm.find_min_sample()[0])
						continue
					if(len(peak_indices) == 1):
						#use this as the peak. 
						if(min(allowed_region) < trm_wfm.get_signal()[peak_indices[0]] < max(allowed_region)):
							strip_numbers.append(self.get_strip_number(ch))
							max_amps_in_window.append(trm_wfm.get_signal()[peak_indices[0]])
						continue

					#otherwise, determine which peak to take based
					#on which side of the strip this digitizing channel is on.
					peak_indices = sorted(peak_indices) 
					if(self.get_strip_side(ch) == side):
						#use the first one chronologically
						if(min(allowed_region) < trm_wfm.get_signal()[peak_indices[0]] < max(allowed_region)):
							strip_numbers.append(self.get_strip_number(ch))
							max_amps_in_window.append(trm_wfm.get_signal()[peak_indices[0]])
						continue
					else:
						#use the peak immediately following the first one. 
						if(min(allowed_region) < trm_wfm.get_signal()[peak_indices[1]] < max(allowed_region)):
							strip_numbers.append(self.get_strip_number(ch))
							max_amps_in_window.append(trm_wfm.get_signal()[peak_indices[1]])
						continue
				#otherwise it is double ended.
				else:
					#only process this channel if it is on the same side
					if(self.get_strip_side(ch) != side):
						continue

					trimwindow = [center_time - window, center_time + window]
					raw_wfm = self.get_waveform(ch)
					#get the min sample of a spline fit waveform in this time. 
					trm_wfm = raw_wfm.trim_on_range_soft(trimwindow)
					#spline fit, we will find peaks. 
					trm_wfm = trm_wfm.get_spline_waveform(3, 40) #very smooth
					if(min(allowed_region) < trm_wfm.find_min_sample()[0] < max(allowed_region)):
						strip_numbers.append(self.get_strip_number(ch))
						max_amps_in_window.append(trm_wfm.find_min_sample()[0])

			#sort both lists simultaneously by strip number. 
			strip_numbers, max_amps_in_window = (list(t) for t in zip(*sorted(zip(strip_numbers, max_amps_in_window))))
			transverse_dict[side]['strips'] = strip_numbers
			transverse_dict[side]['amps'] = max_amps_in_window

		return transverse_dict





	#currently doesn't do well with edge bridged pulses. 
	def nnls_all(self):
		template_wfm = self.get_template_waveform()
		
		#append wraparound to end of event buffer
		#equal to 3 times the total width of the template waveform.
		wraparound_samples = 6*len(template_wfm.get_times())
		#the number of samples to re-write after the fit
		#has completed, when one re-wraps around on the
		#first few samples
		rewrap_samples = (int)(0.5*wraparound_samples)

		self.nnlswavs = {}
		self.components = {}

		#loop over all channels
		t0 = time.time()
		for ch in self.chs:
			if(ch in self.bad_chs):
				continue

			if(ch in self.syncs):
				continue

			testwfm = Waveform.Waveform(self.get_waveform(ch).get_signal(), self.get_waveform(ch).get_times())
			end_index = len(testwfm.get_times()) #the last sample index before adding wraparound. 
			testwfm.add_wraparound(wraparound_samples)

			testwfm_sig = testwfm.get_signal()
			tmpsig = template_wfm.get_signal()

			n = testwfm.get_num_samples()
			A = np.zeros((n, n))
			x = np.zeros(n)
			b = np.zeros(n)

		
			#b is the signal with noise
			for _ in range(len(b)):
				b[_] = testwfm_sig[_]

			#A is a matrix encoding the template
			for i in range(n):
				for j in range(n):
					if(j>=i and (j-i) < len(tmpsig)):
						A[j][i] = tmpsig[j-i]

			#run the nnls algorithm
			x, residual = optimize.nnls(A, b)

			new_sig = np.dot(A, x)
			fit_wfm = Waveform.Waveform(new_sig, testwfm.get_times())

			#now we need to remove wraparound
			fitsig = fit_wfm.get_signal()
			fittimes = fit_wfm.get_times()
			zerotime = fittimes[end_index]
			rewrap_fitsig = []
			rewrap_fittimes = []
			rewrap_x = []
			for i in range(end_index, end_index + rewrap_samples):
				rewrap_fitsig.append(fitsig[i])
				rewrap_fittimes.append(fittimes[i] - zerotime)
				rewrap_x.append(x[i])

			for i in range(rewrap_samples, end_index):
				rewrap_fitsig.append(fitsig[i])
				rewrap_fittimes.append(fittimes[i])
				rewrap_x.append(x[i])

			fit_wfm = Waveform.Waveform(rewrap_fitsig, rewrap_fittimes)
			x = rewrap_x




			self.nnlswavs[ch] = fit_wfm #dictionary



			#for some reason, some edges of
			#events have a BIG component waveform
			#that doesnt appear in the full fit_wfm.
			#this components handling also
			#doesnt yet deal with wraparound. 
			for i in range(len(x) - 1):
				if(x[i] < 1e-6):
					continue
				comptime = testwfm.get_times()[i]
				if(ch in self.components):
					self.components[ch].append([comptime, x[i]])
				else:
					self.components[ch] = [[comptime, x[i]]]

			
			
			#temporary for testing
			"""
			fig, ax = plt.subplots(figsize=(10, 7))
			testwfm.plot(ax)
			fit_wfm.plot(ax, 'b-')
			ax.get_lines()[-1].set_linewidth(2)

			for c in self.components[ch]:
				comptimes = [_+ c[0] for _  in template_wfm.get_times()]
				compsig = [_*c[1] for _ in template_wfm.get_signal()]
				compwfm = Waveform.Waveform(compsig, comptimes)
				compwfm.plot(ax, 'k')


			ax.set_ylabel("mV", fontsize=15)
			ax.set_xlabel("time (ns)", fontsize=15)
			ax.get_xaxis().set_tick_params(labelsize=15, length=14, width=2, which='major')
			ax.get_xaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
			ax.get_yaxis().set_tick_params(labelsize=15,length=14, width=2, which='major')
			ax.get_yaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
			majorLocatorX = MultipleLocator(4)
			minorLocatorX = MultipleLocator(1)
			ax.get_xaxis().set_major_locator(majorLocatorX)
			ax.get_xaxis().set_minor_locator(minorLocatorX)

			
			plt.show()
			"""
				
		print("Done in : " + str(time.time() - t0) + " seconds")






#----------------metadata parsers -------------------------_#

	#for old acdc-daq
	def get_trigger_time(self):
		#metadata is jarbled in these datasets....
		#possibly need dig_timestamp_*
		lo = self.metadata["timestamp_lo"] 
		mid = self.metadata["timestamp_mid"]
		hi = self.metadata["timestamp_hi"]

		return lo, mid, hi

	def get_readout_time(self):
		lo = self.metadata["dig_timestamp_lo"]
		mid = self.metadata["dig_timestamp_mid"]
		hi = self.metadata["dig_timestamp_hi"]

		return lo, mid, hi

""" for new ACDC software acdc-daq-revc
	def get_trigger_time(self):
		#metadata is jarbled in these datasets....
		#possibly need dig_timestamp_*
		lo = self.metadata["trig_time_lo"] 
		mid = self.metadata["trig_time_mid"]
		hi = self.metadata["trig_time_hi"]

		return lo, mid, hi

	def get_readout_time(self):
		lo = self.metadata["readout_time_lo"]
		mid = self.metadata["readout_time_mid"]
		hi = self.metadata["readout_time_hi"]

		return lo, mid, hi

"""

		
