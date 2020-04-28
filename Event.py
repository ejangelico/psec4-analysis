import Waveform
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from itertools import combinations
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import scipy.interpolate as inter 
import peakutils
import itertools
import sys



class Event:
	def __init__(self, boards, dataset_event_no):
		self.no = dataset_event_no
		self.boards = boards
		self.ignore_event = False

	def get_waveform(self, board, channel):
		for b in self.boards:
			if(b.get_board_no() == board):
				return b.get_waveform(channel)

		print("Couldnt find board " + str(board))
		return None

	def get_event_no(self):
		return self.no

	def set_endedness(self, board, end):
		for b in self.boards:
			if(b.get_board_no() == board):
				b.set_endedness(end)

	def ignore(self):
		return self.ignore_event

	def median_subtract(self):
		for b in self.boards:
			b.median_subtract()

	def get_boards(self):
		return self.boards

	def set_sync_chs(self, sync_chs):
		for b in self.boards:
			b.set_syncs(sync_chs)

	def set_bad_chs(self, bad_chs):
		for b in bad_chs:
			#b is board number with bad chs
			for bo in self.boards:
				if(bo.get_board_no() == b):
					bo.set_bad_chs(bad_chs[b])

	#this function is somewhat specific use, 
	#removing times and signals from a wave
	#as specified in nanoseconds in cut_dict. 
	#cut_dict[board] = list of time ranges [3, 6] for example
	#does so on all channels. 
	def cut_waveform_times(self, cut_dict):
		for bo in cut_dict:
			for b in self.boards:
				if(b.get_board_no() == bo):
					b.cut_waveform_times(cut_dict[bo])

	def remove_spikes(self):
		for b in self.boards:
			b.remove_spikes()


	def baseline_subtract(self, thresh=None):
		for b in self.boards:
			b.baseline_subtract(thresh)

	def nnls_all(self):
		for b in self.boards:
			b.nnls_all()


	def get_pulses(self):
		pulses = {} #indexed by board
		for b in self.boards:
			if(len(b.get_pulses()) != 0):
				pulses[b.get_board_no()] = b.get_pulses()

		return pulses

	def pulse_find_all(self):
		for b in self.boards:
			b.find_pulses()

	def two_sided_longitudinal_correlation(self):
		for b in self.boards:
			b.two_sided_longitudinal_correlation() #solves and saves position wfms into board object

	def get_sync_offsets(self):
		offsets = {} #dictionary indexed by board index.
		for b in self.boards:
			offsets[b.get_board_no()] = b.get_sync_wave_offsets()
		
		return offsets

	#bo temps is a dictionary with
	#template filenames.
	def set_templates(self, bo_temps):
		for bo in bo_temps:
			for b in self.boards:
				if(b.get_board_no() == bo):
					b.set_template_waveform(bo_temps[bo])


	def get_pulses(self):
		pulses = {} #indexed by board
		for b in self.boards:
			if(len(b.get_pulses()) != 0):
				pulses[b.get_board_no()] = b.get_pulses()

		return pulses

	def get_board_clocks(self):

		board_times = {}
		readout_times = {}

		for b in self.boards:
			no = b.get_board_no()
			lo, mid, hi = b.get_trigger_time()
			board_times[no] = lo + 65536*mid + 65536*65536*hi

			lo, mid, hi = b.get_readout_time()
			readout_times[no] = lo + 65536*mid + 65536*65536*hi

		return board_times, readout_times


	def get_board_time_differences(self):
		trigs, readouts = self.get_board_clocks()
		#a list of tuples representing pairs of boards. 
		#each pair will have an associated time difference. 
		board_pairs = [] 
		if(len(trigs) >= 2):
			bcombs = combinations(trigs.keys(), 2) #tuples, individually sorted by ascending board number
			for bc in bcombs:
				board_pairs.append(bc)
		else:
			if(len(trigs) == 1):
				for key in trigs:
					board_pairs.append(tuple([key])) #really just a singlet. 

		differences = {}
		for bp in board_pairs:
			if(len(bp) == 1):
				difference = trigs[bp[0]]
			else:
				difference = trigs[bp[1]] - trigs[bp[0]]

			if bp in differences.keys():
				differences[bp] = difference
			else:
				differences[bp] = difference


		return differences


	#returns which channels triggered on each
	#board. dict[board] = [channel numbers]
	def which_boards_triggered(self):
		bos = []
		for b in self.boards:
			bos.append(b.get_board_no())

		return bos

		

	#looks at the transverse pulse profile 
	#in amplitude space and calculates first
	#and second moments. 
	def get_transverse_moments(self):
		
		#transverse_moments[bo][side] = [centroid, second moment, total charge]
		transverse_moments = {}
		for b in self.boards:
			transverse_profile = b.get_transverse_pulse_profile()
			if(transverse_profile is None):
				#no pulse on this board
				continue

			#one for each side. 
			transverse_moments[b.get_board_no()] = {1: [], -1: []}
			for side in transverse_profile:
				nums = transverse_profile[side]['strips']
				amps = transverse_profile[side]['amps']

				#amp sum
				sumamp = np.sum(amps)
				#first moment, centroid, a one
				#dimensional vector in the transverse. 
				centroid = 0
				for i, strip in enumerate(nums):
					centroid += amps[i]*b.get_strip_position(strip)/sumamp

				#find second moment. 
				rms = 0
				for i, strip in enumerate(nums):
					rms += (amps[i]*(centroid - b.get_strip_position(strip))**2)/sumamp

				rms = np.sqrt(rms)

				transverse_moments[b.get_board_no()][side] = [centroid, rms, sumamp]

		return transverse_moments


	#The original moments method doesn't really reconstruct
	#the transverse peak very well, and overestimates the width. 
	#Instead here I use a spline fit to the transverse profile, 
	#find the main peak, and then find FWHM. special cases for endpoints. 
	#only accept the largest peak in the transverse spectrum
	#and the main strip NOT on an endpoint - i.e. golden events. Peak
	#finder wont find a peak if the main channel is on an endpoint. 
	def get_transverse_moments_from_spline(self):
		
		#transverse_moments[bo][side] = [centroid, second moment, total charge]
		transverse_moments = {}
		for b in self.boards:
			transverse_profile = b.get_transverse_pulse_profile()
			if(transverse_profile is None):
				#no pulse on this board
				continue

			#one for each side. 
			bo = b.get_board_no()

			transverse_moments[bo] = {1: [], -1: []}
			#fig, ax = plt.subplots(nrows = 2, figsize=(14, 10))
			for i, side in enumerate(transverse_moments[bo]):
				nums = transverse_profile[side]['strips']
				pos = [b.get_strip_position(_) for _ in nums]
				amps = transverse_profile[side]['amps']
				#put positions in order. 
				pos, amps = (list(t) for t in zip(*sorted(zip(pos, amps))))
				#required by spline fitter. happens very rarely. 
				if(len(amps) < 4 or len(pos) < 4 or len(pos) != len(amps)):
					transverse_moments[bo][side] = None
					continue
				#interpolate at a fine positional resolution
				pos_interval = 0.05 #mm
				s1 = inter.InterpolatedUnivariateSpline(pos, amps)
				newnums = np.arange(min(pos), max(pos), pos_interval)
				#use peakutils to find local minima
				#peakutils is easier to use in positive polarity
				invsig = [-1*_ for _ in s1(newnums)] 

				#threshold of peakutils is normalized
				#to % of the data *range*. But argument
				#is in mV. 
				threshold = 20 #mV
				mindist = 6.9*3 #mm
				data_range = max(invsig) - min(invsig)
				pu_thresh = (threshold - min(invsig))/(max(invsig) - min(invsig))

				 
				indices = []
				#there will be nothing above threshold
				#if the pu_thresh is greater than 1. 
				if(pu_thresh < 1):
					#find indices of peaks with this threshold. Absolute
					#value of the threshold, so finding peaks on either side of zero. 
					indices = peakutils.peak.indexes(np.array(invsig), thres=pu_thresh, min_dist=mindist)

				#this is the statement that
				#(1) the main strip is not an endpoint strip
				#(2) only one local minima was found. 
				#find the FWHM of the spline around this minimum. 
				#special case: one side is cut-off, then only use
				#one half and assume symmetry. 
				if(len(indices) == 1):
					hidex = indices[0]
					lowdex = indices[0]
				elif(len(indices) > 1):
					#use the biggest peak. 
					max_idx = None
					max_peakvalue = 0
					for _ in indices:
						if(max_idx is None):
							max_idx = _
							max_peakvalue = s1(newnums[_])
						if(abs(s1(newnums[_])) > abs(max_peakvalue)):
							max_peakvalue = s1(newnums[_])
							max_idx = _

					hidex = max_idx
					lowdex = max_idx

				else:
					transverse_moments[bo][side] = None	
					continue

				peakpos = newnums[hidex]
				peakval = s1(peakpos)
				lowcross = None
				hicross = None 
				while True:
					#most ideal break condition
					if(lowcross is not None and hicross is not None):
						break
					#other break condition, if one sided
					if(hidex == len(newnums) - 1 and lowdex == 0):
						break 

					lowval = s1(newnums[lowdex])
					hival = s1(newnums[hidex])
					if(abs(lowval) <= abs(0.5*peakval) and lowcross is None):
						#not interpolated, we are at 50 micron
						#interpolation resolution already. 
						lowcross = newnums[lowdex]
					if(abs(hival) <= abs(0.5*peakval) and hicross is None):
						hicross = newnums[hidex]


					if(hidex != len(newnums) - 1):
						hidex += 1
					if(lowdex != 0):
						lowdex -= 1


				#if both are none, reject event
				if(hicross is None and lowcross is None):
					fwhm = None

				#if one is none and the other is not
				elif(None in [hicross, lowcross]):
					#one sided FWHM assuming symmetry. 
					if(lowcross is None):
						fwhm = 2*np.abs(hicross - peakpos)
						#ax[i].axvspan(peakpos, hicross, color='m', alpha=0.3, label="fwhm = " + str(round(fwhm, 2)))
					else:
						fwhm = 2*np.abs(lowcross - peakpos)
						#ax[i].axvspan(lowcross, peakpos, color='m', alpha=0.3, label="fwhm = " + str(round(fwhm, 2)))
				#otherwise, both are found
				else:
					fwhm = np.abs(lowcross - hicross)
					#ax[i].axvspan(lowcross, hicross, color='m', alpha=0.3, label="fwhm = " + str(round(fwhm, 2)) + " mm")
					


				#calculate charge centroid in a region around the
				#primary transverse peak. 
				centroid_window = 2*fwhm # plus or minus this. 
				centroid = 0 
				ampsum = 0
				for x, psn in enumerate(pos):
					if(peakpos - centroid_window <= psn <= peakpos + centroid_window):
						centroid += amps[x]*psn
						ampsum += amps[x]
				centroid /= ampsum 

				transverse_moments[bo][side] = [peakpos, fwhm, centroid, abs(peakval), ampsum]

				#debugging
				"""
				ax[i].scatter(peakpos, peakval, marker='^', s=80, color='r', label="Spline peak = " + str(round(peakpos, 2)) + " mm")
				
				maxp, maxch = b.get_largest_pulse_info()
				maxstrip = b.get_strip_number(maxch)
				stripwidth = 5.13
				#ax[i].axvline(b.get_strip_position(maxstrip), color='b')
				ax[i].plot(newnums, s1(newnums), 'r-', linewidth=2)
				ax[i].errorbar(pos, amps, fmt = 'ko', xerr=[stripwidth/2.0]*len(pos))
				ax[i].scatter([centroid], [s1(centroid)], marker='d', s=80, color='g', label="Centroid = " + str(round(centroid, 2)) +  " mm")
				ax[i].set_xlim([-3, 103])
				ax[i].set_xlabel("transverse position (mm)", fontsize=16)
				ax[i].set_ylabel("strip amplitude (mV)", fontsize=16)
				ax[i].get_xaxis().set_tick_params(labelsize=15, length=14, width=2, which='major')
				ax[i].get_xaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
				ax[i].get_yaxis().set_tick_params(labelsize=15,length=14, width=2, which='major')
				ax[i].get_yaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
				majorLocatorY = MultipleLocator(10)
				minorLocatorY = MultipleLocator(5)
				xmajs = [i*6.9 + 6.9/2.0 for i in range(15)]
				xmins = [i*6.9/4 for i in range(60)]
				ax[i].set_xticks(xmins, minor=True)
				ax[i].set_xticks(xmajs)
				ax[i].get_yaxis().set_major_locator(majorLocatorY)
				ax[i].get_yaxis().set_minor_locator(minorLocatorY)
				ax[i].grid(True)
				ax[i].legend(fontsize=16)
				"""
				
				

			#plt.show()



				
		return transverse_moments



	#looks at the transverse pulse profile 
	#in amplitude space and calculates first
	#and second moments. 
	def get_transverse_profiles(self):
		#one for each side. 
		#transverse_moments[bo][side]['stips'] = [strip numbers]
		transverse_profiles = {}
		for b in self.boards:
			transverse_profile = b.get_transverse_pulse_profile()
			if(transverse_profile is None):
				#no pulse on this board
				continue

			bo = b.get_board_no()
			transverse_profiles[bo] = {}
			for side in transverse_profile:
				transverse_profiles[bo][side] = transverse_profile[side]


		return transverse_profiles



	def get_longitudinal_data(self):
		result_dict = {}
		for b in self.boards:
			long_data = b.get_longitudinal_data()
			#if there are no pulse pairs that pass cuts in
			#the Board function, then continue. 
			if(len(long_data['strips']) == 0):
				continue

			bo = b.get_board_no()
			result_dict[bo] = long_data

		return result_dict
			

	def get_all_samples(self):
		thedict = {}
		for b in self.boards:
			bnum = b.get_board_no()
			thedict[bnum] = b.get_all_samples() #dictionary indexed by channel

		return thedict



	#if ended is true, it does each board separately and compairs its pair. 
	def plot_waveforms_separated(self, ended=False):
		if(ended == False):
			fig, ax = plt.subplots(figsize=(13,7),ncols=6, nrows=5)
			gs1 = gridspec.GridSpec(6, 5)
			gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

			fig.suptitle("Event " + str(self.no))
			for b in self.boards:
				b.plot_board_separated(ax)


			plt.show()
		else:
			fig, ax = plt.subplots(figsize=(12, 6), ncols=5, nrows=3)
			gs1 = gridspec.GridSpec(4, 3)
			gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
			for b in self.boards:
				b.plot_board_ended(ax = ax)

			#print board difference
			differences = self.get_board_time_differences()
		
			fig.suptitle("Event " + str(self.no))

		plt.show()


	def plot_sync_channels(self, ax = None):
		if(ax is None):
			fig, ax = plt.subplots(ncols = len(self.boards), figsize=(25, 7))
		for i, b in enumerate(self.boards):
			b.plot_sync_channels(ax[i])


	def plot_heatmaps(self):
		fig, ax = plt.subplots(figsize=(15, 8), ncols = 2, nrows = 2)
		gs1 = gridspec.GridSpec(2, 2)
		gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

		ax_1d = []
		for _ in ax:
			for i in range(len(_)):
				ax_1d.append(_[i])

		for i, bo in enumerate(self.boards):
			#requires that all channels have the
			#same time values 
			x = []
			y = []
			z=[]
			channels = bo.get_chs()
			wfms = bo.get_wfms()
			for j, ch in enumerate(channels):
				wfm = wfms[j]
				times = wfm.get_times()
				sig = wfm.get_signal()

				for _, t in enumerate(times):
					voltage = sig[_]
					x.append(t)
					y.append(ch)
					z.append(voltage)
				

			xbins = times
			ybins = channels
			h = ax_1d[i].hist2d(x, y, bins=[xbins, ybins], weights=z, cmap=plt.inferno(), cmin=-200, cmax=30)
			cb = plt.colorbar(h[3],ax=ax_1d[i])
		
		#plt.show()



	#find synchronization signals
	#and store their information in
	#board attributes
	def find_sync_waves(self):
		for b in self.boards:
			b.find_sync_waves()


	def get_calibration_data(self, good_times):
		result_dict = {}
	

		for b in self.boards:
			bo = b.get_board_no()
			individ_results = b.get_calibration_data(good_times[bo])
			if(not(bo in result_dict.keys())):
				result_dict[bo] = {}

			for ch in individ_results:
				if(not(ch in result_dict[bo].keys())):
					result_dict[bo][ch] = {'resids': {}, 'deriv': {}, 'fits': {}, 'popts': []}

				for idx in individ_results[ch]['resids']:

					result_dict[bo][ch]['resids'][idx] = individ_results[ch]['resids'][idx]
					result_dict[bo][ch]['deriv'][idx] = individ_results[ch]['deriv'][idx]
					result_dict[bo][ch]['fits'][idx] = individ_results[ch]['fits'][idx]
					result_dict[bo][ch]['popts'] = individ_results[ch]['popts']

		return result_dict

