import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, correlate
import scipy.interpolate as inter 
import scipy
from scipy.optimize import curve_fit
from collections import deque
from mpl_toolkits.axes_grid1 import make_axes_locatable
import lmfit
import peakutils
from scipy import integrate

import sys

class Waveform:
	#Takes array of numbers that is the waveform, and the time step between each number
	#Give timestep in nanoseconds and waveform in millivolts
	def __init__(self, sg, times, strip_position=None, strip_transit_time=None):
		self.signal = sg #should be a list of voltages
		self.times = times
		self.timestep = min([times[i+1] - times[i] for i in range(len(times)) if i+1 < len(times)])

		self.num_samples = len(self.signal)

		#the position of the stripline in the transverse coordinate
		self.pos = strip_position 
		#the effective length (in nanoseconds)
		#of the channel including the board trace
		#lengths given an input wave with similar
		#risetime as an MCP pulse
		self.transit_time = strip_transit_time #in nanoseconds
	
	#Adds two waveforms together. Uses linear interpolation to align the time indices.
	#returns a waveform with an exclusive intersection of the two waveforms' time indices. 
	#i.e. the times will be the minimum range between the two summands. 
	def __add__(self, wf):

		#dont do complicated linear interpolation if you dont have to
		if(list(self.times) == list(wf.get_times())):
			thatsig = wf.get_signal()
			sumsig = [self.signal[i] + thatsig[i] for i in range(len(self.times))]
			return Waveform(sumsig, wf.get_times())

		#takes the times of the largest wave. 
		if(len(self.times) > len(wf.get_times())):
			times = self.times
		else:
			times = wf.get_times()

		other_wfm_sigs = [wf.get_signal_at(t) for t in times]
		these_wfm_sigs = [self.get_signal_at(t) for t in times]
		newtimes = [] #exclusive union of two waveform sets. 
		sumsig = []
		for i in range(len(times)):
			#happens if the interpolator was out of time range
			if(other_wfm_sigs[i] is None or these_wfm_sigs[i] is None):
				continue
			else:
				sumsig.append(these_wfm_sigs[i]+other_wfm_sigs[i])
				newtimes.append(times[i])

		if(len(newtimes) <= 1):
			return None

		return Waveform(sumsig, newtimes)
				
		
	#Multiplies a waveform by the number num. Allows for scaling. Num must be second
	def __mul__(self, num):
		newsig = [x*num for x in self.signal]
		return Waveform(newsig, self.times)

	
	#adds nsamples of samples from the 
	#beginning of the wave to the end of the
	#wave. Utitility for functions that
	#have edge effects
	def add_wraparound(self, nsamples):
		for i in range(nsamples):
			self.signal.append(self.signal[i])
			self.times.append(self.times[-1]+self.timestep)

	#does the same but adds wraparound to the beginning
	#of the buffer from the end. 
	def add_wraparound_beginning(self, nsamples):
		for i in range(nsamples):
			self.signal.insert(0, self.signal[-1*(i+1)])
			self.times.insert(0, self.times[0] - self.timestep)

	#shift the waveform by n samples 
	#No touching the times, so it really 
	#is not respecting the true sample timing
	#of each SCA sample. 
	def circular_shift_time(self, n):
		#rotate signals only (SCA caps don't)
		#have a 1-to-1 map with sample times yet. 
		deqsig = deque(self.signal)
		#a collections.deque method
		deqsig.rotate(-1*n)
		self.signal = list(deqsig)


	def pop_first_sample(self):
		self.signal.pop(0)
		self.times.pop(0)


	#Returns the number of samples in a pulse
	def get_num_samples(self):
		return len(self.signal)
	
	#Returns a soft copy of the signal
	def get_signal(self):
		ts = [_ for _ in self.signal]
		return ts

	#get the signal at any continuous
	#time variable, "time", by interpolation
	def get_signal_at(self, time):
		if(not(self.times[0] <= time <= self.times[-1])):
			return None

		#closest index to that time. 
		time_idx = (np.abs(np.asarray(self.times) - time)).argmin()

		#if exactly this time exists in the waveform
		if(self.times[time_idx] == time):
			return self.signal[time_idx]
		else:
			#decide whether to interpolate using
			#higher sample and lower sample. This
			#automatically handles edge of list effects
			if(self.times[time_idx] < time):
				x0 = self.times[time_idx]
				x1 = self.times[time_idx + 1]
				y0 = self.signal[time_idx]
				y1 = self.signal[time_idx + 1]
			else:
				x0 = self.times[time_idx - 1]
				x1 = self.times[time_idx]
				y0 = self.signal[time_idx - 1]
				y1 = self.signal[time_idx]

			m = (y1 - y0)/(x1 - x0)
			#if slope zero, just
			#return the sample value
			if(m  < 1e-6):
				return y0

			b = y1 - m*x1

			return (m*time + b)



	#find sample number closest to the input time
	def get_sample_at(self, time):
		tarr = np.asarray(self.times)
		idx = (np.abs(tarr - time)).argmin()
		return idx


	def get_timestep(self):
		return self.timestep

	def get_times(self):
		ts = [_ for _ in self.times]
		return ts

	#can synthetically add a sample. 
	#useful for forming templates. 
	def append_sample(self, sig, time):
		self.signal.append(sig)
		self.times.append(time)

	#shift waveform such that the 
	#first sample is at time 0
	def zero_times(self):
		zt = self.times[0]
		self.times = [_ - zt for _ in self.times]

		
	def find_max_above_thresh(self, thresh):
		time_over_thresh = 5 #5 samples = 500ps above threshold is minimum 
		s0 = None #sample that breaks threshold
		maxes = [] #for each region that goes above thresh
		tempmax = 0
		maxtime = None
		#built only for negative going pulses
		for i, v in enumerate(self.signal):
			if(v <= -1*np.abs(thresh)):
				if(s0 is None):
					s0 = 1
				else:
					s0 += 1

				if(v < tempmax):
					tempmax = v
					maxtime = self.times[i]

			else:
				#never went above threshold
				if(s0 is None):
					continue
				#just fell below threshold
				#and not enough time over. 
				#reset all variables
				if(s0 < time_over_thresh):
					s0 = None
					tempmax = 0
					maxtime = None
					continue
				#good region, save max and time
				if(s0 >= time_over_thresh):
					maxes.append([tempmax, maxtime])
					s0 = None
					tempmax = 0
					maxtime = None
					continue

		#at the end if it was still above threshold
		if(maxtime is not None):
			maxes.append([tempmax, maxtime])

		return maxes


	
	
	#this function is an attempt to 
	#ignore samples that are outside of 
	#the 4096 ADC counts of the ADC. This
	#is some error in firmware or in the chip
	def remove_spikes(self):
		difference_threshold = 100 #not allowed to have 400 mv change in one sample
		hi = 0
		lo = 0
		newsig = []
		modified_sig = None
		for i, s in enumerate(self.signal):
			if(modified_sig is None):
				sigdif = abs(s - self.signal[i-1]) #even good for i = 0
			else:
				sigdif = abs(s - modified_sig)

			if(sigdif > difference_threshold):
				if(modified_sig is None):
					modified_sig = self.signal[i-1]
					newsig.append(self.signal[i-1])
				else:
					newsig.append(modified_sig)

			else:
				newsig.append(s)
				modified_sig = None

		#debugging
		"""
		if(newsig != self.signal):
			fig, ax = plt.subplots()
			self.plot(ax)
			ax.scatter(self.times, newsig, color='black', s=1)
			ax.plot(self.times, newsig, color='black', linewidth=0.5)
			plt.show()
		"""

		self.signal = newsig
		return


	#Finds the sample with the largest absolute value
	#if multiples, just returns one of them. 
	def find_absmax_sample(self):
		abs_sig = [abs(_) for _ in self.signal]
		absmax_sig = max(abs_sig)
		idx = abs_sig.index(absmax_sig)
		return absmax_sig, idx

	#find minimum of waveform and index of this minimum
	#if multiple of the same voltage, just return one of them	
	def find_min_sample(self):
		min_sig = min(self.signal)
		idx = list(self.signal).index(min_sig)
		return min_sig, idx

	#find maximum of waveform and index of this maximum
	#if multiple of the same voltage, just return one of them	
	def find_max_sample(self):
		max_sig = max(self.signal)
		idx = list(self.signal).index(max_sig)
		return max_sig, idx


	#trims waveform on a time range
	def trim_on_range(self, ran):
		newsig = []
		newtimes = []
		for i, t in enumerate(self.times):
			if(min(ran) <= t <= max(ran)):
				newsig.append(self.signal[i])
				newtimes.append(t)

		self.times = newtimes
		self.signal = newsig

	#trims waveform on a time range
	def trim_on_range_soft(self, ran):
		newsig = []
		newtimes = []
		for i, t in enumerate(self.times):
			if(min(ran) <= t <= max(ran)):
				newsig.append(self.signal[i])
				newtimes.append(t)

		return Waveform(newsig, newtimes)

	def trim_range_out(self, ran, wfm=True):
		newsig = []
		newtimes = []
		for i, t in enumerate(self.times):
			if(min(ran) <= t <= max(ran)):
				continue
			else:
				newsig.append(self.signal[i])
				newtimes.append(t)

		if(wfm == True):
			return Waveform(newsig, newtimes)
		else:
			self.signal = newsig
			self.times = newtimes
			self.num_samples = len(self.times)
		

	#shift all samples by value a
	def shift_waveform(self, a):
		self.signal = [_+a for _ in self.signal]

	#shift all times by time a
	def time_shift(self, a):
		self.times = [_+a for _ in self.times]

	def get_time_shifted_waveform(self, a):
		newtimes = [_+a for _ in self.times]
		return Waveform(self.signal, newtimes)


	#integrate over some range. return
	#in units of mV*ns. range is times in ns
	def integrate_ranged(self, rang=None):
		
		#trim on range on a new waveform
		trm_wave = Waveform(self.get_signal(), self.get_times())

		if(rang is None):
			pass
		else:
			#if the first value of range is greater than
			#the second value of range, we need to circulate the
			#buffer such that the times are in order, i.e. the range
			#doesnt split an event buffer window. 
			if(rang[0] > rang[1]):
				#fig, ax = plt.subplots()
				#trm_wave.plot(ax)
				#ax.axvspan(rang[0], rang[1], alpha=0.2, color='black', label=str(rang))
				trm_wave.circular_shift_time(int(len(self.times)/2.0)) #half an event window
				rang[0] = rang[0] - self.timestep*(int(len(self.times)/2.0))
				rang[1] = rang[1] + self.timestep*(int(len(self.times)/2.0))
				#trm_wave.plot(ax)
				#ax.axvspan(rang[0], rang[1], alpha=0.4, label=str(rang))
				#ax.legend()
				#plt.show()



			trm_wave.trim_on_range(rang)
			if(len(trm_wave.get_signal()) < 4 or len(trm_wave.get_times()) < 4 ):
				#some crazy event
				return None


		tr_int = np.trapz(np.array(trm_wave.get_signal()), np.array(trm_wave.get_times()))
		return tr_int



	
	#Computes the power spectral density of the waveform
	def get_power_spectral_density(self):
		sampling_freq = 1.0/self.timestep
		freqs, psd = scipy.signal.welch(np.array(self.signal), sampling_freq)
		return psd, freqs

	#smoothing is a least squares error, so 0.1 allows
	#less error in the fit than 1.5. ranges from 0 to inf
	#sample_factor is the multiplier of the number of samples
	#for the new sample times in the spline waveform
	def get_spline_waveform(self, sample_factor, smoothing = None):
		if(smoothing is None):
			s1 = inter.InterpolatedUnivariateSpline(self.times, self.signal)
		else:
			s1 = inter.UnivariateSpline(self.times, self.signal, s=smoothing)

		new_times = np.linspace(min(self.times), max(self.times), self.get_num_samples()*sample_factor)
		return Waveform(s1(new_times), new_times)


	#range can define a time range
	def median_subtract(self, rang = None):
		if(rang is None):
			med = np.median(self.signal)
			self.signal = [_ - med for _ in self.signal]

		else:
			med = np.median(self.signal[self.get_sample_at(min(rang)):self.get_sample_at(max(rang))])
			self.signal = [_ - med for _ in self.signal]




	#returns butterworth bandpass
	#all frequencies in Hz
	def get_butterworthed_signal(self, lowcut, highcut, filter_order=5):
		#buttFreq assumed in GHz
		nyq = 0.5/(self.timestep) #in "GHz"
		low = lowcut/nyq
		high = highcut/nyq
		b, a = butter(filter_order,[low, high], btype='band')
		hfilt = lfilter(b, a, self.signal)
		return hfilt

	#modifies the signal by filtering with a butterworth
	def filter_signal_butterworth(self, lowcut, highcut, filter_order=5):
		self.signal = self.get_butterworthed_signal(lowcut, highcut, filter_order)

	#Plots the pulse
	def plot(self, ax = None, fmt = None, label= None):
		if(ax is None):
			fig, ax = plt.subplots()

		if(fmt is None):
			ax.plot(self.times, self.signal, 'o-', markersize=1, linewidth = 0.5)
		else:
			ax.plot(self.times, self.signal, fmt, markersize=1, linewidth = 0.5)

		if(label is not None):
			ax.get_lines()[-1].set_label(label)
		#ax.set_xlabel("time (ns)")
		#ax.set_ylabel("signal waveform (mV)")
		return ax


	#Plots the pulse and spline fit
	def plot_spline_waveform(self, ax):
		if(ax is None):
			fig, ax = plt.subplots()

		smoothing = 10000
		sample_multiplier = 10
		spline_wave, spline_times = self.get_spline_waveform(sample_multiplier, smoothing)
		ax.plot(spline_times, spline_wave, label="spline fit")
		ax.set_xlabel("time (ns)")
		ax.set_ylabel("signal waveform (mV)")
		return ax

	#Plots the pulse with a butterworth filter on it
	def plot_filtered_butterworth(self, lowcut, highcut, filter_order, ax):
		if(ax is None):
			fig, ax = plt.subplots()
		filtwave = self.get_butterworthed_signal(lowcut, highcut, filter_order)
		ax.plot(self.times, filtwave, label="butterworth filtered")
		ax.set_xlabel("time (ns)")
		ax.set_ylabel("signal waveform (mV)")
		return ax


	#uses the peakutils function set
	#to find peaks. Threshold is in mV
	#and is assuming a negative value (but
	#should be positive. it is going to be
	#absolute valued anyway.) mindist in units
	#of samples
	def find_peaks_peakutils(self, thresh, mindist, plot=False):

		#peakutils is easier to use in positive polarity
		invsig = [-1*_ for _ in self.signal] 

		#threshold of peakutils is normalized
		#to % of the data *range*. But argument
		#is in mV. 
		data_range = max(invsig) - min(invsig)
		pu_thresh = (thresh - min(invsig))/(max(invsig) - min(invsig))

		#if the threshold is greater than 
		#the datarange. Empty list, no
		#indices.  
		if(pu_thresh > 1):
			return []

		#find indices of peaks with this threshold. Absolute
		#value of the threshold, so finding peaks on either side of zero. 
		indices = peakutils.peak.indexes(np.array(invsig), thres=pu_thresh, min_dist=mindist)
		if(plot and len(indices) != 0):
			fig, ax = plt.subplots()
			self.plot(ax)
			for i in indices:
				ax.axvline(self.times[i])
			plt.show()


		return list(indices)






	#returns the phase, amplitude, and frequency
	#of the signal assuming it is a sin wave.
	#returns:
	#fit params
	#residual over the range
	#sqrt of diagonal of covariance matrix ("err")
	def get_sin_wave_information(self, time_range = None,  ampguess = None, plot=False):
		#I know the frequency will be 200MHz so just force that
		freq_guess = 200.0 #MHz
		offset_guess = np.mean(self.signal)
		amp_guess = (np.abs(max(self.signal))) - offset_guess
		phase_guess = 0.5 #pretty insensitive

		"""
		print("timestep: " + str(self.timestep))
		print("freq guess: " + str(freq_guess))
		print("offset guess: " + str(offset_guess))
		print("amplitude guess: " + str(amp_guess))
		print("phase guess : " + str(phase_guess))
		"""


		def sin_fitfunc(x, amp, freq, phas, off):
			return (amp*np.sin(x*freq*2*np.pi - phas) + off)


		#time_range defines a region to do the fit over.
		if(time_range is None):
			#do the whole range
			newtimes = self.times
			newsignal = self.signal
		else:
			newtimes = [_ for _ in self.times if min(time_range) < _ < max(time_range)]
			newsignal = [self.signal[i] for i in range(len(self.times)) if min(time_range) < self.times[i] < max(time_range)]
			if(len(newtimes) == 0):
				return None, None, None

		try:
			popt, pcov = scipy.optimize.curve_fit(sin_fitfunc, np.array(newtimes), np.array(newsignal), p0=[amp_guess, freq_guess/1000.0, phase_guess, offset_guess])
		except RuntimeError:
			print("curve-fit failed on a sine wave")
			"""
			fig, ax = plt.subplots()
			ax.plot(self.times, self.signal, '-o', markersize=2)
			ax.plot(newtimes, newsignal, 'r-', linewidth=3)
			plt.show()
			"""
			return None, None, None

		if(popt[0] < 0):
			popt[0] = -1*(popt[0])
			popt[2] = popt[2] + np.pi
			popt[2] = popt[2] % (2*np.pi)
		#calculate the residual
		resid = self.get_sin_residuals(popt, time_range)

		if(plot):
			fig, ax = plt.subplots()
			ax.plot(self.times, self.signal, '-o', markersize=2)
			ax.plot(self.times, sin_fitfunc(np.array(self.times), *popt), label="a=%5.3f,f=%5.7f,p=%5.5f,o=%5.3f" % tuple(popt))
			divider = make_axes_locatable(ax)
			ax2 = divider.append_axes("bottom", size="20%", pad=0)
			ax2.plot(newtimes, resid, 'k')
			ax2.set_xlim([0, max(self.times)])
			ax.set_xlim([0, max(self.times)])
			print("Error is " + str(np.sqrt(np.diag(pcov))))
			ax.legend()
			plt.show()


	
		return popt, resid, np.sqrt(np.diag(pcov))





	def get_sin_residuals(self, popt, time_range):
		#time_range defines a region to do the residual calculation over
		if(time_range is None):
			#do the whole range
			newtimes = self.times
			newsignal = self.signal
		else:
			newtimes = [_ for _ in self.times if min(time_range) < _ < max(time_range)]
			newsignal = [self.signal[i] for i in range(len(self.times)) if min(time_range) < self.times[i] < max(time_range)]


		def sin_fitfunc(x, amp, freq, phas, off):
			return (amp*np.sin(x*freq*2*np.pi - phas) + off)

		res = []
		for i, t in enumerate(newtimes):
			res.append(sin_fitfunc(t, *popt) - newsignal[i])

		return res



	#cross correlate the self with cwfm
	def lag_cross_correlate(self, cwfm, plot=False):

		#correlation relies on the two waveforms having the
		#same timing and same number of samples. re-sample
		#such that the largest time window is included and 
		#the smallest sampling size. 

		#special case when both quantities are the same

		corr = correlate(self.signal, cwfm.get_signal())

		#lag = np.argmax(correlate(self.signal, cwfm.get_signal()))
		#newsig = np.roll(cwfm.get_signal(), shift=int(np.ceil(lag)))


		if(plot):
			fig, (ax1, ax2) = plt.subplots(ncols=2)
			self.plot(ax1)
			cwfm.plot(ax1)
			#ax2.plot(self.times, newsig)
			#self.plot(ax2)
			ax2.plot(corr)
			plt.show()
		
		return corr, self.timestep










