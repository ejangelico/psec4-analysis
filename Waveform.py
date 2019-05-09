import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import scipy.interpolate as inter 
import scipy

import sys

class Waveform:
	#Takes array of numbers that is the waveform, and the time step between each number
	#Give timestep in nanoseconds and waveform in millivolts
	def __init__(self, sg, times, strip_position=None, strip_transit_time=None):
		self.signal = sg #should be a list of voltages
		self.times = times
		self.timestep = min([times[i+1] - times[i] for i in xrange(len(times)) if i+1 < len(times)])
		self.num_samples = len(self.signal)

		#for filling later after an analysis
		#function is called. Can be any analysis, 
		#tot or otherwise
		self.pulse_list = []

		#the position of the stripline in the transverse coordinate
		self.pos = strip_position 
		#the effective length (in nanoseconds)
		#of the channel including the board trace
		#lengths given an input wave with similar
		#risetime as an MCP pulse
		self.transit_time = strip_transit_time #in nanoseconds
	
	#Adds two waveforms together. Checks for same number of samples. If one is zero, returns other pulse
	def __add__(self, wf):
		if self.num_samples == 0:
			return wf
		elif wf.get_num_samples() == 0:
			return self
		elif self.get_num_samples() != wf.get_num_samples():
			print "Cannot add two pulses with different numbers of points."
			print "Waveform lhs number of samples:", self.get_num_samples()
			print "Waveform rhs number of samples:", wf.get_num_samples()
			print "Returning lhs waveform"
			return self
		else:
			wfsig = wf.get_signal()
			waveform_sum = [self.signal[i]+wfsig[i] for i in range(self.num_samples)]
			return Waveform(waveform_sum, self.times)
				
		
	#Multiplies a waveform by the number num. Allows for scaling. Num must be second
	def __mul__(self, num):
		new_signal = [x*num for x in self.signal]
		return Waveform(new_signal, self.times)
	
	#Returns the number of samples in a pulse
	def get_num_samples(self):
		return self.num_samples
	
	#Returns waveform array of the pulse
	def get_signal(self):
		return self.signal

	def get_timestep(self):
		return self.timestep

	def get_times(self):
		return self.times
	
	
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
		idx = self.signal.index(min_sig)
		return min_sig, idx

	#find maximum of waveform and index of this maximum
	#if multiple of the same voltage, just return one of them	
	def find_max_sample(self):
		max_sig = max(self.signal)
		idx = self.signal.index(max_sig)
		return max_sig, idx


	#trims waveform on a time range
	def trim_on_range(self, ran):
		newsig = []
		newtimes = []
		for i, t in enumerate(self.times):
			if(min(ran) <= t <= max(ran)):
				newsig.append(self.signal[i])
				newtimes.append(t)

		return newsig, newtimes



	#Computes the integral of the square of the waveform
	def get_squared_integral(self):
		tot = 0
		for x in self.signal:
			tot += x**2
		tot *= self.timestep
		return tot #units of mV*ns
	
	#Computes the power spectral density of the waveform
	def get_power_spectral_density(self):
		sampling_freq = 1.0/(self.timestep*10**(-9))
		freqs, psd = scipy.signal.welch(np.array(self.signal), sampling_freq)
		freqs_GHz = [_/(10**9) for _ in freqs]
		psd_GHz = [_*np.sqrt(10**9) for _ in psd]
		return psd_GHz, freqs_GHz

	#smoothing is a least squares error, so 0.1 allows
	#less error in the fit than 1.5. ranges from 0 to inf
	#sample_factor is the multiplier of the number of samples
	#for the new sample times in the spline waveform
	def get_spline_waveform(self, sample_factor, smoothing = None):
		if(smoothing is None):
			s1 = inter.InterpolatedUnivariateSpline(self.times, self.signal)
		else:
			s1 = inter.UnivariateSpline(self.times, self.signal, s=smoothing)

		new_times = np.linspace(min(self.times), max(self.times), self.num_samples*sample_factor)
		return s1(new_times), new_times


	#returns a waveform with a DC offset removed
	def get_signal_baseline_subtracted(self):
		#make dc offset simple the median of
		#the remaining samples
		dcoff = np.median(self.signal)
		shifted_signal = [_ - dcoff for _ in self.signal]
		return shifted_signal

	#modifies the signal by subtracting the baseline
	def baseline_subtract(self):
		self.signal = get_signal_baseline_subtracted()

	#returns the waveform with a butterworth low pass
	#filter applied
	def get_butterworthed_signal(self, butt_freq, filter_order):
		#buttFreq assumed in GHz
		nyq = np.pi/(self.timestep) #for the scipy package, they define nyquist this way in rads/sec
		rad_buttFreq = (np.pi/180.0)*butt_freq #now in radian units
		b, a = butter(filter_order,rad_buttFreq/nyq)
		hfilt = lfilter(b, a, self.signal)
		return hfilt

	#modifies the signal by filtering with a butterworth
	def filter_signal_butterworth(self, butt_freq, filter_order):
		self.signal = get_butterworthed_signal(butt_freq, filter_order)

	#Plots the pulse
	def plot(self, ax = None):
		if(ax is None):
			fig, ax = plt.subplots()

		ax.plot(self.times, self.signal, label="raw signal")
		ax.set_xlabel("time (ns)")
		ax.set_ylabel("signal waveform (mV)")
		return ax

	#plot waveform with colors for pulses that 
	#have been found by analysis
	def plot_found_pulses(self, ax = None):
		if(ax is None):
			fig, ax = plt.subplots()

		for pulse in self.pulse_list:
			#find indices of closest values
			ts = np.asarray(self.times)
			idxmin = (np.abs(ts - min(pulse))).argmin()
			idxmax = (np.abs(ts - max(pulse))).argmin()
			ax.plot(self.times[idxmin:idxmax], self.signal[idxmin:idxmax], 'ro-')

		ax.plot(self.times, self.signal, label="raw signal")
		ax.set_xlabel("time (ns)")
		ax.set_ylabel("signal waveform (mV)")
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
	def plot_filtered_butterworth(self, butt_freq, filter_order, ax):
		if(ax is None):
			fig, ax = plt.subplots()
		filtwave = self.get_butterworthed_signal(butt_freq, filter_order)
		ax.plot(self.times, filtwave, label="butterworth filtered")
		ax.set_xlabel("time (ns)")
		ax.set_ylabel("signal waveform (mV)")
		return ax

	#loops inside the guess_range of times (ns) and finds
	#sample closes to threshold voltage. Then interpolates linearly
	#to find the time of threshold crossing
	def get_interpolated_time_at_threshold(self, thresh, guess_range, direc):
		trim_sig, trim_times = self.trim_on_range(guess_range) 

		if(direc == 'falling'):
			drv = -1
		elif(direc == "rising"):
			drv = +1
		else:
			#assume rising
			drv = +1

		#number of samples to look about either side of index
		#to find average derivative
		ave_drv_len = 2 

		for i, s in enumerate(trim_sig):
			if(s >= thresh and drv == 1):
				#check derivative
				trg_drv = self.check_ave_derivative(trim_sig, trim_times, i, ave_drv_len)
				if(np.sign(trg_drv) == drv):
					sigtime = self.linear_interpolation(thresh, trim_sig, trim_times, i)
					return sigtime
			elif(s <= thresh and drv == -1):
				#check derivative
				trg_drv = self.check_ave_derivative(trim_sig, trim_times, i, ave_drv_len)
				if(np.sign(trg_drv) == drv):
					sigtime = self.linear_interpolation(thresh, trim_sig, trim_times, i)
					return sigtime
			else:
				continue

		return None


	def find_pulses_tot(self, sig_thresh, pulse_loc):
		if(pulse_loc is None):
			#search entire range
			trim_sig = self.signal
			trim_times = self.times
		else:
			trim_sig, trim_times = self.trim_on_range(pulse_loc)

		#if thresh is negative, a 'falling' threshold
		#is default. If positive, a 'rising' threshold
		tot = 0.5 #ns
		totsum = 0
		drv_samples = 1 #number of samples about index to take derivative
		trigd = False 
		trigt = None 
		trigidx = None 
		pulse_list = [] #[[passtime, endtime], ...]
		for i, s in enumerate(trim_sig):
			thisdrv = self.check_ave_derivative(trim_sig, trim_times, i, drv_samples)

			#if it is passing threshold requirements
			if(trigd == False):
				#threshold requirement
				#1: derivative is the right sign
				#2: going above const threshold
				#3: on the right side of 0
				if(np.sign(thisdrv) == np.sign(sig_thresh) \
				and np.sign(s) == np.sign(sig_thresh) \
				and np.abs(s) >= np.abs(sig_thresh)):
					trigd = True
					trigt = trim_times[i]
					trigidx = i
				
			#if we already triggered
			else:
				#is this the sample that falls
				#below threshold? conditions:
				#1: going below const threshold
				#2: avg derivative is opposite of threshold sign
				if(np.abs(s) <= np.abs(sig_thresh)\
					and np.sign(thisdrv) != np.sign(sig_thresh)):
					#if it passed TOT, append to pulse list
					if(totsum >= tot):
						endtime = trim_times[i - 1]
						pulse_list.append([trigt, endtime])

					trigt = None
					trigd = False
					trigidx = None
					totsum = 0
				else:
					totsum = trim_times[i] - trigt


		#if you reach the end of the event and we are still above
		#threshold and it is longer than tot threshold, call it a pulse
		if(trigd == True and totsum >= tot):
			endtime = trim_times[i - 1]
			pulse_list.append([trigt, endtime])


		self.pulse_list = pulse_list

		return pulse_list













	#--------------functional utilities------------#
	#finds average derivative about "rang" samples
	#about idx
	def check_ave_derivative(self, y, x, idx, rang):
		avdr = 0
		samcount = 0
		for i in range(idx - rang, idx + rang + 1):
			if(i < 0):
				continue
			if(i >= len(x) - 1):
				continue
			avdr += (y[i+1] - y[i])/(x[i+1] - x[i])
			samcount += 1

		avdr /= samcount
		return avdr



	#interpolates about idx which is closest value to the key
	def linear_interpolation(self, key, y, x, idx):
		#case 1: idx is 0, need to project back
		#with the slope found from sample ahead
		if(idx == 0):
			m = (y[idx + 1] - y[idx])/(x[idx + 1] - x[idx])
			b = y[idx] - m*x[idx]
			

		#case 2: idx is at the end. use slope 
		#from previous sample
		elif(idx == len(x) - 1):
			m = (y[idx] - y[idx-1])/(x[idx] - x[idx-1])
			b = y[idx] - m*x[idx]

		else:
			#by default, use slope on the left side
			#of the closest index
			m = (y[idx] - y[idx-1])/(x[idx] - x[idx-1])
			b = y[idx] - m*x[idx]

		#sometimes m is zero. 
		#in this case, use x[idx]
		if(np.abs(m) < 1e-6):
			return x[idx]
			
		return (key - b)/m









