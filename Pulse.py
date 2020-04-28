class Pulse:
	def __init__(self, amplitude=None, peak_time=None, arrival_time=None, rise_time=None, charge=None):
		self.amplitude = amplitude
		self.peak_time = peak_time	
		self.arrival_time = arrival_time
		self.rise_time = rise_time
		self.charge = charge


	def set_amplitude(self, a):
		self.amplitude = a

	def set_peak_time(self, t):
		self.peak_time = t

	def set_arrival_time(self, t):
		self.arrival_time = t 

	def set_rise_time(self, t):
		self.rise_time = t 

	def set_charge(self, q):
		self.charge = q

	def get_amplitude(self):
		return self.amplitude

	def get_peak_time(self):
		return self.peak_time

	def get_arrival_time(self):
		return self.arrival_time 

	def get_rise_time(self):
		return self.rise_time 

	def get_charge(self):
		return self.charge


