import numpy as np 
import sys

class Metadata:
	def __init__(self, metadata_list):
		metadata_list = [int(_) for _ in metadata_list]
		self.event_id = metadata_list[0] #event id in event dataset (starts from 0 and counts)
		self.board_id = metadata_list[1] #board id relative to ACC card (0 - 8)
		self.bin_count_rise = metadata_list[7] #"bin count rise edge" from EJO
		self.self_trig_settings2 = metadata_list[8] #"self trigger settings" convert to hex to learn more
		self.self_trig_settings1 = metadata_list[12] #"self trigger settings" convert to hex to learn more
		self.self_trig_mask = metadata_list[26]
		self.dig_timestamp = [metadata_list[29], metadata_list[28], metadata_list[27]] #[hi, mid, low]
		self.dig_event_id = metadata_list[30]
		self.board_timestamp = [metadata_list[32], metadata_list[33], metadata_list[34]] #[hi, mid, low]
		self.acc_timestamp = [metadata_list[39], metadata_list[38], metadata_list[37]] #[hi, mid, low]
		self.acc_event_id = metadata_list[36]
		self.acc_bin_count = metadata_list[35]

		#chip metadata
		self.actual_clock = [] #MHz
		self.target_clock = [] #MHz
		self.bias = [] #mv
		self.peds = [] #mv
		self.trig_thesh = [] #mv
		self.number_of_chips = 6
		bits = 4096.0
		ref_mv = 1200.0
		for i in range(self.number_of_chips):
			self.actual_clock.append(metadata_list[40+i]*10*(2**11)/(10**6))
			self.target_clock.append(metadata_list[45+i]*10*(2**11)/(10**6))
			self.peds.append(metadata_list[50+i]*ref_mv/bits)
			self.trig_thesh.append(metadata_list[55+i]*ref_mv/bits)
			self.bias.append(metadata_list[60+i]*ref_mv/bits)


	def printChipSettings(self):
		for i in range(self.number_of_chips):
			print "PSEC: " + str(i),
			print "| ADC clock/trgt: " + str(self.actual_clock[i]) + "/" + str(self.target_clock[i]) + "MHz",
			print ", bias: " + str(self.bias[i]) + "mV",
			print "| Ped: "  + str(self.peds[i]) + "mV",
			print "| Trig: " + str(self.trig_thesh[i]) + "mV"
