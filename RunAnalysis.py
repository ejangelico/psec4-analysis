import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys
import Waveform
import DataSet
import cPickle as pickle
import time


def plot_time_metadata(d):
	fig, ax = plt.subplots(figsize=(12,7),ncols=3, nrows=3)
	ax_1d = []
	for _ in ax:
		for i in range(len(_)):
			ax_1d.append(_[i])

	time_keys = ["dig_timestamp_hi", "dig_timestamp_mid", "dig_timestamp_lo", \
				"timestamp_hi", "timestamp_mid", "timestamp_lo", \
				"CC_TIMESTAMP_HI", "CC_TIMESTAMP_MID", "CC_TIMESTAMP_LO"]

	for i, key in enumerate(time_keys):
		events, boards, values = d.get_all_metadata(key)
		ev_b0 = []
		v_b0 = []
		ev_b2 = []
		v_b2 = []
		for j in range(len(events)):
			if(events[j] > 50):
				break
			if(boards[j] == 0):
				ev_b0.append(events[j])
				v_b0.append(values[j])
			else:
				ev_b2.append(events[j])
				v_b2.append(values[j])

		ax_1d[i].plot(ev_b0, v_b0, 'b', label=key)
		ax_1d[i].plot(ev_b2, v_b2, 'r', label=key)

	plt.show()





if __name__ == "__main__":
	
	f = "test-data/test.acdc"
	#f = "data/fnaldata/trigtest/coinc_valid_1win.acdc"
	t0 = time.time()
	d = DataSet.DataSet(f)
	t1 = time.time()
	print str(t1 - t0)
	d.plot_all_events()
	#d.plot_event_heatmap(2, 0)
	#d.plot_random_events(4, [2])
	#d.load_random_events(10)
	#sys.exit()
	#plot_time_metadata(d)
	#sys.exit()
	#d.print_metadata(4, 0)
	#sys.exit()
	#chs = [d.which_channels_triggered(_, 2) for _ in range(5)]
	#print chs
	#sys.exit()
	#d.print_metadata(0, 2)
	sys.exit()

	events, boards, values = d.get_all_metadata("self_trig_settings_2")
	print values
	print boards
	sys.exit()
	#print [bin(_) for _ in values]
	maxev = d.get_max_events()
	brds = d.get_boards()
	chs = []

	#fig, ax = plt.subplots()
	#ax.scatter(events, values)


	#plt.show()



