import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates
import sys
import Waveform
import DataSet
import pickle
import time
from itertools import combinations
from scipy.stats import norm
import Board
import Event
import Pulse
import random
import os 
from datetime import datetime
from os import listdir
from os.path import isfile, join
from scipy.optimize import curve_fit





#similar to create_event but only saves the specified channels. 
def create_event_reduced(dset, evno, chs, sync_chs = None, board_mask = None):
	triggered_boards = dset.which_boards_triggered(evno)
	#if no boards triggered, then no event exists with that number
	if(len(triggered_boards) == 0):
		return None

	#create board objects for the triggered boards
	boards = [] 
	for bnum in triggered_boards:
		if(board_mask is None):
			wfmlist = []
			chlist = []
			for ch in dset.get_chs():
				if(ch in chs):
					wfmlist.append(dset.get_waveform(evno, bnum, ch))
					chlist.append(ch)
					if(wfmlist[-1] is None):
						return None

			metadata = dset.get_metadata_structure(evno, bnum) #returns a dictionary object

			boards.append(Board.Board(wfmlist, chlist, bnum, metadata, sync_chs))

		else:
			if(bnum in board_mask):
				wfmlist = []
				chlist = []
				for ch in dset.get_chs():
					if(ch in chs):
						wfmlist.append(dset.get_waveform(evno, bnum, ch))
						chlist.append(ch)
						if(wfmlist[-1] is None):
							return None

				metadata = dset.get_metadata_structure(evno, bnum) #returns a dictionary object

				boards.append(Board.Board(wfmlist, chlist, bnum, metadata, sync_chs))


	event = Event.Event(boards, evno)
	return event


#a general template for creating event objects
#if boards is None, will load all boards. If it is a list,
#will load only those boards mentioned in the list. 
def create_event(dset, evno, sync_chs = None, board_mask = None):
	triggered_boards = dset.which_boards_triggered(evno)
	#if no boards triggered, then no event exists with that number
	if(len(triggered_boards) == 0):
		return None

	#create board objects for the triggered boards
	boards = [] 
	for bnum in triggered_boards:
		if(board_mask is None):
			wfmlist = []
			for ch in dset.get_chs():
				wfmlist.append(dset.get_waveform(evno, bnum, ch))
				if(wfmlist[-1] is None):
					return None

			metadata = dset.get_metadata_structure(evno, bnum) #returns a dictionary object

			boards.append(Board.Board(wfmlist, dset.get_chs(), bnum, metadata, sync_chs))

		else:
			if(bnum in board_mask):
				wfmlist = []
				for ch in dset.get_chs():
					wfmlist.append(dset.get_waveform(evno, bnum, ch))
					if(wfmlist[-1] is None):
						return None

				metadata = dset.get_metadata_structure(evno, bnum) #returns a dictionary object

				boards.append(Board.Board(wfmlist, dset.get_chs(), bnum, metadata, sync_chs))


	event = Event.Event(boards, evno)
	return event



def plot_random_events(nev, f):
	t0 = time.time()
	d = DataSet.DataSet(f)
	t1 = time.time()
	sync_chs = [6,12,18,24,30]
	endedness = {0:"double", 2:"double"}
	#print_dataset_statistics(d, sync_chs)
	maxevt = d.get_max_events()
	allevts = range(maxevt)
	ran_events = random.sample(allevts, nev) #random elements from allevts without repeating
	for ev in ran_events:
		event = create_event(d, ev, sync_chs)
		for b in endedness:
			event.set_endedness(b, endedness[b])

		event.plot_waveforms_separated(False)
		#event.plot_heatmaps()
		#event.plot_sync_channels()




#this function takes all files
#in a directory and saves events
#that have triggers in all of the 
#specified boards. Structure of the
#coincidence specification is:
#coinc = [[1,2], (or) [0,2], (or) [0,1,2], ...]
#filetag is appended to each input filename. 
def filter_on_coincidence(indir, filetag, coinc):

	#this line enabled will include all files in the folder
	#files = [indir+f for f in listdir(indir) if isfile(join(indir, f)) and f[-5:] == ".acdc"]
	#this line enabled will treat the indir as one file to do. 
	files = [indir+".acdc"]

	sync_chs = []
	endedness = {0:"double", 2:"double", 3:"single"}
	board_template_files = {0: "organized_last_beam/pion_run/template_board2_iteration2.p", 2:"organized_last_beam/pion_run/template_board3_iteration2.p", 3:"cosmic/run2/template_board3.p"}
	bad_chs = {0:[5, 29, 30], 2:[28, 27, 26, 29, 30], 3:[3, 27]}
	filecounter = 0 #counts output files, adding a tag for large batches. 
	for f in files:
		print("Loading file " + f)
		d = DataSet.DataSet(f[:-5]) #without .acdc or .meta tag

		pass_events = [] #list of event objects that pass the cut
		maxevt = d.get_max_events()
		for ev in range(maxevt):
			#incremental save command for batches with large numbers of events
			if(len(pass_events) >= 1000):
				#save a file with the same name but a different tag.
				print("Writing events to file " + f[:-5]+"_"+filetag+str(filecounter)+".p")
				#pickle.dump(pass_events, open(f[:-5]+"_"+filetag+str(filecounter)+".p", "wb"))
				filecounter += 1
				pass_events = [] #empty the list. 



			if(ev % 1 == 0):
				print("On event " + str(ev) + " of " + str(maxevt))

			event = create_event(d, ev)
			if(event is None):
				continue

			event.set_sync_chs(sync_chs)
			for b in endedness:
				event.set_endedness(b, endedness[b])

			event.set_bad_chs(bad_chs)

			trigs, readouts = event.get_board_clocks() #just to see which boards triggered. 
			trignos = sorted(trigs.keys()) #sorted board numbers that triggered
			#check each input board combination. 
			for comb in coinc:
				if(trignos == sorted(comb)):
					#save the event. it passes
					print("Passing board event " + str(comb))
					#take these out if you don't want to run the
					#long pulse finder. 
					print("Expert mode, finding pulses")
					event.set_templates(board_template_files)
					event.median_subtract()
					event.nnls_all()
					event.pulse_find_all()
					pulses = event.get_pulses()
					if(len(pulses) != 0):
						#only save the event if there are pulses. 
						print("Got pulses, saving event")
						pass_events.append(event)
					pass_events.append(event)
					break

		#save a file with the same name but a different tag.
		print("Writing events to file " + f[:-5]+filetag+".p")
		pickle.dump(pass_events, open(f[:-5]+"_"+filetag+".p", "wb"))




def make_templates():

	# This is a first pass before doing any NNLS-ing.
	events = pickle.load(open("organized_last_beam/pion_run/bigchain_lappdcoinc.p", "rb"))

	for ev in events:
		#good template event for board 2
		if(ev.get_event_no() == 392):
			temp_channel = 11
			bo = ev.get_boards()[0]
			wfm = bo.get_waveform(temp_channel)
			wfm.median_subtract([13, 28])
			wfm.trim_on_range([29.56, 33.91])
			splwf = wfm.get_spline_waveform(10, 3)
			splwf = splwf.__mul__(1.0/5.0)
			pickle.dump(splwf, open("organized_last_beam/pion_run/template_board2.p", "wb"))

		#good template event for board 3
		if(ev.get_event_no() == 438):
			temp_channel = 7
			bo = ev.get_boards()[1]
			wfm = bo.get_waveform(temp_channel)
			wfm.median_subtract([9.5, 12.5])
			wfm.trim_on_range([12.9, 15.9])
			splwf = wfm.get_spline_waveform(10, 10)
			splwf = splwf.__mul__(1.0/15.0)
			pickle.dump(splwf, open("organized_last_beam/pion_run/template_board3.p", "wb"))

	






if __name__ == "__main__":
	infile = "your file here without the acdc tag" 
	plot_random_events(10, infile)

	


