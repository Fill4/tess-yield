#!/usr/bin/env python

import sys
import star_sample, planet_sample

if (verbose):
	def vprint(*args):
		print(*args)
else:
	vprint = lambda *a: None

planet_rate = 0.01
num_iter = 300

# Run to complete the star_sample file with all the data from the stars
star_sample.complete_pop_file()


# Repeat planet seeding and plot the resulting histograms 
for i in range(num_iter):
	planet_sample.planet_seeding(planet_rate=planet_rate, write_output=True)
	print("Iterations {:}/{:}".format(i+1, num_iter), end="\r", flush=True)

planet_sample.plot_result_hist(planet_rates[0.1,0.05,0.01])