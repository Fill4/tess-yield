#!/usr/bin/env python

#import gxutil
import matplotlib.pyplot as plt
import numpy as np
import ebf
import sys
import tvguide
from despyastro.coords import *


#Define constant variables
Msun = 1.989e33
Rsun = 6.955e10
Mearth = 5.974e27
Rearth = 6.378e8
G = 6.6743e-8

# Execution flags
verbose 					= 0

star_population 			= 0
plot_hr 					= 0

# Star data: mass, radius, teff, logg, ra, dec, observed_days, has_planet
# Planet data: planet_mass, planet_radius, period, has_transit, t_duration

# load simulation
galaxia = ebf.read('data/tess.ebf','/')
num_stars = galaxia["teff"].size*2

# Define llrgb cuts
teff_lower = 0
teff_upper = 5500
logg_lower = 2.5
logg_upper = 3.5

# Selects LLRGB stars south of the ecliptic that will be observed with TESS
if star_population:
	# -------- Synthetic stellar population ------------

	# Cut in Teff
	teff_cut = 10**galaxia["teff"] < teff_upper
	# Cut in Logg
	logg_cut = np.logical_and(galaxia["grav"] > logg_lower, galaxia["grav"] < logg_upper)
	# Join vectors
	llrgb_mask = np.logical_and(teff_cut, logg_cut)
	non_llrgb_mask = np.invert(llrgb_mask)
	llrgb_index = np.where(llrgb_mask)[0]
	num_llrgb = llrgb_index.size*2

	# Get galactic coords
	g_lon = galaxia["glon"][llrgb_mask]
	g_lat = galaxia["glat"][llrgb_mask]

	# Convert to ecliptic coords
	ec_lon, ec_lat = gal2ec(g_lon, g_lat)

	# Select from south ecliptic pole
	llrgb_south_mask = ec_lat < 0
	llrgb_south_index = llrgb_index[llrgb_south_mask]
	num_llrgb_south = llrgb_south_index.size*2

	ec_lon = ec_lon[llrgb_south_mask]
	ec_lat = ec_lat[llrgb_south_mask]


	# Convert to equatorial
	ra, dec = ec2eq(ec_lon, ec_lat)

	# Check sectors for stars using tvguide
	min_sec, max_sec, med_sec = tvguide.check_many(ra, dec)[:,2:5].T

	# Select only stars that show up in one sector at least
	med_mask = med_sec > 0
	sample_index = llrgb_south_index[med_mask]
	num_sample = sample_index.size*2
	
	# Ge data for all stars in sample
	med_sec = med_sec[med_mask]
	ra = ra[med_mask]
	dec = dec[med_mask]

	mass = galaxia["mact"][sample_index]
	teff = 10**(galaxia["teff"][sample_index])
	logg = galaxia["grav"][sample_index]
	radius = np.sqrt(G * mass * Msun / 10**logg) / Rsun
	observed_days = med_sec.astype(int) * 27.4

	# Join data in a matrix
	data = np.column_stack((sample_index, ra, dec, mass, radius, teff, logg, observed_days))

	# Generate header and save all data to file
	header =  "Data from the complete star file, including the factor of 2 correction \n"
	header += "{:}{:}\n".format("Total number of stars: ", num_stars)
	header += "{:}{:}\n".format("Total number of llrgb stars: ", num_llrgb)
	header += "{:}{:}\n".format("Total number of llrgb stars south of the ecliptic: ", num_llrgb_south)
	header += "{:}{:}\n\n".format("Total number of llrgb stars south of the ecliptic, observable by TESS: ", num_sample)
	header += "{:5}{:>10}{:>10}{:>10}{:>10}{:>13}{:>10}{:>7}".format("Index", "Ra", "Dec", "Mass", "Radius", "Teff", "Logg", "Days")
	
	np.savetxt("data/star_sample.dat", data, fmt='%7d %9.4f %9.4f %9.4f %9.4f %12.4f %9.4f %6.1f', header=header)

	"""
	# convert to apparent mag and apply reddening
	gxutil.abs2app(galaxia,corr=True)

	plt.hist(galaxia['ubv_i'],bins=100)
	plt.show()
	"""

if plot_hr:
	# Cut in Teff
	teff_cut = 10**galaxia["teff"] < teff_upper
	# Cut in Logg
	logg_cut = np.logical_and(galaxia["grav"] > logg_lower, galaxia["grav"] < logg_upper)
	# Join vectors
	llrgb_mask = np.logical_and(teff_cut, logg_cut)
	non_llrgb_mask = np.invert(llrgb_mask)

	plt.figure(9, figsize=(24,18), dpi=100)
	plt.scatter(galaxia["teff"][non_llrgb_mask][1::500], galaxia["grav"][non_llrgb_mask][1::500], s=2, color="blue")
	plt.scatter(galaxia["teff"][llrgb_mask][1::500], galaxia["grav"][llrgb_mask][1::500], s=2, color="red")
	plt.gca().invert_xaxis()
	plt.gca().invert_yaxis()
	plt.xlabel(r"$T_{eff}$ (K)", fontsize=24)
	plt.ylabel(r"log $g$ (g $cm^{-2}$)", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "hr_diagram.png")