#!/usr/bin/env python

#import gxutil
import matplotlib.pyplot as plt
import numpy as np
import ebf
import sys, os
import tvguide
import gxutil
from despyastro.coords import *

#Define constant variables
Msun = 1.989e33
Rsun = 6.955e10
Mearth = 5.974e27
Rearth = 7.478e8
G = 6.6743e-8

# Star data: mass, radius, teff, logg, ra, dec, observed_days, has_planet
# Planet data: planet_mass, planet_radius, period, has_transit, t_duration

# Selects LLRGB stars south of the ecliptic that will be observed with TESS
def star_population(teff_upper = 5500, logg_lower = 2.5, logg_upper = 3.5, pop_multi=10):
	# -------- Synthetic stellar population ------------
	galaxia = ebf.read('data/tess.ebf','/')
	num_stars = galaxia["teff"].size*pop_multi
	
	# Convert magnitudes to apparent mag and apply reddening
	gxutil.abs2app(galaxia,corr=True)

	# Magnitude cut
	mag_cut = galaxia["ubv_i"] < 13
	# Cut in Teff
	teff_cut = 10**galaxia["teff"] < teff_upper
	# Cut in Logg
	logg_cut = np.logical_and(galaxia["grav"] > logg_lower, galaxia["grav"] < logg_upper)
	# Join vectors
	llrgb_mask = np.logical_and(np.logical_and(teff_cut, logg_cut), mag_cut)
	non_llrgb_mask = np.invert(llrgb_mask)
	llrgb_index = np.where(llrgb_mask)[0]
	num_llrgb = llrgb_index.size*pop_multi

	# Get galactic coords
	g_lon = galaxia["glon"][llrgb_mask]
	g_lat = galaxia["glat"][llrgb_mask]

	# Convert to ecliptic coords
	ec_lon, ec_lat = gal2ec(g_lon, g_lat)

	# Select from south ecliptic pole
	llrgb_south_mask = ec_lat < 0
	llrgb_south_index = llrgb_index[llrgb_south_mask]
	num_llrgb_south = llrgb_south_index.size*pop_multi

	ec_lon = ec_lon[llrgb_south_mask]
	ec_lat = ec_lat[llrgb_south_mask]
	g_lon = g_lon[llrgb_south_mask]
	g_lat = g_lat[llrgb_south_mask]

	# Convert to equatorial
	ra, dec = ec2eq(ec_lon, ec_lat)

	# Check sectors for stars using tvguide
	min_sec, max_sec, med_sec = tvguide.check_many(ra, dec)[:,2:5].T

	# Select only stars that show up in one sector at least
	med_mask = med_sec > 0
	sample_index = llrgb_south_index[med_mask]
	num_sample = sample_index.size*pop_multi
	
	# Get data for all stars in sample
	med_sec = med_sec[med_mask]
	min_sec = min_sec[med_mask]
	ra = ra[med_mask]
	dec = dec[med_mask]
	ec_lon = ec_lon[med_mask]
	ec_lat = ec_lat[med_mask]
	g_lon = g_lon[med_mask]
	g_lat = g_lat[med_mask]


	mass = galaxia["mact"][sample_index]
	teff = 10**(galaxia["teff"][sample_index])
	feh = galaxia["feh"][sample_index]
	age = galaxia["age"][sample_index]
	lum = galaxia["lum"][sample_index]
	logg = galaxia["grav"][sample_index]
	radius = np.sqrt(G * mass * Msun / 10**logg) / Rsun
	med_obs = med_sec.astype(int) * 27.4
	min_obs = min_sec.astype(int) * 27.4

	ubv_u = galaxia["ubv_u"][sample_index]
	ubv_b = galaxia["ubv_b"][sample_index]
	ubv_v = galaxia["ubv_v"][sample_index]
	ubv_r = galaxia["ubv_r"][sample_index]
	ubv_i = galaxia["ubv_i"][sample_index]
	ubv_j = galaxia["ubv_j"][sample_index]
	ubv_h = galaxia["ubv_h"][sample_index]
	ubv_k = galaxia["ubv_k"][sample_index]

	# Join data in a matrix
	data = np.column_stack((sample_index, g_lon, g_lat, ec_lon, ec_lat, ra, dec, mass, radius, age, lum, teff, logg, feh,
		med_obs, min_obs, ubv_u, ubv_b, ubv_v, ubv_r, ubv_i, ubv_j, ubv_h, ubv_k))

	# Generate file header
	header =  "Data from the complete star file, including the factor of 2 correction \n"
	header += "{:}{:}\n".format("Total number of stars: ", num_stars)
	header += "{:}{:}\n".format("Total number of llrgb stars: ", num_llrgb)
	header += "{:}{:}\n".format("Total number of llrgb stars south of the ecliptic: ", num_llrgb_south)
	header += "{:}{:}\n\n".format("Total number of llrgb stars south of the ecliptic, observable by TESS: ", num_sample)
	header += "{:5}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>13}{:>10}{:>10}{:>7}{:>7}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}".format(
		"Index", "Gal_Lon", "Gal_Lat", "Ec_Lon", "Ec_Lat", "Ra", "Dec", "Mass", "Radius", "Age", "Lum", "Teff", 
		"Logg", "FeH", "Med Obs", "Min Obs", "uvb_u", "uvb_b", "uvb_v", "uvb_r", "uvb_i", "uvb_j", "uvb_h", "uvb_k", )
	
	# Save data to file. Prompt for overwrite if file exists
	txtformat = '%7d %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %12.4f %9.4f %9.4f %6.1f %6.1f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f'
	savefile = "star_sample_2.dat"
	if os.path.exists("data/" + savefile):
		while True:
			entry = input("Overwrite star_sample.dat file? [y/n]")
			if entry.lower() == "y" or entry.lower() == "yes":
				np.savetxt("data/" + savefile, data, fmt=txtformat, header=header)
				break
			elif entry.lower() == "n" or entry.lower() == "no":
				break
			else:
				print("Please input y or n")
				continue
	else:
		np.savetxt("data/" + savefile, data, fmt=txtformat, header=header)



def plot_hr(pop_multi=10):
	galaxia = ebf.read('data/tess.ebf','/')
	num_stars = galaxia["teff"].size*pop_multi

	# Cut in Teff
	teff_cut = 10**galaxia["teff"] < teff_upper
	# Cut in Logg
	logg_cut = np.logical_and(galaxia["grav"] > logg_lower, galaxia["grav"] < logg_upper)
	# Join vectors
	llrgb_mask = np.logical_and(teff_cut, logg_cut)
	non_llrgb_mask = np.invert(llrgb_mask)

	# Plot HR diagram of all the stars in the simulation with the selected ones highlighted
	plt.figure(9, figsize=(24,18), dpi=100)
	plt.scatter(galaxia["teff"][non_llrgb_mask][1::500], galaxia["grav"][non_llrgb_mask][1::500], s=2, color="blue")
	plt.scatter(galaxia["teff"][llrgb_mask][1::500], galaxia["grav"][llrgb_mask][1::500], s=2, color="red")
	plt.gca().invert_xaxis()
	plt.gca().invert_yaxis()
	plt.xlabel(r"$T_{eff}$ (K)", fontsize=24)
	plt.ylabel(r"log $g$ (g $cm^{-2}$)", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "hr_diagram.png")

	plt.close("all")


if __name__ == "__main__":
	star_population()
	#plot_hr()
