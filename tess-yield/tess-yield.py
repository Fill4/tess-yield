#!/usr/bin/env python

import ebf
#import gxutil
import matplotlib.pyplot as plt
import numpy as np
import sys
import tvguide
import scipy.stats as stats
from despyastro.coords import *

#Define constant variables
Msun = 1.989e33
Rsun = 6.955e10
Mearth = 5.974e27
Rearth = 6.378e8
G = 6.6743e-8

# Execution flags
verbose 					= 1

star_population 			= 0
planet_seeding 				= 0
plot_hist 					= 0
plot_hr 					= 0
plot_result_distribution	= 1

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
if planet_seeding:
	# -------- Planet Seeding ------------
	_, _, _, mass, radius, teff, logg, observed_days = np.loadtxt("data/star_sample.dat", unpack=True)

	# Rates
	planet_rate = 0.1
	min_n_transits = 2
	# Seed planet
	has_planet = np.random.uniform(0.0, 1.0, mass.size) < planet_rate

	# Planet radius distribution
	lower, upper, scale = 4, 22, 18
	X = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)

	invalid_planets = np.copy(has_planet)
	while np.any(invalid_planets):
		# Draw period samples (in days from a lognormal distribution. Values from Thomas North)
		period = np.random.lognormal(2.344, 1.275, sum(invalid_planets))
		# Draw planet radius samples (in R_earth)
		planet_radius = X.rvs(sum(invalid_planets))
		# Determine planet mass using Equation 29 from Sullivan et al. 2015 (in M_earth)
		planet_mass = 2.69 * planet_radius**0.93
		# Determine Roche limit
		roche = 1.26 * planet_radius*Rearth * (mass[invalid_planets]*Msun / (planet_mass*Mearth))**(1.0/3.0)
		# Determine semi-major axis
		a = ((G * mass[invalid_planets]*Msun * (period*365.25*24*3600)**2) / (4 * np.pi**2))**(1.0/3.0)
		# Check if any star is above roche limit
		n_invalid_planets = sum(a < roche)
		# If there are no invalid planets exit loop
		if n_invalid_planets != 0:
			invalid_planets[invalid_planets] = a < roche
		else:
			break
			

	# Draw cos(i) distribution
	cos_i = np.random.uniform(0.0, 1.0, sum(has_planet))
	# Determine impact parameter
	b = (a * cos_i) / (radius[has_planet]*Rsun)
	# Choose planets that transit the planet
	has_transit = np.copy(has_planet)
	has_transit[has_transit] = abs(b) < 1
	transiting_planet = abs(b) < 1

	rho_star = mass[has_transit] / ((4.0/3.0) * np.pi * radius[has_transit]**3)
	rho_sun = Msun / ((4.0/3.0) * np.pi * Rsun**3)
	t_duration = 13 * (period[transiting_planet]/365.0)**(1.0/3.0) * (rho_star/rho_sun)**(-1.0/3.0) * (np.sqrt(1-b[transiting_planet]**2))

	# -------- Transit Detectability ------------

	coeffs = np.loadtxt("data/noisecoeffs.dat", unpack=True)
	tdurs = np.array([0.1,0.5,1.0,1.5,2.0,2.5])

	res = np.array([np.polyval(coeffs[:,i][::-1], logg[has_transit]) for i in range(6)])

	rms = np.array([np.interp(t_duration[i]/24.0, tdurs, res[:,i]) for i in range(t_duration.size)])
	SNR = 10
	n_transits = np.trunc(observed_days[has_transit]/period[transiting_planet]).astype(int)

	with np.errstate(divide='ignore'):
		Rmin = radius[has_transit]*Rsun * ((SNR * rms)**0.5) * (n_transits**(-1.0/4))
	is_detectable = Rmin < (planet_radius[transiting_planet] * Rearth)
	is_detectable = np.logical_and(is_detectable, n_transits >= min_n_transits)
	num_detectable = sum(is_detectable)

	Rmin[Rmin == np.inf] = np.nan

	#---------------------------------------------------------

	if verbose:
		print("Number of stars with seeded planets: " + str(sum(has_planet) * 2))
		print("Percentage of stars with seeded planets: " + str(sum(has_planet)*1.0 / mass.size))
		print("Number of stars with transiting planets: " + str(sum(transiting_planet) * 2))
		print("Percentage of transiting planets from seeded planets: " + str(sum(transiting_planet)*1.0 / sum(has_planet)*1.0))

		print("Number of detectable transiting planets: " + str(num_detectable*2))
		print("Percentage of detectable planets from transiting planets: " + str(num_detectable/sum(transiting_planet)))
 	
if plot_hist:
	# PLots for all the planets
	plt.figure(1, figsize=(24,16), dpi=100)
	plt.hist(planet_radius, bins=20, normed=1)
	plt.xlabel(r"Planet Radius [$R_\oplus$]", fontsize=24)
	plt.ylabel(r"Frequency", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "all_planet_radius_scale={:}.png".format(scale))

	plt.figure(2, figsize=(24,16), dpi=100)
	plt.hist(period, range=[0, 100], bins=40, normed=1)
	plt.xlabel(r"Planet Period [days]", fontsize=24)
	plt.ylabel(r"Frequency", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "all_planet_period.png")

	plt.figure(3, figsize=(24,18), dpi=100)
	plt.scatter(planet_radius, np.log(period), color="blue", s=1)
	plt.xlabel(r"Planet Radius [$R_\oplus$]", fontsize=24)
	plt.ylabel(r"Planet Period [days]", fontsize=24)
	#plt.ylim([0, 100])
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "all_period_radius.png")

	# Plots for transiting planets
	plt.figure(4, figsize=(24,16), dpi=100)
	plt.hist(planet_radius[transiting_planet], bins=20, normed=1)
	plt.xlabel(r"Planet Radius [$R_\oplus$]", fontsize=24)
	plt.ylabel(r"Frequency", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "transit_planet_radius_scale={:}.png".format(scale))

	plt.figure(5, figsize=(24,16), dpi=100)
	plt.hist(period[transiting_planet], range=[0, 100], bins=40, normed=1)
	plt.xlabel(r"Planet Period [days]", fontsize=24)
	plt.ylabel(r"Frequency", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "transit_planet_period.png")

	plt.figure(6, figsize=(24,18), dpi=100)
	plt.scatter(planet_radius[transiting_planet], period[transiting_planet], color="blue", s=10)
	plt.xlabel(r"Planet Radius [$R_\oplus$]", fontsize=24)
	plt.ylabel(r"Planet Period [days]", fontsize=24)
	plt.ylim([0, 100])
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "transit_period_radius.png")
	
	plt.figure(7, figsize=(24,16), dpi=100)
	plt.hist(t_duration, bins=20, normed=1)
	plt.xlabel(r"Transit Duration [hours]", fontsize=24)
	plt.ylabel(r"Frequency", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "transit_duration.png")

	plt.figure(8, figsize=(24,16), dpi=100)
	plt.scatter(logg[has_transit][is_detectable], (planet_radius[transiting_planet][is_detectable]*Rearth) / Rmin[is_detectable], s=25, color='red', label="Detectable")
	plt.scatter(logg[has_transit][~is_detectable], (planet_radius[transiting_planet][~is_detectable]*Rearth) / Rmin[~is_detectable], s=25, color='blue', label="Not detectable")
	plt.xlabel(r"log $g$ (g $cm^{-2}$)", fontsize=24)
	plt.ylabel(r"$R_{p}$ / $R_{min}$", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.legend(fontsize=24)
	plt.savefig("figures/" + "rmin_logg.png")

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

if plot_result_distribution:
	planets, transits, detections = np.loadtxt("data/planet_rate_0.1_ntr_2.dat", unpack=True)

	plt.figure(10, figsize=(24,18), dpi=100)
	plt.hist(planets, bins=20, normed=1)
	plt.xlabel(r"Number of planets seeded", fontsize=24)
	plt.ylabel(r"Frequency", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "hist_planets_0.1_ntr_2.png")

	plt.figure(11, figsize=(24,18), dpi=100)
	plt.hist(transits, bins=20, normed=1)
	plt.xlabel(r"Number of planet transits", fontsize=24)
	plt.ylabel(r"Frequency", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "hist_transits_0.1_ntr_2.png")

	plt.figure(12, figsize=(24,18), dpi=100)
	plt.hist(detections, bins=20, normed=1)
	plt.xlabel(r"Number of planet detections", fontsize=24)
	plt.ylabel(r"Frequency", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "hist_detections_0.1_ntr_2.png")

plt.close("all")