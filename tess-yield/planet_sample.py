#!/usr/bin/env python

#import gxutil
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.stats as stats


#Define constant variables
Msun = 1.989e33
Rsun = 6.955e10
Mearth = 5.974e27
Rearth = 6.378e8
G = 6.6743e-8

# Execution flags
verbose 					= 0

planet_seeding 				= 1
plot_hist 					= 0
plot_result_distribution	= 0

# Star data: mass, radius, teff, logg, ra, dec, observed_days, has_planet
# Planet data: planet_mass, planet_radius, period, has_transit, t_duration

if planet_seeding:
	# -------- Planet Seeding ------------
	_, _, _, mass, radius, teff, logg, observed_days = np.loadtxt("data/star_sample.dat", unpack=True)

	# Rates
	planet_rate = 1.0
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
	#b = (a * cos_i) / (radius[has_planet]*Rsun)
	b = np.zeros(planet_radius.size)
	# Choose planets that transit the planet
	has_transit = np.copy(has_planet)
	has_transit[has_transit] = abs(b) < 1
	transiting_planet = abs(b) < 1

	# Determine density of star
	rho_star = mass[has_transit] / ((4.0/3.0) * np.pi * radius[has_transit]**3)
	rho_sun = Msun / ((4.0/3.0) * np.pi * Rsun**3)

	# Determine transit duration and depth
	t_duration = 13 * (period[transiting_planet]/365.0)**(1.0/3.0) * (rho_star/rho_sun)**(-1.0/3.0) * (np.sqrt(1-b[transiting_planet]**2))
	t_depth = ((planet_radius[transiting_planet] * Rearth)**2) / ((radius[has_transit] * Rsun)**2)
	# Determine number of transits. Use ceiling to improve number of planets
	n_transits = np.ceil(observed_days[has_transit]/period[transiting_planet]).astype(int)

	# Choose planets that transit more than 2 times
	has_min_transits = np.copy(has_planet)
	has_min_transits[has_min_transits] = n_transits >= min_n_transits
	min_transits = n_transits >= min_n_transits

	# -------- Transit Detectability ------------
	# Use Dan Huber's method to determine rms values for a star's logg for various predefined transit durations
	coeffs = np.loadtxt("data/noisecoeffs.dat", unpack=True)
	tdurs = np.array([0.1,0.5,1.0,1.5,2.0,2.5])
	res = np.array([np.polyval(coeffs[:,i][::-1], logg[has_min_transits]) for i in range(6)])

	# Determine the rms for the planets transit by interpolating the res values
	rms = np.array([np.interp(t_duration[min_transits][i]/24.0, tdurs, res[:,i]) for i in range(t_duration[min_transits].size)])
	
	# Signal to noise ratio
	#SNR = 10
	SNR = (t_depth[min_transits] / (rms)) * np.sqrt(n_transits[min_transits])

	# Calculate minimum detectable radius
	Rmin = radius[has_min_transits]*Rsun * ((SNR * rms)**0.5) * (n_transits[min_transits]**(-1.0/4))
	is_detectable = Rmin < (planet_radius[min_transits] * Rearth)
	num_detectable = sum(is_detectable)

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

if plot_result_distribution:
	planet_rates = [0.1, 0.05, 0.01]
	for rate in planet_rates:
		planets, transits, detections = np.loadtxt("data/planet_rate_"+str(rate)+"_ntr_2.dat", unpack=True)

		plt.figure(10, figsize=(24,18), dpi=100)
		plt.hist(planets, bins=20, normed=1, alpha=0.6, label="Planet rate - "+str(int(rate*100))+"%")
		plt.xlabel(r"Number of planets seeded", fontsize=24)
		plt.ylabel(r"Frequency", fontsize=24)
		plt.tick_params(labelsize=24)
		plt.legend(fontsize=24)
		plt.savefig("figures/hist_results/" + "hist_planets_"+str(rate)+".png")

		plt.figure(11, figsize=(24,18), dpi=100)
		plt.hist(transits, bins=20, normed=1, alpha=0.6, label="Planet rate - "+str(int(rate*100))+"%")
		plt.xlabel(r"Number of planet transits", fontsize=24)
		plt.ylabel(r"Frequency", fontsize=24)
		plt.tick_params(labelsize=24)
		plt.legend(fontsize=24)
		plt.savefig("figures/hist_results/" + "hist_transits_"+str(rate)+".png")

		val = rate*100
		plt.figure(12, figsize=(24,18), dpi=100)
		if rate == 0.01:
			bins = [0,2,4,6,8,10,12,14,16,18,20]
			plt.xticks(bins)
		else:
			bins = 20
		plt.hist(detections, bins=bins, normed=1, alpha=0.6, label="Planet rate - "+str(int(rate*100))+"%")
		plt.xlabel(r"Number of planet detections", fontsize=24)
		plt.ylabel(r"Frequency", fontsize=24)
		plt.tick_params(labelsize=24)
		plt.legend(fontsize=24)
		plt.savefig("figures/hist_results/" + "hist_detections_"+str(rate)+".png")

		plt.close("all")

plt.close("all")