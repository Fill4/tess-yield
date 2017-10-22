#!/usr/bin/env python

#import gxutil
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.stats as stats
import TASC_detection_recipe as mat_script


#Define constant variables
Msun = 1.989e33
Rsun = 6.955e10
Mearth = 5.974e27
Rearth = 6.378e8
G = 6.6743e-8

# Star data: mass, radius, teff, logg, ra, dec, observed_days, has_planet
# Planet data: planet_mass, planet_radius, period, has_transit, t_duration

def planet_seeding(planet_rate, min_n_transits=2, pop_multi=10, write_output=0, sensitivity_csv=0, plot_hist=0, plot_noise=0, verbose=0):
	
	# -------------------------------------------
	# ------------ Planet Seeding ---------------
	# -------------------------------------------

	(sample_index, g_lon, g_lat, ec_lon, ec_lat, ra, dec, mass, radius, age, lum, teff, logg, feh, observed_days_med, observed_days_min,
	ubv_u, ubv_b, ubv_v, ubv_r, ubv_i, ubv_j, ubv_h, ubv_k) = np.loadtxt("data/star_sample_complete.dat", unpack=True)

	# Parameter definition for building csv data
	if sensitivity_csv:
		planet_rate = 1.0
		min_n_transits = 2

	# Seed planet according to planet rate
	has_planet = np.random.uniform(0.0, 1.0, mass.size) < planet_rate

	# Planet radius distribution
	lower, upper, scale = 4, 22, 18
	X = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)

	# Create arrays for while loop
	period 			= np.zeros(has_planet.sum())
	planet_radius 	= np.zeros(has_planet.sum())
	planet_mass 	= np.zeros(has_planet.sum())
	roche 			= np.zeros(has_planet.sum())
	a 				= np.zeros(has_planet.sum())

	# Loop that samples periods and radius for the seeded planets until all values are physically consistent
	invalid_planets = np.ones(has_planet.sum(), dtype=bool)
	while np.any(invalid_planets):
		# ADD sensitivity_csv OPTION WITH UNIFORM DISTRIBUTIONS
		# Draw period samples (in days from a lognormal distribution. Values from Thomas North)
		period[invalid_planets] = np.random.lognormal(2.344, 1.275, invalid_planets.sum())
		# Draw planet radius samples (in R_earth)
		planet_radius[invalid_planets] = X.rvs(invalid_planets.sum())
		# Determine planet mass using Equation 29 from Sullivan et al. 2015 (in M_earth)
		planet_mass[invalid_planets] = 2.69 * planet_radius[invalid_planets]**0.93
		# Determine Roche limit
		roche[invalid_planets] = 1.26 * planet_radius[invalid_planets]*Rearth * (mass[has_planet][invalid_planets]*Msun / (planet_mass[invalid_planets]*Mearth))**(1.0/3.0)
		# Determine semi-major axis
		a[invalid_planets] = ((G * mass[has_planet][invalid_planets]*Msun * (period[invalid_planets]*24*3600)**2) / (4 * np.pi**2))**(1.0/3.0)
		# Check if any star is above roche limit
		n_invalid_planets = sum(a < roche)
		# If there are no invalid planets exit loop
		if n_invalid_planets != 0:
			invalid_planets = a < roche
		else:
			break

	# Draw cos(i) distribution
	cos_i = np.random.uniform(0.0, 1.0, has_planet.sum())
	# Determine impact parameter
	if sensitivity_csv:
		b = np.zeros(planet_radius.size)
	else:
		b = (a * cos_i) / (radius[has_planet]*Rsun)

	# Choose planets that transit the planet
	has_transit = abs(b) < 1

	# Determine density of star
	rho_star = (mass[has_planet][has_transit]*Mearth) / ((4.0/3.0) * np.pi * (radius[has_planet][has_transit]*Rearth)**3)
	rho_sun = Msun / ((4.0/3.0) * np.pi * Rsun**3)

	# Determine transit duration and depth
	t_duration = 13 * (period[has_transit]/365.0)**(1.0/3.0) * (rho_star/rho_sun)**(-1.0/3.0) * (np.sqrt(1-b[has_transit]**2))
	t_depth = ((planet_radius[has_transit] * Rearth)**2) / ((radius[has_planet][has_transit] * Rsun)**2)
	
	# Determine number of transits. Use ceiling to improve number of planets
	n_transits = np.ceil(observed_days_med[has_planet][has_transit]/period[has_transit]).astype(int)

	# Choose planets that transit more than 2 times
	has_min_transits = n_transits >= min_n_transits

	# -------------------------------------------
	# -------- Transit Detectability ------------
	# -------------------------------------------

	# Use Dan Huber's method to determine gran_noise values for a star's logg for various predefined transit durations
	coeffs = np.loadtxt("data/noisecoeffs.dat", unpack=True)
	tdurs = np.array([0.1,0.5,1.0,1.5,2.0,2.5])
	res = np.array([np.polyval(coeffs[:,i][::-1], logg[has_planet][has_transit][has_min_transits]) for i in range(6)])

	# Determine the gran_noise for the planets transit by interpolating the res values
	gran_noise = np.array([np.interp(t_duration[has_min_transits][i]/24.0, tdurs, res[:,i]) for i in range(t_duration[has_min_transits].size)])

	# Determine shot noise
	_, _, _, npix_aper = mat_script.pixel_cost(ubv_i[has_planet][has_transit][has_min_transits])
	shotnoise = mat_script.calc_noise(ubv_i[has_planet][has_transit][has_min_transits], 1800, 
		teff[has_planet][has_transit][has_min_transits], ec_lon[has_planet][has_transit][has_min_transits], 
		ec_lat[has_planet][has_transit][has_min_transits], g_lon[has_planet][has_transit][has_min_transits], 
		g_lat[has_planet][has_transit][has_min_transits], npix_aper=npix_aper)
	# Shot noise initial value is for one hour of observations. Convert to transit duration
	shotnoise2 = shotnoise / (np.sqrt(t_duration[has_min_transits]))

	print(sum((gran_noise/shotnoise) > 1))
	print(gran_noise[:30]/shotnoise[:30])
	sys.exit()

	# Add noise in quadrature
	noise = np.sqrt(gran_noise**2 + shotnoise**2)

	# Signal to noise ratio
	if sensitivity_csv:
		SNR = (t_depth[has_min_transits] / (noise)) * np.sqrt(n_transits[has_min_transits])
	else:
		SNR = 10

	# Calculate minimum detectable radius
	Rmin = radius[has_planet][has_transit][has_min_transits]*Rsun * ((SNR * noise)**0.5) * (n_transits[has_min_transits]**(-1.0/4))
	is_detectable = Rmin < (planet_radius[has_transit][has_min_transits] * Rearth)
	num_detectable = is_detectable.sum()

	# -------------------------------------------
	# ------------- Handle Results --------------
	# -------------------------------------------

	# Writes the number of planets seeded, with transits and detected into a file
	if write_output:
		with open("data/planet_rate_" + str(planet_rate) + ".dat", "a") as f:
			f.write("{:9d}   {:8d}   {:10d}\n".format(has_planet.sum() * pop_multi, has_transit.sum() * pop_multi, num_detectable * pop_multi))
	
	# Only called when building the csv file to compute the sensitivity plot
	if sensitivity_csv:
		data = np.column_stack((mass[has_planet][has_transit][has_min_transits], radius[has_planet][has_transit][has_min_transits], 
								teff[has_planet][has_transit][has_min_transits], logg[has_planet][has_transit][has_min_transits], 
								observed_days_med[has_planet][has_transit][has_min_transits], period[has_transit][has_min_transits], 
								planet_radius[has_transit][has_min_transits], t_duration[has_min_transits], noise, SNR))

		header = "Columns:\n{:} - {:} - {:} - {:} - {:} -\n{:} - {:} - {:} - {:} - {:}\n".format("Star Mass (Msun)", "Star Radius (Rsun)", "Teff (K)", 
				"Log(g) (cm s-2)", "Obs time (days)", "Planet Period (days)", "Planet Radius (Rearth)", "Transit duration (hours)", "Noise", "SNR")
		
		np.savetxt("data/planet_sample.csv", data, fmt='%.4f,%.4f,%.4f,%.4f,%.1f,%.4f,%.4f,%.4f,%.10f,%.5f', header=header)

	# ----------------------------------------------
	if verbose:
		print("\nPlanet rate of " + str(planet_rate*100) + "%")
		print("\nNumber of stars with seeded planets: " + str(has_planet.sum() * pop_multi))
		print("Percentage of stars with seeded planets: " + str(has_planet.sum()*1.0 / mass.size))
		print("\nNumber of stars with transiting planets: " + str(has_transit.sum() * pop_multi))
		print("Percentage of transiting planets from seeded planets: " + str(has_transit.sum()*1.0 / has_planet.sum()*1.0))
		print("\nNumber of planets with at least 2 transits: " + str(has_min_transits.sum()*pop_multi))
		print("Percentage of transiting planets with at least 2 transits: " + str(has_min_transits.sum()*1.0 / has_transit.sum()*1.0))
		print("\nNumber of detectable planets: " + str(num_detectable*pop_multi))
		print("Percentage of detectable planets from transiting planets: " + str(num_detectable/has_transit.sum()))

	# ----------------------------------------------
	if plot_hist:
		plot_hist_function(logg, planet_radius, period, t_duration, has_planet, has_transit, has_min_transits, is_detectable)
	if plot_noise:
		plot_noise_function(logg, shotnoise, gran_noise, noise, has_planet, has_transit, has_min_transits, is_detectable)

 	
# Plots the distributions (planet_radius, period) for the last run of planet seeding
def plot_hist_function(logg, planet_radius, period, t_duration, has_planet, has_transit, has_min_transits, is_detectable):
	# PLots for all the planets
	plt.figure(1, figsize=(24,16), dpi=100)
	plt.hist(planet_radius, bins=20, normed=1, alpha=0.6)
	plt.xlabel(r"Planet Radius [$R_\oplus$]", fontsize=24)
	plt.ylabel(r"Frequency", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "all_planet_radius_scale={:}.png".format(scale))

	plt.figure(2, figsize=(24,16), dpi=100)
	plt.hist(period, range=[0, 100], bins=40, normed=1, alpha=0.6)
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
	plt.hist(planet_radius[has_transit][has_min_transits], bins=20, normed=1, alpha=0.6)
	plt.xlabel(r"Planet Radius [$R_\oplus$]", fontsize=24)
	plt.ylabel(r"Frequency", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "transit_planet_radius_scale={:}.png".format(scale))

	plt.figure(5, figsize=(24,16), dpi=100)
	plt.hist(period[has_transit][has_min_transits], range=[0, 100], bins=40, normed=1, alpha=0.6)
	plt.xlabel(r"Planet Period [days]", fontsize=24)
	plt.ylabel(r"Frequency", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "transit_planet_period.png")

	plt.figure(6, figsize=(24,18), dpi=100)
	plt.scatter(planet_radius[has_transit][has_min_transits], period[has_transit][has_min_transits], color="blue", s=10)
	plt.xlabel(r"Planet Radius [$R_\oplus$]", fontsize=24)
	plt.ylabel(r"Planet Period [days]", fontsize=24)
	plt.ylim([0, 100])
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "transit_period_radius.png")
	
	plt.figure(7, figsize=(24,16), dpi=100)
	plt.hist(t_duration, bins=20, normed=1, alpha=0.6)
	plt.xlabel(r"Transit Duration [hours]", fontsize=24)
	plt.ylabel(r"Frequency", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.savefig("figures/" + "transit_duration.png")

	plt.figure(8, figsize=(24,16), dpi=100)
	plt.scatter(logg[has_planet][has_transit][has_min_transits][is_detectable], 
		(planet_radius[has_transit][has_min_transits][is_detectable]*Rearth) / Rmin[is_detectable], 
		s=10, color='red', label="Detectable")
	plt.scatter(logg[has_planet][has_transit][has_min_transits][~is_detectable], 
		(planet_radius[has_transit][has_min_transits][~is_detectable]*Rearth) / Rmin[~is_detectable], 
		s=10, color='blue', label="Not detectable")
	plt.xlabel(r"log $g$ (g $cm^{-2}$)", fontsize=24)
	plt.ylabel(r"$R_{p}$ / $R_{min}$", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.legend(fontsize=24)
	plt.savefig("figures/" + "rmin_logg.png")

	plt.close("all")

# Plots the histograms for the distribution of results from the multiple runs of planet seeding
def plot_result_hist(planet_rates):
	print_top = 1
	for rate in planet_rates:
		try:
			planets, transits, detections = np.loadtxt("data/planet_rate_"+str(rate)+".dat", unpack=True)
		except NameError:
			print("No result file with rate {:}".format(rate))
			continue

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
			plt.xticks=(bins)
			#plt.xticks(np.linspace(0,100,num=100/4+1))
		else:
			bins = 20
		plt.hist(detections, bins=bins, normed=1, alpha=0.6, label="Planet rate - "+str(int(rate*100))+"%")
		plt.xlabel(r"Number of planet detections", fontsize=24)
		plt.ylabel(r"Frequency", fontsize=24)
		plt.tick_params(labelsize=24)
		plt.legend(fontsize=24)
		plt.savefig("figures/hist_results/" + "hist_detections_"+str(rate)+".png")

		plt.close("all")
		
		if verbose:
			if print_top:
				print("{:^11} | {:^11} | {:^11}".format("Planet rate", "Median", "Std. Dev."))
				print_top = 0
			print("{:^11}   {:^11.1f}   {:^11.4f}".format(str(int(rate*100)) + " %", np.median(detections), np.std(detections)))

# Plot the impact of each noise component in function of the logg of the star
def plot_noise_function(logg, shotnoise, gran_noise, noise, has_planet, has_transit, has_min_transits, is_detectable):
	plt.figure(13, figsize=(24,16), dpi=100)
	plt.scatter(logg[has_planet][has_transit][has_min_transits][is_detectable], abs(shotnoise[is_detectable]-noise[is_detectable]) / noise[is_detectable], 
		s=20, color='red', label="Detectable")
	plt.scatter(logg[has_planet][has_transit][has_min_transits][~is_detectable], abs(shotnoise[~is_detectable]-noise[~is_detectable]) / noise[~is_detectable], 
		s=20, color='blue', label="Not detectable")
	plt.xlabel(r"log $g$ (g $cm^{-2}$)", fontsize=24)
	plt.ylabel(r"Contribution of shot noise in total noise", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.legend(fontsize=24)
	plt.savefig("figures/" + "shotnoise_logg.png")

	plt.figure(14, figsize=(24,16), dpi=100)
	plt.scatter(logg[has_planet][has_transit][has_min_transits][is_detectable], abs(gran_noise[is_detectable]-noise[is_detectable]) / noise[is_detectable], 
		s=20, color='red', label="Detectable")
	plt.scatter(logg[has_planet][has_transit][has_min_transits][~is_detectable], abs(gran_noise[~is_detectable]-noise[~is_detectable]) / noise[~is_detectable], 
		s=20, color='blue', label="Not detectable")
	plt.xlabel(r"log $g$ (g $cm^{-2}$)", fontsize=24)
	plt.ylabel(r"Contribution of granulation noise in total noise", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.legend(fontsize=24)
	plt.savefig("figures/" + "grannoise_logg.png")

	plt.figure(15, figsize=(24,16), dpi=100)
	plt.scatter(logg[has_planet][has_transit][has_min_transits][is_detectable], np.log10(gran_noise[is_detectable] / shotnoise[is_detectable]), 
		s=20, color='red', label="Detectable")
	plt.scatter(logg[has_planet][has_transit][has_min_transits][~is_detectable], np.log10(gran_noise[~is_detectable] / shotnoise[~is_detectable]), 
		s=20, color='blue', label="Not detectable")
	plt.xlabel(r"log $g$ (g $cm^{-2}$)", fontsize=24)
	plt.ylabel(r"Granulation noise / Shot noise", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.legend(fontsize=24)
	plt.savefig("figures/" + "gran_shot_noise_logg.png")

	plt.figure(16, figsize=(24,16), dpi=100)
	plt.scatter(logg[has_planet][has_transit][has_min_transits], gran_noise, 
		s=25, color='red', label="Granulation noise, Det")
	plt.scatter(logg[has_planet][has_transit][has_min_transits], shotnoise, 
		s=25, color='blue', label="Shot noise")
	plt.xlabel(r"log $g$ (g $cm^{-2}$)", fontsize=24)
	plt.ylabel(r"Noise", fontsize=24)
	plt.tick_params(labelsize=24)
	plt.legend(fontsize=24)
	plt.savefig("figures/" + "gran_shot_noise_logg.png")

	plt.close("all")

# Runs miltiple times the planet seeding routine for diferent rates and saves the results in a file
# Then it plots the histograms of those results
def multi_seeding(planet_rates, num_iter=300):
	for rate in planet_rates:
		for i in range(num_iter):
			planet_seeding(planet_rate=rate, write_output=True)
			print("Iterations {:}/{:}".format(i+1, num_iter), end="\r", flush=True)
	plot_result_hist()


if __name__ == "__main__":
	#planet_rates = [0.1, 0.05, 0.01]
	#multi_seeding(planet_rates, num_iter=500)
	planet_seeding(0.01, verbose=1, plot_noise=1)