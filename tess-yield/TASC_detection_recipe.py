"""
Detection Recipe - 12.0.4.17

References:
(1)  'Asteroseismic detection predictions: TESS' by Chaplin (2015)
(2)  'On the use of empirical bolometric corrections for stars' by Torres (2010)
(3)  'The amplitude of solar oscillations using stellar techniques' by Kjeldson (2008)
(4)  'An absolutely calibrated Teff scale from the infrared flux method'
	 by Casagrande (2010) table 4
(5)  'Characterization of the power excess of solar-like oscillations in red giants with Kepler'
	 by Mosser (2011)
(6)  'Predicting the detectability of oscillations in solar-type stars observed by Kepler'
	 by Chaplin (2011)
(7)  'The connection between stellar granulation and oscillation as seen by the Kepler mission'
	 by Kallinger et al (2014)
(8)  'The Transiting Exoplanet Survey Satellite: Simulations of Planet Detections and
	 Astrophysical False Positives' by Sullivan et al. (2015)
(9)  Astropysics module at https://pythonhosted.org/Astropysics/coremods/coords.html
(10) Bill Chaplin's calc_noise IDL procedure for TESS.
(11) Bill Chaplin's soldet6 IDL procedure to calculate the probability of detecting
	 oscillations with Kepler.
(12) Coordinate conversion at https://ned.ipac.caltech.edu/forms/calculator.html
(13) Bedding 1996
(14) 'The Asteroseismic potential of TESS' by Campante et al. 2016
"""


import numpy as np
from itertools import groupby
from operator import itemgetter
import sys
import pandas as pd
from scipy import stats
import warnings
warnings.simplefilter("ignore")


def bv2teff(b_v):
	# from Torres 2010 table 2. Applies to MS, SGB and giant stars
	# B-V limits from Flower 1996 fig 5
	a = 3.979145106714099
	b = -0.654992268598245
	c = 1.740690042385095
	d = -4.608815154057166
	e = 6.792599779944473
	f = -5.396909891322525
	g = 2.192970376522490
	h = -0.359495739295671
	lteff = a + b*b_v + c*(b_v**2) + d*(b_v**3) + e*(b_v**4) + f*(b_v**5) + g*(b_v**6) + h*(b_v**7)
	teff = 10.0**lteff

	return teff


# from F Pijpers 2003. BCv values from Flower 1996 polynomials presented in Torres 2010
# Av is a keword argument. If reddening values not available, ignore it's effect
def Teff2bc2lum(teff, parallax, parallax_err, vmag, Av=0):
    lteff = np.log10(teff)
    BCv = np.full(len(lteff), -100.5)

    BCv[lteff<3.70] = (-0.190537291496456*10.0**5) + \
    (0.155144866764412*10.0**5*lteff[lteff<3.70]) + \
    (-0.421278819301717*10.0**4.0*lteff[lteff<3.70]**2.0) + \
    (0.381476328422343*10.0**3*lteff[lteff<3.70]**3.0)

    BCv[(3.70<lteff) & (lteff<3.90)] = (-0.370510203809015*10.0**5) + \
    (0.385672629965804*10.0**5*lteff[(3.70<lteff) & (lteff<3.90)]) + \
    (-0.150651486316025*10.0**5*lteff[(3.70<lteff) & (lteff<3.90)]**2.0) + \
    (0.261724637119416*10.0**4*lteff[(3.70<lteff) & (lteff<3.90)]**3.0) + \
    (-0.170623810323864*10.0**3*lteff[(3.70<lteff) & (lteff<3.90)]**4.0)

    BCv[lteff>3.90] = (-0.118115450538963*10.0**6) + \
    (0.137145973583929*10.0**6*lteff[lteff > 3.90]) + \
    (-0.636233812100225*10.0**5*lteff[lteff > 3.90]**2.0) + \
    (0.147412923562646*10.0**5*lteff[lteff > 3.90]**3.0) + \
    (-0.170587278406872*10.0**4*lteff[lteff > 3.90]**4.0) + \
    (0.788731721804990*10.0**2*lteff[lteff > 3.90]**5.0)

    u = 4.0 + 0.4 * 4.73 - 2.0 * np.log10(parallax) - 0.4 * (vmag - Av + BCv)
    lum = 10**u  # in solar units
    e_lum = (2.0 / parallax * 10**u)**2 * parallax_err**2
    e_lum = np.sqrt(e_lum)

    return lum, e_lum


# calculate seismic parameters
def seismicParameters(teff, lum):

	# solar parameters
	teff_solar = 5777.0 # Kelvin
	teffred_solar = 8907.0 #in Kelvin
	numax_solar = 3090.0 # in micro Hz
	dnu_solar = 135.1 # in micro Hz

	cadence = 120 # in s
	vnyq = (1.0 / (2.0*cadence)) * 10**6 # in micro Hz
	teffred = teffred_solar*(lum**-0.093) # from (6) eqn 8. red-edge temp
	rad = lum**0.5 * ((teff/teff_solar)**-2) # Steffan-Boltzmann law
	numax = numax_solar*(rad**-1.85)*((teff/teff_solar)**0.92) # from (14)

	return cadence, vnyq, rad, numax, teffred, teff_solar, teffred_solar, numax_solar, dnu_solar


# no coordinate conversion before calculating tess field observing time. Only
# works with ecliptic coordinates
def tess_field_only(e_lng, e_lat):

    # create a list to append all of the total observing times 'T' in the TESS field to
    T = [] # units of sectors (0-13)

    # create a list to append all of the maximum contiguous observations to
    max_T = [] # units of sectors (0-13)

    for star in range(len(e_lng)):

        # 'n' defines the distance between each equidistant viewing sector in the TESS field.
        n = 360.0/13

        # Define a variable to count the total number of sectors a star is observed in.
        counter = 0

        # Define a variable to count all of the observations for each star.
        # Put each observation sector into sca separately in order to find the largest number
        # of contiguous observations for each star.
        sca = []

        # 'ranges' stores all of the contiguous observations for each star.
        ranges = []

        # Defines the longitude range of the observing sectors at the inputted stellar latitude
        lngrange = 24.0/abs(np.cos(np.radians(e_lat[star])))
        if lngrange>=360.0:
        	lngrange=360.0

        # if the star is in the northern hemisphere:
        if e_lat[star] >= 0.0:

        	# For each viewing sector.
        	for i in range(1,14):

        		# Define an ra position for the centre of each sector in increasing longitude.
        		# if a hemisphere has an overshoot, replace 0.0 with the value.
        		a = 0.0+(n*(i-1))

        		# calculate the distances both ways around the
        		# circle between the star and the centre of the sector.
        		# The smallest distance is the one that should be used
        		# to see if the star lies in the observing sector.
        		d1 = abs(e_lng[star]-a)
        		d2 = (360.0 - abs(e_lng[star]-a))
        		if d1>d2:
        			d1 = d2

        		# if the star is in the 'overshoot' region for some sectors, calculate d3 and d4;
        		# the distances both ways around the circle bwtween the star and the centre of the
        		# 'overshooting past the pole' region of the sector.
        		# The smallest distance is the one that should be used
        		# to see if the star lies in the observing sector.
        		# the shortest distances between the centre of the sector and star, and the sector's
        		# overshoot and the star should add to 180.0 apart (i.e d1+d3=180.0)
        		d3 = abs(e_lng[star] - (a+180.0)%360.0)
        		d4 = 360.0 - abs(e_lng[star] - (a+180.0)%360.0)
        		if d3>d4:
        			d3 = d4

        		# check if a star lies in the field of that sector.
        		if (d1<=lngrange/2.0 and 6.0<=e_lat[star]) or (d3<=lngrange/2.0 and 78.0<=e_lat[star]):
        			counter += 1
        			sca = np.append(sca, i)
        		else:
        			pass

        # if the star is in the southern hemisphere:
        if e_lat[star] < 0.0:

        	# For each viewing sector.
        	for i in range(1,14):

        		# Define an ra position for the centre of each sector in increasing longitude.
        		# if a hemisphere has an overshoot, replace 0.0 with the value.
        		a = 0.0+(n*(i-1))

        		# calculate the distances both ways around the
        		# circle between the star and the centre of the sector.
        		# The smallest distance is the one that should be used
        		# to see if the star lies in the observing sector.
        		d1 = abs(e_lng[star]-a)
        		d2 = (360 - abs(e_lng[star]-a))
        		if d1>d2:
        			d1 = d2

        		# if the star is in the 'overshoot' region for some sectors, calculate d3 and d4;
        		# the distances both ways around the circle between the star and the centre of the
        		# 'overshooting past the pole' region of the sector.
        		# The smallest distance of the 2 is the one that should be used
        		# to see if the star lies in the observing sector.
        		d3 = abs(e_lng[star]  - (a+180.0)%360.0)
        		d4 = (360 - abs(e_lng[star] - (a+180.0)%360.0))
        		if d3>d4:
        			d3 = d4

        		# check if a star lies in the field of that sector.
        		if (d1<=lngrange/2.0 and -6.0>=e_lat[star]) or (d3<=lngrange/2.0 and -78.0>=e_lat[star]):
        			counter += 1
        			sca = np.append(sca, i)
        		else:
        			pass

        if len(sca) == 0:
        	ranges = [0]

        else:
        	for k,g in groupby(enumerate(sca), lambda i_x:i_x[0]-i_x[1]):
        		group = map(itemgetter(1), g)

        		if np.array(group).sum() !=0:
        			ranges.append([len(list(group))])

        T=np.append(T, counter)
        max_T = np.append(max_T, np.max(np.array(ranges)))

    return T, max_T


def calc_noise(imag, exptime, teff, e_lng = 0, e_lat = 30, g_lng = 96, g_lat = -30, subexptime = 2.0, npix_aper = 10, \
frac_aper = 0.76, e_pix_ro = 10, geom_area = 60.0, pix_scale = 21.1, sys_limit = 0):

    omega_pix = pix_scale**2.0
    n_exposures = exptime/subexptime

    # electrons from the star
    megaph_s_cm2_0mag = 1.6301336 + 0.14733937*(teff-5000.0)/5000.0
    e_star = 10.0**(-0.4*imag) * 10.0**6 * megaph_s_cm2_0mag * geom_area * exptime * frac_aper
    e_star_sub = e_star*subexptime/exptime

    # e/pix from zodi
    dlat = (abs(e_lat)-90.0)/90.0
    vmag_zodi = 23.345 - (1.148*dlat**2.0)
    e_pix_zodi = 10.0**(-0.4*(vmag_zodi-22.8)) * (2.39*10.0**-3) * geom_area * omega_pix * exptime

    # e/pix from background stars
    dlat = abs(g_lat)/40.0*10.0**0

    dlon = g_lng
    q = np.where(dlon>180.0)
    if len(q[0])>0:
    	dlon[q] = 360.0-dlon[q]

    dlon = abs(dlon)/180.0*10.0**0
    p = [18.97338*10.0**0, 8.833*10.0**0, 4.007*10.0**0, 0.805*10.0**0]
    imag_bgstars = p[0] + p[1]*dlat + p[2]*dlon**(p[3])
    e_pix_bgstars = 10.0**(-0.4*imag_bgstars) * 1.7*10.0**6 * geom_area * omega_pix * exptime

    # compute noise sources
    noise_star = np.sqrt(e_star) / e_star
    noise_sky  = np.sqrt(npix_aper*(e_pix_zodi + e_pix_bgstars)) / e_star
    noise_ro   = np.sqrt(npix_aper*n_exposures)*e_pix_ro / e_star
    noise_sys  = 0.0*noise_star + sys_limit/(1*10.0**6)/np.sqrt(exptime/3600.0)

    noise1 = np.sqrt(noise_star**2.0 + noise_sky**2.0 + noise_ro**2.0)
    noise2 = np.sqrt(noise_star**2.0 + noise_sky**2.0 + noise_ro**2.0 + noise_sys**2.0)

    return noise2


# calculate the granulation at a set of frequencies from (7) eqn 2 model F
def granulation(nu0, dilution, a_nomass, b1, b2, vnyq):

    # Divide by dilution squared as it affects stars in the time series.
    # The units of dilution change from ppm to ppm^2 microHz^-1 when going from the
    # time series to frequency. p6: c=4 and zeta = 2*sqrt(2)/pi
    Pgran = (((2*np.sqrt(2))/np.pi) * (a_nomass**2/b1) / (1 + ((nu0/b1)**4)) \
    + ((2*np.sqrt(2))/np.pi) * (a_nomass**2/b2) / (1 + ((nu0/b2)**4))) / (dilution**2)

    # From (9). the amplitude suppression factor. Normalised sinc with pi (area=1)
    eta = np.sinc((nu0/(2*vnyq)))

    # the granulation after attenuation
    Pgran = Pgran * eta**2

    return Pgran, eta


# the total number of pixels used by the highest ranked x number of targets in the tCTL
def pixel_cost(x):
    N = np.ceil(10.0**-5.0 * 10.0**(0.4*(20.0-x)))
    N_tot = 10*(N+10)
    total = np.cumsum(N_tot)

    # want to find: the number of ranked tCTL stars (from highest to lowest rank) that correspond to a pixel cost of 1.4Mpix at a given time
    per_cam = 26*4 # to get from the total pixel cost to the cost per camera at a given time, divide by this
    pix_limit = 1.4e6 # the pixel limit per camera at a given time

    return total[-1], per_cam, pix_limit, N_tot


# detection recipe to find whether a star has an observed solar-like Gaussian mode power excess
def globalDetections(g_lng, g_lat, e_lng, e_lat, imag, \
lum, rad, teff, numax, max_T, teffred, teff_solar, \
teffred_solar, numax_solar, dnu_solar, sys_limit, dilution, vnyq, cadence, vary_beta=False):

	dnu = dnu_solar*(rad**-1.42)*((teff/teff_solar)**0.71) # from (14) eqn 21
	beta = 1.0-np.exp(-(teffred-teff)/1550.0) # beta correction for hot solar-like stars from (6) eqn 9.
	if isinstance(teff, float):  # for only 1 star
	    if (teff>=teffred):
	        beta = 0.0
	else:
	    beta[teff>=teffred] = 0.0

	# to remove the beta correction, set Beta=1
	if vary_beta == False:
	    beta = 1.0

	# modified from (6) eqn 11. Now consistent with dnu proportional to numax^0.77 in (14)
	amp = 0.85*2.5*beta*(rad**1.85)*((teff/teff_solar)**0.57)

	# From (5) table 2 values for delta nu_{env}. env_width is defined as +/- some value.
	env_width = 0.66 * numax**0.88
	env_width[numax>100.] = numax[numax>100.]/2.  # from (6) p12

	total, per_cam, pix_limit, npix_aper = pixel_cost(imag)

	noise = calc_noise(imag=imag, teff=teff, exptime=cadence, e_lng=e_lng, e_lat=e_lat, \
	g_lng=g_lng, g_lat=g_lat, sys_limit=sys_limit, npix_aper=npix_aper)
	noise = noise*10.0**6 # total noise in units of ppm

	a_nomass = 0.85 * 3382*numax**-0.609 # multiply by 0.85 to convert to redder TESS bandpass.
	b1 = 0.317 * numax**0.970
	b2 = 0.948 * numax**0.992

	# call the function for the real and aliased components (above and below vnyq) of the granulation
	# the order of the stars is different for the aliases so fun the function in a loop
	Pgran, eta = granulation(numax, dilution, a_nomass, b1, b2, vnyq)
	Pgranalias = np.zeros(len(Pgran))
	etaalias = np.zeros(len(eta))

	# if vnyq is 1 fixed value
	if isinstance(vnyq, float):
	    for i in range(len(numax)):

	    	if numax[i] > vnyq:
	    		Pgranalias[i], etaalias[i] = granulation((vnyq - (numax[i] - vnyq)), \
	    		dilution, a_nomass[i], b1[i], b2[i], vnyq)

	    	elif numax[i] < vnyq:
	    		Pgranalias[i], etaalias[i] = granulation((vnyq + (vnyq - numax[i])), \
	    		dilution, a_nomass[i], b1[i], b2[i], vnyq)

	# if vnyq varies for each star
	else:
	    for i in range(len(numax)):

	    	if numax[i] > vnyq[i]:
	    		Pgranalias[i], etaalias[i] = granulation((vnyq[i] - (numax[i] - vnyq[i])), \
	    		dilution, a_nomass[i], b1[i], b2[i], vnyq[i])

	    	elif numax[i] < vnyq[i]:
	    		Pgranalias[i], etaalias[i] = granulation((vnyq[i] + (vnyq[i] - numax[i])), \
	    		dilution, a_nomass[i], b1[i], b2[i], vnyq[i])

	Pgrantotal = Pgran + Pgranalias

	ptot = (0.5*2.94*amp**2.*((2.*env_width)/dnu)*eta**2.) / (dilution**2.)
	Binstr = 2.0 * (noise)**2. * cadence*10**-6.0 # from (6) eqn 18
	bgtot = ((Binstr + Pgrantotal) * 2.*env_width) # units are ppm**2

	snr = ptot/bgtot # global signal to noise ratio from (11)
	fap = 0.05 # false alarm probability
	pdet = 1.0 - fap
	pfinal = np.full(rad.shape[0], -99)
	idx = np.where(max_T != 0) # calculate the indexes where T is not 0
	tlen=max_T[idx]*27.4*86400.0 # the length of the TESS observations in seconds

	bw=1.0 * (10.0**6.0)/tlen
	nbins=(2.*env_width[idx]/bw).astype(int) # from (11)
	snrthresh = stats.chi2.ppf(pdet, 2.0*nbins) / (2.0*nbins) - 1.0

	pfinal[idx] = stats.chi2.sf((snrthresh+1.0) / (snr[idx]+1.0)*2.0*nbins, 2.*nbins)
	return pfinal, snr, dnu # snr is needed in TESS_telecon2.py


def BV2VI(bv, vmag, g_mag_abs):

	whole = pd.DataFrame(data={'B-V': bv, 'Vmag': vmag, 'g_mag_abs': g_mag_abs, 'Ai': 0})

	# Mg: empirical relation from Tiago to separate dwarfs from giants
	# note: this relation is observational; it was made with REDDENED B-V and g_mag values
	whole['Mg'] = 6.5*whole['B-V'] - 1.8

	# B-V-to-teff limits from (6) fig 5
	whole = whole[(whole['B-V'] > -0.4) & (whole['B-V'] < 1.7)]
	print(whole.shape, 'after B-V cuts')

	# B-V limits for dwarfs and giants, B-V conditions from (1)
	# if a star can't be classified as dwarf or giant, remove it
	condG = (whole['B-V'] > -0.25) & (whole['B-V'] < 1.75) & (whole['Mg'] > whole['g_mag_abs'])
	condD1 = (whole['B-V'] > -0.23) & (whole['B-V'] < 1.4) & (whole['Mg'] < whole['g_mag_abs'])
	condD2 = (whole['B-V'] > 1.4) & (whole['B-V'] < 1.9) & (whole['Mg'] < whole['g_mag_abs'])

	whole = pd.concat([whole[condG], whole[condD1], whole[condD2]], axis=0)
	print(whole.shape, 'after giant/dwarf cuts')
	whole['V-I'] = 100. # write over these values for dwarfs and giants separately

	# coefficients for giants and dwarfs
	cg = [-0.8879586e-2, 0.7390707, 0.3271480, 0.1140169e1, -0.1908637, -0.7898824,
		0.5190744, 0.5358868]
	cd1 = [0.8906590e-1, 0.1319675e1, 0.4461807, -0.1188127e1, 0.2465572, 0.8478627e1,
		0.1046599e2, 0.3641226e1]
	cd2 = [-0.5421588e2, 0.8011383e3, -0.4895392e4, 0.1628078e5, -0.3229692e5,
		0.3939183e5, -0.2901167e5, 0.1185134e5, -0.2063725e4]

	# calculate (V-I) for giants
	x = whole['B-V'][condG] - 1
	y = (cg[0] + cg[1]*x + cg[2]*(x**2) + cg[3]*(x**3) + cg[4]*(x**4) +\
		cg[5]*(x**5) + cg[6]*(x**6) + cg[7]*(x**7))
	whole['V-I'][condG] = y + 1
	x, y = [[] for i in range(2)]

	# calculate (V-I) for dwarfs (1st B-V range)
	x = whole['B-V'][condD1] - 1
	y = (cd1[0] + cd1[1]*x + cd1[2]*(x**2) + cd1[3]*(x**3) + cd1[4]*(x**4) +\
		cd1[5]*(x**5) + cd1[6]*(x**6) + cd1[7]*(x**7))
	whole['V-I'][condD1] = y + 1
	x, y = [[] for i in range(2)]

	# calculate (V-I) for dwarfs (2nd B-V range)
	x = whole['B-V'][condD2] - 1
	y = (cd2[0] + cd2[1]*x + cd2[2]*(x**2) + cd2[3]*(x**3) + cd2[4]*(x**4) +\
		cd2[5]*(x**5) + cd2[6]*(x**6) + cd2[7]*(x**7) + cd2[8]*(x**8))
	whole['V-I'][condD2] = y + 1
	x, y = [[] for i in range(2)]

	# calculate Imag from V-I and reredden it
	whole['Imag'] = whole['Vmag']-whole['V-I']
	whole['Imag_reddened'] = whole['Imag'] + whole['Ai']

	"""
	# make Teff, luminosity, Plx and ELat cuts to the data
	whole = whole[(whole['teff'] < 7700) & (whole['teff'] > 4300) & \
		(whole['Lum'] > 0.3) & (whole['lum_D'] < 50) & ((whole['e_Plx']/whole['Plx']) < 0.5) \
		& (whole['Plx'] > 0.) & ((whole['ELat']<=-6.) | (whole['ELat']>=6.))]
	print(whole.shape, 'after Teff/L/Plx/ELat cuts')
	"""

	whole.drop(['Ai', 'Imag_reddened', 'Mg'], axis=1, inplace=True)

	return whole.as_matrix().T


# make cuts to the data
def cuts(teff, e_teff, metal, e_metal, g_lng, g_lat, e_lng, e_lat, Tmag, e_Tmag, Vmag, e_Vmag, plx, e_plx, lum, star_name):
    d = {'teff':teff, 'e_teff':e_teff, 'metal':metal, 'e_metal':e_metal, 'g_lng':g_lng, 'g_lat':g_lat, 'e_lng':e_lng, 'e_lat':e_lat,
         'Tmag':Tmag, 'e_Tmag':e_Tmag, 'Vmag':Vmag, 'e_Vmag':e_Vmag, 'plx':plx, 'e_plx':e_plx, 'lum':lum, 'star_name':star_name}
    whole = pd.DataFrame(d, columns = ['teff', 'e_teff', 'metal', 'e_metal', 'g_lng', 'g_lat', 'e_lng', 'e_lat', 
                                       'Tmag', 'e_Tmag', 'Vmag', 'e_Vmag', 'plx', 'e_plx', 'lum', 'star_name'])
    whole = whole[(whole['teff'] < 7700.) & (whole['teff'] > 4300.) & (whole['e_teff'] > 0.) & \
		(whole['lum'] > 0.3) & (whole['lum'] < 50.) & ((whole['e_plx']/whole['plx']) < 0.5) & \
		(whole['plx'] > 0.) & ((whole['e_lat']<=-6.) | (whole['e_lat']>=6.)) & \
         (whole['Tmag'] > 3.5) & (whole['e_metal'] > 0.)]
    print(whole.shape, 'after cuts to the data')
    return whole.as_matrix().T


if __name__ == '__main__':

    df = pd.read_csv('files/MAST_Crossmatch_TIC4.csv', header=0,
                     index_col=False)
    data = df.values

    # star_name = data[:, 1]
    teff = pd.to_numeric(data[:, 88])
    # e_teff = pd.to_numeric(data[:, 89])
    # metal = pd.to_numeric(data[:, 92])
    # e_metal = pd.to_numeric(data[:, 93])
    # g_lng = pd.to_numeric(data[:, 48])
    # g_lat = pd.to_numeric(data[:, 49])
    # e_lng = pd.to_numeric(data[:, 50])
    # e_lat = pd.to_numeric(data[:, 51])
    # Tmag = pd.to_numeric(data[:, 84])
    # e_Tmag = pd.to_numeric(data[:, 85])
    Vmag = pd.to_numeric(data[:, 54])
    # e_Vmag = pd.to_numeric(data[:, 55])
    plx = pd.to_numeric(data[:, 45])
    e_plx = pd.to_numeric(data[:, 46])

    lum, e_lum = Teff2bc2lum(teff, plx, e_plx, Vmag)
    df[' Luminosity'] = pd.Series(lum)
    df[' Luminosity Err.'] = pd.Series(e_lum)
    
    # teff, e_teff, metal, e_metal, g_lng, g_lat, e_lng, e_lat, Tmag, e_Tmag, \
    #     Vmag, e_Vmag, plx, e_plx, lum, star_name = cuts(teff, e_teff, metal, e_metal,
    #         g_lng, g_lat, e_lng, e_lat, Tmag, e_Tmag, Vmag, e_Vmag, 
    #             plx, e_plx, lum, star_name)
    
    # make cuts to the data
    df = df[(df[' T_eff'] < 7700.) & (df[' T_eff'] > 4300.) & (df[' T_eff Err.'] > 0.) & \
		(df[' Luminosity'] > 0.3) & (df[' Luminosity'] < 50.) & ((df[' Parallax Err.']/df[' Parallax']) < 0.5) & \
		(df[' Parallax'] > 0.) & ((df[' Ecl. Lat.']<=-6.) | (df[' Ecl. Lat.']>=6.)) & \
         (df[' TESS Mag.'] > 3.5) & (df[' Metallicity Err.'] > 0.)]
    df = df.reset_index(drop=True)
    print(df.shape, 'after cuts to the data')
    
    data = df.values
    teff = pd.to_numeric(data[:, 88])
    lum = pd.to_numeric(data[:, 113])
    Tmag = pd.to_numeric(data[:, 84])
    Vmag = pd.to_numeric(data[:, 54])
    plx = pd.to_numeric(data[:, 45])
    g_lng = pd.to_numeric(data[:, 48])
    g_lat = pd.to_numeric(data[:, 49])
    e_lng = pd.to_numeric(data[:, 50])
    e_lat = pd.to_numeric(data[:, 51])
    
    cadence, vnyq, rad, numax, teffred, teff_solar, teffred_solar, \
        numax_solar, dnu_solar = seismicParameters(teff, lum)

    T, max_T = tess_field_only(e_lng, e_lat)

    pdet, snr, dnu = globalDetections(g_lng=g_lng, g_lat=g_lat, e_lng=e_lng, e_lat=e_lat, \
	    imag=Tmag, lum=lum, rad=rad, teff=teff, numax=numax, max_T=max_T, \
	    teffred=teffred, teff_solar=teff_solar, teffred_solar=teffred_solar, \
	    numax_solar=numax_solar, dnu_solar=dnu_solar, sys_limit=0., dilution=1., \
	    vnyq=vnyq, cadence=cadence, vary_beta=True)
    
    df[' Dnu'] = pd.Series(dnu)
    e_dnu = dnu * 0.021
    df[' Dnu Err.'] = pd.Series(e_dnu)
    df[' nu_max'] = pd.Series(numax)
    e_numax = numax * 0.046
    df[' nu_max Err.'] = pd.Series(e_numax)
    df[' P_det'] = pd.Series(pdet)
    df[' SNR'] = pd.Series(snr)

    # make new cut to the data (pdet > 0.5)
    df = df[df[' P_det'] > 0.5]
    df = df.reset_index(drop=True)
    print(df.shape, 'after new cut to the data (pdet > 0.5)')
    
    # inflating errors on teff and metal
    data = df.values
    e_teff = pd.to_numeric(data[:, 89])
    e_teff = (e_teff**2 + 59**2)**0.5
    df[' T_eff Err.'] = pd.Series(e_teff)
    e_metal = pd.to_numeric(data[:, 93])
    e_metal = (e_metal**2 + 0.062**2)**0.5
    df[' Metallicity Err.'] = pd.Series(e_metal)

    df.to_csv('/Users/Tiago/Work/PostDocBham/KnownHosts_with_TESS/catalog_knownhosts/PARAM_input.csv', index=False)
