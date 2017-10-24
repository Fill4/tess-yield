#!/usr/bin/env python

import numpy as np 
import sys, os
import ebf
import gxutil
import matplotlib.pyplot as plt

galaxia = ebf.read('data/tess.ebf','/')
(sample_index, g_lon, g_lat, ec_lon, ec_lat, ra, dec, mass, radius, age, lum, teff, logg, feh, observed_days_med, observed_days_min,
	ubv_u, ubv_b, ubv_v, ubv_r, ubv_i, ubv_j, ubv_h, ubv_k) = np.loadtxt("data/star_sample_complete.dat", unpack=True)

ubv_i_abs = ubv_i

gxutil.abs2app(galaxia, corr=True)
ubv_u = galaxia["ubv_u"][sample_index.astype(int)]
ubv_b = galaxia["ubv_b"][sample_index.astype(int)]
ubv_v = galaxia["ubv_v"][sample_index.astype(int)]
ubv_r = galaxia["ubv_r"][sample_index.astype(int)]
ubv_i = galaxia["ubv_i"][sample_index.astype(int)]
ubv_j = galaxia["ubv_j"][sample_index.astype(int)]
ubv_h = galaxia["ubv_h"][sample_index.astype(int)]
ubv_k = galaxia["ubv_k"][sample_index.astype(int)]

plt.figure(1)
plt.hist(galaxia["ubv_i"], bins=100)
#plt.xlim([8, 18])
#plt.savefig("corr.png")
plt.show()

# Used for testing