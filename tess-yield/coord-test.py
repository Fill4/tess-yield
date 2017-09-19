#!/usr/bin/env python

import numpy as np
from despyastro.coords import *

tic, t_ra, t_dec, t_g_lon, t_g_lat, t_ec_lon, t_ec_lat = np.loadtxt("data/tic_data.dat", unpack=True)
tic = tic.astype(int)

ec_lon, ec_lat = gal2ec(t_g_lon, t_g_lat)
ra, dec = ec2eq(ec_lon, ec_lat)

ec_comp = np.column_stack((t_ec_lon, ec_lon, t_ec_lat, ec_lat))
print("  true_ec_lon       ec_lon       true_ec_lat     ec_lat")
print(ec_comp)
print("\nMaximum diference in ec_lat: {:}".format(max(abs(ec_lat-t_ec_lat))))
print("Maximum diference in ec_lon: {:}\n".format(max(abs(ec_lon-t_ec_lon))))

eq_comp = np.column_stack((t_ra, ra, t_dec, dec))
print("    true_ra           ra         true_dec         dec")
print(eq_comp)
print("\nMaximum diference in ra: {:}".format(max(abs(ra-t_ra))))
print("Maximum diference in dec: {:}".format(max(abs(dec-t_dec))))