"""
The module applies extinction and computes the apparent magntiude,  
when applied to data created by the code Galaxia.
It can aslo be used to compute proper motion and radial velocity.
Copyright (c) 2014 Sanjib Sharma

SETUP
::
In file gxutil.py provide the path to aebv_factor.ebf file.

USAGE
::

>>>data=ebf.read(myfile,'/')
>>>gxutil.abs2app(data,corr=True)
>>>gxutil.append_pm(data)

"""
import ebf
import numpy as np
import sys

def _aebv_factor():
	filename='data/aebv_factor.ebf'
	aebvdata=ebf.read(filename,'/')
	aebv={}
	for myfilter,factor in zip(aebvdata['filter'],aebvdata['aebv_factor']):
		aebv[str(myfilter).lower()]=factor
	return aebv


def append_pm(data):
	"""
	Append proper motion and radial velocity to data.
	Keys mul (arcsec/yr), mub (arcsec/yr) and vr (km/s) 
	added to data
	Args:
		data: galaxia data
	"""
	[vl,vb,vr]=_vxyz2lbr(data['px'],data['py'],data['pz'],data['vx'],data['vy'],data['vz'])
	r=np.sqrt(data['px']*data['px']+data['py']*data['py']+data['pz']*data['pz'])
#    data['pm']=np.sqrt(vl*vl+vb*vb)/(4.74e3*r)
	data['vr']=vr
	data['mul']=vl/(4.74e3*r)
	data['mub']=vb/(4.74e3*r)


def abs2app(data,noext=False,dered=False,corr=False):
	"""
	Apply extinction and computes the apparent magntiude.  
	All magnitude related keys in data are modified.
	It is advisabe to run it with option corr=True to enable 
	low latitude correction to Schlegel maps.
	Args:
		data: galaxia data 
	"""
	dmod=5.0*np.log10(100.0*data['rad'])
	if dered:
		ebv=data['exbv_schlegel']-data['exbv_schlegel_inf']
	else:
		ebv=data['exbv_schlegel']*1.0
	if corr:
		ebv=ebv*_extcorr(data['exbv_schlegel_inf'])

	aebv=_aebv_factor()
	j=0.0
	for temp in data.keys():
		temp1 = "b'" + temp.lower() + "'"
		#print(temp1)
		#print(aebv.keys())
		#sys.exit()
		if temp1 in aebv:
			#print(temp)
			if noext==1:
				data[temp]=data[temp]+dmod
			else: 
				data[temp]=data[temp]+dmod+aebv[temp1]*ebv
			j+=1

	if  j == 0:
		raise RuntimeError('No quantity to add extinction')


def _extcorr(ebv):
	return (1-np.tanh((ebv-0.15)/0.075))*0.5*(1-0.6)+0.6



def _vxyz2lbr(px,py,pz,vx,vy,vz):
	r=np.sqrt(px*px+py*py+pz*pz)
	ind=np.where(r == 0.0)[0]
	r[ind]=1.0
	px=px/r
	py=py/r
	pz=pz/r
	rc=np.sqrt(px*px+py*py)
	ind=np.where(rc == 0.0)[0]
	rc[ind]=1.0
	tm_00=-py/rc; tm_01=-pz*px/rc; tm_02= rc*px/rc
	tm_10= px/rc; tm_11=-pz*py/rc; tm_12= rc*py/rc
	tm_20= 0.0  ; tm_21= rc  ; tm_22= pz
	vl=(vx*tm_00+vy*tm_10+vz*tm_20)
	vb=(vx*tm_01+vy*tm_11+vz*tm_21)
	vr=(vx*tm_02+vy*tm_12+vz*tm_22)
	return vl,vb,vr


def _vlbr2xyz(l,b,r,vl,vb,vr):
	l=np.radians(l)
	b=np.radians(b)
	tm_00=-np.sin(l) ; tm_01=-np.sin(b)*np.cos(l) ; tm_02= np.cos(b)*np.cos(l)
	tm_10= np.cos(l) ; tm_11=-np.sin(b)*np.sin(l) ; tm_12= np.cos(b)*np.sin(l)
	tm_20= 0.0       ; tm_21= np.cos(b)           ; tm_22= np.sin(b)
	vx=vl*tm_00+vb*tm_01+vr*tm_02
	vy=vl*tm_10+vb*tm_11+vr*tm_12
	vz=vl*tm_20+vb*tm_21+vr*tm_22
	# px=r*np.cos(b)*np.cos(l)
	# py=r*np.cos(b)*np.sin(l)
	# pz=r*np.sin(b)
	return vx,vy,vz







