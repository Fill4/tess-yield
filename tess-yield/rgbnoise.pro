; rough SNRs for giants orbiting giants
; rstar & mstar in solar units, rplanet in Rjup, period in days
function rgbnoise,rstar,mstar,rplanet,period

r_sun = double(6.9599e10)		; cm
m_sun = double(1.9891e33)		; g
gconst = double(6.6726e-8)
r_jup = double(71.398e8)

; all this assumes e=0, b=0
semia = ( ((period*86400D)^2*gconst*mstar*m_sun)/(4D*!DPI^2))^(1D/3D)
tdur = (rstar*r_sun*period)/(!DPI*semia)
depth = (rplanet*r_jup/(rstar*r_sun))^2D
;print,rplanet,rstar,depth

; pre-calculated 2nd order polynomials of rms versus logg for trial durations
restore,'noisecoeffs.idl'
res=dblarr(6)
logg=alog10(gconst*mstar*m_sun/(rstar*r_sun^2.))
for q=0.,n_elements(res)-1 do res[q]=poly(logg,coeffs[q,*])

; interpolate to target duration
tdurs=[0.1,0.5,1.0,1.5,2.0,2.5]
rms=interpol(res,tdurs,tdur)
  
ndata=85.
ntr = ndata/period
snr=(depth/rms) * sqrt(ntr)
;print,'snr:',snr

return,snr

end