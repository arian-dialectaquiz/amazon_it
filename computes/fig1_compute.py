#############################
"""
fig01_compute.py
=================================================================
Figure 1  -  The margin and the controls that make the experiment
			 interpretable
Amazon shelf internal tide manuscript, version 2

Computes and stores everything panels (a) to (e) need:

  (a) bathymetry, isobaths, generation sites, moorings, sections
  (b) barotropic M2 control, the difference between the runs in the
	  depth averaged M2 current and in the M2 elevation
  (c) surface trapping validity, h / hp with the h = 3 hp contour
  (d) points per mode one wavelength against depth in both runs, with
	  the fraction of cells resolved in BOTH
  (e) criticality alpha along the 200 m isobath in both runs, with
	  the difference in the inset

Output:  fig01_data.nc   (single dataset, read by fig01_plot.py)

Why these five panels
---------------------
The manuscript rests on three premises, and this figure is where each
of them is tested rather than asserted.

  1. The two runs share the barotropic tide. Conversion is
	 C = <p_bc(-H) w_bt(-H)> with w_bt = -u_bt . grad H, so if u_bt is
	 common to the runs then every conversion anomaly belongs to the
	 baroclinic pressure at the bed. Panel (b) measures that.
  2. The lens is surface trapped where the criticality argument is
	 applied. A 20 to 50 m lens does reach the bed over the mid shelf,
	 so the geometric argument holds only where h >> hp. Panel (c)
	 draws the boundary of the region where it does.
  3. Both runs resolve the wave they carry. The plume lengthens the
	 mode, so CTRL is the binding constraint, and the cross run
	 comparisons are restricted to the columns where both clear the
	 Hallberg (2013) criterion. Panel (d) defines that subdomain.

Panel (e) then shows that the criticality along the break is common to
the two experiments, which is the control that removes the geometry
from the explanation.

No discharge series is read or stored. The river forcing was built by
interpolation in time and the file does not carry the values a
discharge panel would need, so the experimental design is documented in
the Methods and by the ctrl_discharge attribute alone.

The spring and neap envelope of the regional sea surface height IS
computed and stored, even though no panel of this figure draws it,
because fig03 marks its block resolved series with it and the two
figures should share one definition.

Physics notes
-------------
* Vertical modes solve the Sturm-Liouville problem with the FREE
  SURFACE condition Phi(0) = (c^2/g) dPhi/dz, retained rather than the
  rigid lid because the shelf is shallow and the surface term is not
  negligible there (Tchilibou et al. 2022). Mode 1 is index 1.
* Modes come from the WINDOW MEAN stratification, the standard
  background state definition for a modal decomposition.
* Criticality uses NEAR BOTTOM N2 by construction (Baines 1982), which
  is exactly why the plume is expected to leave it almost untouched.
  A null result in panel (e) is the control, not a failed diagnostic.
=================================================================
"""

import warnings
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from scipy.signal import butter, filtfilt, hilbert

try:
	import xroms
	HAS_XROMS = True
except ImportError:
	HAS_XROMS = False
	warnings.warn('xroms not found, falling back to plain xarray + gsw')

try:
	import gsw
	HAS_GSW = True
except ImportError:
	HAS_GSW = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # headless, only to extract contours
from matplotlib.path import Path as MplPath


# =====================================================================
# CONFIGURATION
# =====================================================================
REALISTIC = '/data1/amasseds_run/new_mean/roms_am2_new_mean.nc'
CONTROL   = '/data1/amasseds_run/const/roms_am2_redo.nc'
GRIDFILE  = '/home/rafaela/ROMS/pyroms_tools_arian/data/grid/pca_grd_river_open.nc'

OUTFILE = 'fig01_data.nc'

# Design value of the constant discharge experiment, carried as metadata
# so the output file documents the experiment. Nothing is read from it.
CTRL_Q = 2.0e5             # m3 s-1

# Hydrological windows. The record runs 1990-01-05 to 1990-07-05, so the
# decreasing window closes on the last day of output and not at the end
# of July. Quote it that way in the manuscript.
WINDOWS = {
	'rising':     ('1990-01-05', '1990-02-28'),
	'peak':       ('1990-03-01', '1990-05-31'),
	'decreasing': ('1990-06-01', '1990-07-05'),
}

MOORINGS = {
	'M1': (-50.3133, 3.0752),
	'M2': (-49.9372, 3.3853),
	'M3': (-49.6225, 4.0715),
}

# Generation sites A to F, first named by Magalhaes et al. (2016) and
# reproduced in the slope maps of Tchilibou et al. (2022) and Assene et
# al. (2024). Digitised, so each carries about +/- 0.3 degrees of reading
# error, removed by snapping onto the model 200 m isobath below.
GEN_SITES = {
	'A': (-46.60,  0.45),
	'B': (-43.80, -1.98),
	'C': (-45.20, -0.23),
	'D': (-47.60,  1.55),
	'E': (-50.50,  4.23),
	'F': (-51.90,  5.47),
}
GEN_SITES_MAIN = ('A', 'B')
SNAP_SITES_TO_BREAK = True

SECTIONS = {
	'S1': (-50.90,  2.40, -47.60,  4.60),
	'S2': (-48.60,  0.60, -45.40,  2.90),
	'S3': (-45.90, -0.90, -43.00,  1.10),
}

BREAK_ISOBATH   = 200.      # m, path along which alpha is reported
SHELF_MAX_DEPTH = 250.      # m, shelf / slope partition

# constants
G        = 9.81
RHO0     = 1025.0
OMEGA_M2 = 2*np.pi/(12.4206012*3600.)
OMEGA_S2 = 2*np.pi/(12.0*3600.)
T_M2     = 12.4206012*3600.
OMEGA_E  = 7.2921e-5

NMODES      = 4         # modes 0..3, mode 0 is barotropic with free surface
N2_FLOOR    = 1e-9
NBOT_METRES = 50.       # m, thickness over which near bottom N2 is averaged
CHUNK_COLS  = 4000

PPW_TARGET  = (8., 12.)     # Hallberg (2013)
PPW_MIN     = 8.            # the value that defines the resolved subdomain
C1_MIN      = 0.05          # m s-1, below this there is no resolvable mode
TRAP_RATIO  = 3.0           # h > TRAP_RATIO*hp for the criticality argument

# Barotropic control. Frequencies fitted over the whole record, which is
# six months and therefore separates M2 from S2 (Rayleigh 14.77 d).
BT_FREQS = (OMEGA_M2, OMEGA_S2)
BT_TCHUNK = 400             # time records per pass
DO_UBT = True               # False skips the depth averaged current fit

# depth bins for panel (d)
DEPTH_BINS = np.logspace(np.log10(8.), np.log10(3000.), 26)


# =====================================================================
# HELPERS
# =====================================================================
def open_run(path, chunks={'ocean_time': 50}):
	"""Open a ROMS history file, with or without xroms."""
	if HAS_XROMS:
		try:
			out = xroms.open_netcdf(path, chunks=chunks)
		except TypeError:
			out = xroms.open_netcdf(path)
		ds = out[0] if isinstance(out, tuple) else out
	else:
		ds = xr.open_dataset(path, chunks=chunks)
	return ds


def compute_N2(temp, salt, z_rho, z_w, lon, lat):
	"""
	Buoyancy frequency squared at w points from window mean T and S.

	Returned on the full w grid, bottom and surface extrapolated from the
	adjacent interior points, which the free surface mode solve needs.
	"""
	if HAS_GSW:
		p  = gsw.p_from_z(z_rho, lat)
		SA = gsw.SA_from_SP(salt, p, lon, lat)
		CT = gsw.CT_from_pt(SA, temp)
		rho = gsw.rho(SA, CT, 0.0)
	else:
		warnings.warn('gsw not available, using a linear equation of state')
		rho = RHO0*(1 - 1.7e-4*(temp - 10.) + 7.6e-4*(salt - 35.))

	rho = np.asarray(rho); zr = np.asarray(z_rho); zw = np.asarray(z_w)
	drho = np.diff(rho, axis=0)
	dz   = np.diff(zr,  axis=0)
	N2_int = -(G/RHO0)*drho/dz

	shp = (zw.shape[0],) + N2_int.shape[1:]
	N2 = np.empty(shp, dtype=float)
	N2[1:-1] = N2_int
	N2[0]    = N2_int[0]
	N2[-1]   = N2_int[-1]
	return N2


def vertical_modes(N2, zw, nmodes=NMODES, free_surface=True,
				   n2_floor=N2_FLOOR, chunk=CHUNK_COLS):
	"""
	Solve  d2Phi/dz2 + (N2/c2) Phi = 0,  Phi(-H) = 0, with the free
	surface condition Phi(0) = (c2/g) dPhi/dz|_0.

	Weak form  int Phi' Psi' dz = lambda ( int N2 Phi Psi dz + g Phi(0)Psi(0) )
	so both matrices stay symmetric and the problem is standard after a
	diagonal similarity transform.

	Returns c (nmodes, M) and Phi (Nw, nmodes, M), Phi normalised to unit
	maximum and positive in the upper half of the column.
	"""
	Nw, M = N2.shape
	n = Nw - 1 if free_surface else Nw - 2

	c   = np.full((nmodes, M), np.nan)
	Phi = np.full((Nw, nmodes, M), np.nan)
	N2c = np.maximum(N2, n2_floor)

	for s in range(0, M, chunk):
		e  = min(s + chunk, M)
		m  = e - s
		n2 = N2c[:, s:e]
		z  = zw[:, s:e]

		hh = np.diff(z, axis=0)
		bad = (~np.isfinite(hh)).any(axis=0) | (hh <= 0).any(axis=0) \
			  | (~np.isfinite(n2)).any(axis=0)
		hh = np.where(np.isfinite(hh) & (hh > 0), hh, 1.0)

		K  = np.zeros((m, n, n))
		Md = np.zeros((m, n))

		for j in range(Nw - 2):
			hb = hh[j]; ha = hh[j + 1]
			K[:, j, j] = 1.0/hb + 1.0/ha
			if j > 0:
				K[:, j, j - 1] = -1.0/hb
			if j < n - 1:
				K[:, j, j + 1] = -1.0/ha
			Md[:, j] = n2[j + 1]*0.5*(hb + ha)

		if free_surface:
			j = n - 1
			hb = hh[-1]
			K[:, j, j]     = 1.0/hb
			K[:, j, j - 1] = -1.0/hb
			Md[:, j] = n2[-1]*0.5*hb + G

		Md = np.maximum(Md, 1e-12)
		D  = 1.0/np.sqrt(Md)
		A  = K*D[:, :, None]*D[:, None, :]
		A  = 0.5*(A + np.swapaxes(A, 1, 2))

		lam, V = np.linalg.eigh(A)
		lam = np.maximum(lam, 1e-20)
		cc  = 1.0/np.sqrt(lam)
		Vphys = V*D[:, :, None]

		k = min(nmodes, n)
		c[:k, s:e] = cc[:, :k].T

		full = np.zeros((m, Nw, k))
		if free_surface:
			full[:, 1:, :] = Vphys[:, :, :k]
		else:
			full[:, 1:-1, :] = Vphys[:, :, :k]

		amax = np.nanmax(np.abs(full), axis=1, keepdims=True)
		amax[amax == 0] = 1.0
		full = full/amax
		upper = full[:, Nw//2:, :].sum(axis=1)
		full = full*np.sign(np.where(upper == 0, 1.0, upper))[:, None, :]

		full[bad] = np.nan
		c[:, s:e] = np.where(bad[None, :], np.nan, c[:, s:e])
		Phi[:, :k, s:e] = np.transpose(full, (1, 2, 0))

	return c, Phi


def grad_h(h, pm, pn):
	"""Bathymetric slope magnitude on a curvilinear grid."""
	dhdxi  = np.gradient(h, axis=1)*pm
	dhdeta = np.gradient(h, axis=0)*pn
	return np.sqrt(dhdxi**2 + dhdeta**2)


def near_bottom_N2(N2, zw, thickness=NBOT_METRES):
	"""Mean N2 over the lowest `thickness` metres, skipping the bed cell."""
	zb = zw[0]
	w  = ((zw - zb) <= thickness) & np.isfinite(N2)
	w[0] = False
	if not w.any():
		w[1:4] = True
	num = np.nansum(np.where(w, N2, 0.), axis=0)
	den = np.nansum(w, axis=0)
	out = np.where(den > 0, num/np.maximum(den, 1), np.nan)
	thin = den == 0
	if thin.any():
		out[thin] = np.nanmean(N2[1:4, thin], axis=0)
	return out


def plume_thickness(N2, zw, zmax=120.):
	"""Depth of the maximum of N2 in the upper column, positive downward."""
	zmid = -0.5*np.abs(zw[0])
	top = zw > np.maximum(zmid, -zmax)
	N2top = np.where(top, N2, -np.inf)
	kmax = np.nanargmax(N2top, axis=0)
	return -np.take_along_axis(zw, kmax[None], axis=0)[0]


def criticality(gradh, N2bot, lat, omega=OMEGA_M2):
	"""
	alpha = |grad h| / s,  s = sqrt((omega^2 - f^2)/(N_b^2 - omega^2)).

	At these latitudes omega >> f, so alpha is set by the near bottom N2
	and the slope alone. That is the point of the diagnostic.
	"""
	f = 2*OMEGA_E*np.sin(np.deg2rad(lat))
	num = omega**2 - f**2
	den = N2bot - omega**2
	s = np.where((num > 0) & (den > 0), np.sqrt(np.abs(num/den)), np.nan)
	return gradh/s, s, f


def extract_isobath(lon, lat, h, level, min_pts=40):
	"""Longest contour of h at `level`, as lon, lat and along path km."""
	fig = plt.figure()
	cs = plt.contour(lon, lat, h, levels=[level])
	segs = [s for s in cs.allsegs[0] if len(s) >= min_pts]
	plt.close(fig)
	if not segs:
		raise RuntimeError(f'no {level} m contour found')
	seg = max(segs, key=len)
	blon, blat = seg[:, 0], seg[:, 1]
	if blat[0] > blat[-1]:
		blon, blat = blon[::-1], blat[::-1]
	dx = 111.2*np.cos(np.deg2rad(0.5*(blat[1:] + blat[:-1])))*np.diff(blon)
	dy = 111.2*np.diff(blat)
	dist = np.concatenate([[0.], np.cumsum(np.hypot(dx, dy))])
	return blon, blat, dist


def build_tree(lon, lat, mask=None):
	ok = np.isfinite(lon) & np.isfinite(lat)
	if mask is not None:
		ok &= mask > 0
	jj, ii = np.where(ok)
	tree = cKDTree(np.column_stack([lon[ok], lat[ok]]))
	return tree, jj, ii


def nearest_ji(tree, jj, ii, plon, plat):
	_, k = tree.query(np.column_stack([np.atleast_1d(plon),
									   np.atleast_1d(plat)]))
	return jj[k], ii[k]


def _hdims(da):
	eta = [d for d in da.dims if d.startswith('eta')]
	xi  = [d for d in da.dims if d.startswith('xi')]
	if not eta or not xi:
		raise KeyError(f'no eta/xi dims on {da.name}, found {da.dims}')
	return eta[0], xi[0]


# ---------------------------------------------------------------------
# streaming harmonic fit
# ---------------------------------------------------------------------
def harmonic_design(tsec, freqs):
	"""Design matrix, a mean plus a cosine and a sine per frequency."""
	cols = [np.ones_like(tsec)]
	for w in freqs:
		cols.append(np.cos(w*tsec))
		cols.append(np.sin(w*tsec))
	return np.column_stack(cols)


def stream_harmonic(da, tsec, freqs=BT_FREQS, tchunk=BT_TCHUNK):
	"""
	Least squares harmonic coefficients of a two dimensional field,
	accumulated one time chunk at a time so the record is never held in
	memory. Returns (K, ny, nx) with K = 1 + 2*len(freqs), the condition
	number of the normal equations, and the number of records used.

	Ordering is [mean, a1, b1, a2, b2, ...] for x = a cos(wt) + b sin(wt).
	"""
	K = 1 + 2*len(freqs)
	nt = da.shape[0]
	shp = da.shape[1:]
	XtX = np.zeros((K, K))
	Xty = np.zeros((K,) + shp)

	for s in range(0, nt, tchunk):
		e = min(s + tchunk, nt)
		X = harmonic_design(tsec[s:e], freqs)
		Y = np.nan_to_num(np.asarray(da[s:e].values, dtype=float))
		XtX += X.T @ X
		Xty += np.tensordot(X.T, Y, axes=([1], [0]))

	cond = float(np.linalg.cond(XtX))
	coef = np.linalg.solve(XtX, Xty.reshape(K, -1)).reshape((K,) + shp)
	return coef, cond, nt


def amp_pha(a, b):
	"""Amplitude and phase, in degrees, of x = a cos(wt) + b sin(wt)."""
	return np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360.


def wrap180(x):
	return (x + 180.) % 360. - 180.


def ellipse_axes(au, bu, av, bv):
	"""
	Semi major and semi minor axes of the tidal ellipse traced by
	u = au cos + bu sin, v = av cos + bv sin. Rotary decomposition, so
	the result does not depend on the orientation of the grid.
	"""
	Wp = 0.5*((au + bv) + 1j*(av - bu))
	Wm = 0.5*((au - bv) + 1j*(av + bu))
	rp, rm = np.abs(Wp), np.abs(Wm)
	return rp + rm, np.abs(rp - rm)


def bar_to_rho(v, ny, nx):
	"""Average a C grid depth averaged field onto the rho points."""
	v = np.asarray(v, dtype=float)
	out = np.full((ny, nx), np.nan)
	if v.shape[-1] == nx - 1:                    # u type, xi faces
		out[:, 1:-1] = 0.5*(v[:, :-1] + v[:, 1:])
		out[:, 0] = v[:, 0]; out[:, -1] = v[:, -1]
	elif v.shape[-2] == ny - 1:                  # v type, eta faces
		out[1:-1, :] = 0.5*(v[:-1, :] + v[1:, :])
		out[0, :] = v[0, :]; out[-1, :] = v[-1, :]
	else:
		out[:v.shape[0], :v.shape[1]] = v
	return out


def spring_neap_envelope(t_s, zeta, dt_s):
	"""
	Fortnightly envelope of the semidiurnal sea surface height, from the
	Hilbert transform of the 11 to 13 h bandpassed signal where the output
	interval resolves the band, and from the astronomical M2 S2 beat
	otherwise. Stored for fig03, no panel of this figure draws it.
	"""
	if dt_s <= 3*3600. + 1:
		fs = 1.0/dt_s
		lo, hi = 1.0/(13*3600.), 1.0/(11*3600.)
		b, a = butter(3, [lo/(fs/2), hi/(fs/2)], btype='band')
		band = filtfilt(b, a, zeta - np.nanmean(zeta))
		return band, np.abs(hilbert(band)), 'hilbert'
	warnings.warn('output interval does not resolve the semidiurnal band, '
				  'spring neap marked from the astronomical M2 S2 beat')
	beat = np.abs(1.0 + 0.35*np.exp(1j*(OMEGA_S2 - OMEGA_M2)*t_s))
	return np.full_like(beat, np.nan), beat, 'astronomical'


# =====================================================================
# MAIN
# =====================================================================
def main():
	print('opening runs')
	dsr = open_run(REALISTIC)
	dsc = open_run(CONTROL)
	grd = xr.open_dataset(GRIDFILE)

	lon = np.asarray(dsr.lon_rho); lat = np.asarray(dsr.lat_rho)
	h   = np.asarray(dsr.h)
	mask = np.asarray(dsr.mask_rho) if 'mask_rho' in dsr else np.ones_like(h)
	pm  = np.asarray(dsr.pm); pn = np.asarray(dsr.pn)
	ang = np.asarray(dsr.angle) if 'angle' in dsr else np.zeros_like(h)
	dx  = 1.0/pm
	ny, nx = h.shape
	wet = mask > 0
	print(f'  grid {ny} x {nx}, {dsr.sizes.get("s_rho")} levels')
	print(f'  horizontal spacing {np.nanmin(dx[wet])/1e3:.2f} to '
		  f'{np.nanmax(dx[wet])/1e3:.2f} km')

	t = dsr.ocean_time.values
	dt_s = float(np.median(np.diff(t))/np.timedelta64(1, 's'))
	tsec = (t - t[0])/np.timedelta64(1, 's')
	print(f'  record {str(t[0])[:10]} to {str(t[-1])[:10]}, '
		  f'{t.size} records at {dt_s/3600.:.2f} h')

	slope = grad_h(h, pm, pn)
	flat = wet.ravel()

	wins = list(WINDOWS.keys())
	nwin = len(wins)
	keys = ['c1_ref', 'c1_ctrl', 'ppw_ref', 'ppw_ctrl',
			'alpha_ref', 'alpha_ctrl', 'n2bot_ref', 'n2bot_ctrl',
			'hp_ref', 'hp_ctrl', 'hratio_ref', 'hratio_ctrl']
	out = {k: np.full((nwin, ny, nx), np.nan) for k in keys}
	flg = {k: np.zeros((nwin, ny, nx), dtype='int8') for k in
		   ['trap_ok_ref', 'trap_ok_ctrl', 'res_ref', 'res_ctrl', 'res_both']}

	# -----------------------------------------------------------------
	# window loop: modes, criticality, plume thickness, resolution
	# -----------------------------------------------------------------
	for iw, wname in enumerate(wins):
		t0, t1 = WINDOWS[wname]
		for tag, ds in (('ref', dsr), ('ctrl', dsc)):
			sub = ds.sel(ocean_time=slice(t0, t1))
			if sub.sizes['ocean_time'] == 0:
				print(f'  {wname}/{tag}: no times in window, skipped')
				continue
			print(f'  {wname}/{tag}: {sub.sizes["ocean_time"]} records')

			temp = sub.temp.mean('ocean_time').values
			salt = sub.salt.mean('ocean_time').values
			zr   = sub.z_rho.mean('ocean_time').values
			zw   = sub.z_w.mean('ocean_time').values

			N2 = compute_N2(temp, salt, zr, zw, lon, lat)
			Nw = zw.shape[0]
			N2f = N2.reshape(Nw, -1)[:, flat]
			zwf = zw.reshape(Nw, -1)[:, flat]

			c, _ = vertical_modes(N2f, zwf, nmodes=NMODES, free_surface=True)
			c1 = np.full(ny*nx, np.nan); c1[flat] = c[1]
			c1 = c1.reshape(ny, nx)

			n2b = np.full(ny*nx, np.nan)
			n2b[flat] = near_bottom_N2(N2f, zwf)
			n2b = n2b.reshape(ny, nx)

			hp = plume_thickness(N2, zw)
			hp = np.where(wet, hp, np.nan)

			ppw = np.where(c1 >= C1_MIN, c1*T_M2/dx, np.nan)
			alpha, _, _ = criticality(slope, n2b, lat)

			with np.errstate(invalid='ignore', divide='ignore'):
				hrat = np.where(hp > 0, h/hp, np.nan)

			out[f'c1_{tag}'][iw]     = c1
			out[f'ppw_{tag}'][iw]    = ppw
			out[f'alpha_{tag}'][iw]  = alpha
			out[f'n2bot_{tag}'][iw]  = n2b
			out[f'hp_{tag}'][iw]     = hp
			out[f'hratio_{tag}'][iw] = hrat

			flg[f'trap_ok_{tag}'][iw] = np.where(
				wet & np.isfinite(hrat) & (hrat > TRAP_RATIO), 1, 0)
			flg[f'res_{tag}'][iw] = np.where(
				wet & np.isfinite(ppw) & (ppw >= PPW_MIN), 1, 0)

			nmask = int(np.sum((c1 < C1_MIN) & wet))
			print(f'    {nmask} wet cells with c1 < {C1_MIN} m/s, '
				  f'points per wavelength undefined there')

		flg['res_both'][iw] = (flg['res_ref'][iw]*flg['res_ctrl'][iw])

	# -----------------------------------------------------------------
	# barotropic control, the M2 tide the two runs share
	# -----------------------------------------------------------------
	print('barotropic M2 control')
	bt = {}
	for tag, ds in (('ref', dsr), ('ctrl', dsc)):
		coef, cond, nt = stream_harmonic(ds.zeta, tsec)
		A, P = amp_pha(coef[1], coef[2])
		bt[f'zeta_amp_{tag}'] = np.where(wet, A, np.nan)
		bt[f'zeta_pha_{tag}'] = np.where(wet, P, np.nan)
		print(f'  {tag}: zeta fit on {nt} records, '
			  f'normal equation condition number {cond:.1f}')

		if DO_UBT and 'ubar' in ds and 'vbar' in ds:
			cu, _, _ = stream_harmonic(ds.ubar, tsec)
			cv, _, _ = stream_harmonic(ds.vbar, tsec)
			au = bar_to_rho(cu[1], ny, nx); bu = bar_to_rho(cu[2], ny, nx)
			av = bar_to_rho(cv[1], ny, nx); bv = bar_to_rho(cv[2], ny, nx)
			maj, mino = ellipse_axes(au, bu, av, bv)
			bt[f'ubt_maj_{tag}'] = np.where(wet, maj, np.nan)
			bt[f'ubt_min_{tag}'] = np.where(wet, mino, np.nan)
			print(f'  {tag}: depth averaged M2 semi major axis, median '
				  f'{np.nanmedian(maj[wet]):.3f} m s-1')
		else:
			if DO_UBT:
				warnings.warn('ubar / vbar absent, the depth averaged M2 '
							  'current control is left empty. Add them to '
							  'the output or set DO_UBT = False.')
			bt[f'ubt_maj_{tag}'] = np.full((ny, nx), np.nan)
			bt[f'ubt_min_{tag}'] = np.full((ny, nx), np.nan)

	with np.errstate(invalid='ignore', divide='ignore'):
		bt['dzeta_rel'] = 100.*(bt['zeta_amp_ref'] - bt['zeta_amp_ctrl']) \
						  / np.where(bt['zeta_amp_ref'] > 1e-3,
									 bt['zeta_amp_ref'], np.nan)
		bt['dzeta_pha'] = wrap180(bt['zeta_pha_ref'] - bt['zeta_pha_ctrl'])
		bt['dubt_rel'] = 100.*(bt['ubt_maj_ref'] - bt['ubt_maj_ctrl']) \
						 / np.where(bt['ubt_maj_ref'] > 1e-3,
									bt['ubt_maj_ref'], np.nan)

	# -----------------------------------------------------------------
	# shelf break path and along break profiles
	# -----------------------------------------------------------------
	print('extracting the shelf break')
	blon, blat, bdist = extract_isobath(lon, lat, np.where(wet, h, np.nan),
										BREAK_ISOBATH)
	tree, jj, ii = build_tree(lon, lat, mask)
	bj, bi = nearest_ji(tree, jj, ii, blon, blat)
	nb = blon.size

	brk = {}
	for v in ['alpha_ref', 'alpha_ctrl', 'c1_ref', 'c1_ctrl',
			  'n2bot_ref', 'n2bot_ctrl', 'hratio_ref', 'hratio_ctrl']:
		brk[v + '_break'] = out[v][:, bj, bi]
	brk['dalpha_break'] = brk['alpha_ref_break'] - brk['alpha_ctrl_break']
	with np.errstate(invalid='ignore', divide='ignore'):
		brk['dn2bot_rel_break'] = 100.*(brk['n2bot_ref_break']
										- brk['n2bot_ctrl_break']) \
								  / brk['n2bot_ref_break']
	brk['res_both_break'] = flg['res_both'][:, bj, bi].astype(float)
	slope_break = slope[bj, bi]
	h_break = h[bj, bi]
	dzeta_rel_break = bt['dzeta_rel'][bj, bi]
	dubt_rel_break = bt['dubt_rel'][bj, bi]

	# -----------------------------------------------------------------
	# depth binned resolution statistics for panel (d)
	# -----------------------------------------------------------------
	print('depth binned resolution')
	ctr = np.sqrt(DEPTH_BINS[1:]*DEPTH_BINS[:-1])
	nbin = ctr.size
	res = {k: np.full((nwin, nbin), np.nan) for k in
		   ['ppw_med_ref', 'ppw_q25_ref', 'ppw_q75_ref',
			'ppw_med_ctrl', 'ppw_q25_ctrl', 'ppw_q75_ctrl',
			'frac_res_both', 'frac_res_ref', 'frac_res_ctrl',
			'frac_trap_ref']}
	hv = h[wet]
	for iw in range(nwin):
		pr = out['ppw_ref'][iw][wet]
		pc = out['ppw_ctrl'][iw][wet]
		rb = flg['res_both'][iw][wet]
		rr = flg['res_ref'][iw][wet]
		rc = flg['res_ctrl'][iw][wet]
		tr = flg['trap_ok_ref'][iw][wet]
		for k in range(nbin):
			sel = (hv >= DEPTH_BINS[k]) & (hv < DEPTH_BINS[k + 1])
			if sel.sum() < 20:
				continue
			for tag, p in (('ref', pr), ('ctrl', pc)):
				q = p[sel]
				if np.isfinite(q).sum() > 10:
					res[f'ppw_med_{tag}'][iw, k] = np.nanmedian(q)
					res[f'ppw_q25_{tag}'][iw, k] = np.nanpercentile(q, 25)
					res[f'ppw_q75_{tag}'][iw, k] = np.nanpercentile(q, 75)
			res['frac_res_both'][iw, k] = rb[sel].mean()
			res['frac_res_ref'][iw, k]  = rr[sel].mean()
			res['frac_res_ctrl'][iw, k] = rc[sel].mean()
			res['frac_trap_ref'][iw, k] = tr[sel].mean()

	# -----------------------------------------------------------------
	# generation sites, snapped to the break
	# -----------------------------------------------------------------
	print('locating the generation sites')
	perim = np.vstack([
		np.column_stack([lon[0, :],     lat[0, :]]),
		np.column_stack([lon[:, -1],    lat[:, -1]]),
		np.column_stack([lon[-1, ::-1], lat[-1, ::-1]]),
		np.column_stack([lon[::-1, 0],  lat[::-1, 0]]),
	])
	domain = MplPath(perim)

	site_names = list(GEN_SITES.keys())
	ns_ = len(site_names)
	s_lon = np.full(ns_, np.nan); s_lat = np.full(ns_, np.nan)
	s_lon0 = np.array([GEN_SITES[k][0] for k in site_names])
	s_lat0 = np.array([GEN_SITES[k][1] for k in site_names])
	s_inside = np.zeros(ns_, dtype='int8')
	s_h = np.full(ns_, np.nan); s_snapkm = np.full(ns_, np.nan)
	s_bdist = np.full(ns_, np.nan)

	for n, k in enumerate(site_names):
		p0 = np.array([[s_lon0[n], s_lat0[n]]])
		s_inside[n] = int(domain.contains_points(p0)[0])
		if SNAP_SITES_TO_BREAK and s_inside[n]:
			dl = (blon - s_lon0[n])*111.2*np.cos(np.deg2rad(s_lat0[n]))
			db = (blat - s_lat0[n])*111.2
			m = int(np.argmin(np.hypot(dl, db)))
			s_lon[n], s_lat[n] = blon[m], blat[m]
			s_snapkm[n] = float(np.hypot(dl, db)[m])
		else:
			s_lon[n], s_lat[n] = s_lon0[n], s_lat0[n]
		if s_inside[n]:
			j, i = nearest_ji(tree, jj, ii, s_lon[n], s_lat[n])
			j, i = int(j[0]), int(i[0])
			s_h[n] = h[j, i]
			s_bdist[n] = min(j, ny - 1 - j, i, nx - 1 - i)*float(dx[j, i])/1e3

	for n, k in enumerate(site_names):
		print(f'  {k}  lon {s_lon[n]:7.2f}  lat {s_lat[n]:6.2f}  '
			  f'in domain {"yes" if s_inside[n] else "NO":>3}  '
			  f'h {s_h[n]:6.0f} m  snap {s_snapkm[n]:5.1f} km  '
			  f'to boundary {s_bdist[n]:5.0f} km')

	# -----------------------------------------------------------------
	# spring neap envelope, stored for fig03
	# -----------------------------------------------------------------
	print('spring neap envelope')
	jb, ib = nearest_ji(tree, jj, ii, -48.0, 2.0)
	zeta = np.asarray(dsr.zeta.isel(eta_rho=int(jb[0]),
									xi_rho=int(ib[0])).values).squeeze()
	band, env, env_kind = spring_neap_envelope(tsec, zeta, dt_s)

	# -----------------------------------------------------------------
	# assemble and write
	# -----------------------------------------------------------------
	print('writing', OUTFILE)
	coords = dict(
		window=('window', wins),
		eta_rho=('eta_rho', np.arange(ny)),
		xi_rho=('xi_rho', np.arange(nx)),
		break_pt=('break_pt', np.arange(nb)),
		site=('site', site_names),
		depth_bin=('depth_bin', ctr),
		time=('time', t),
	)
	dv = {}
	for k, v in out.items():
		dv[k] = (('window', 'eta_rho', 'xi_rho'), v)
	for k, v in flg.items():
		dv[k] = (('window', 'eta_rho', 'xi_rho'), v)
	for k, v in bt.items():
		dv[k] = (('eta_rho', 'xi_rho'), v)
	for k, v in brk.items():
		dv[k] = (('window', 'break_pt'), v)
	for k, v in res.items():
		dv[k] = (('window', 'depth_bin'), v)

	dv.update(dict(
		lon_rho=(('eta_rho', 'xi_rho'), lon),
		lat_rho=(('eta_rho', 'xi_rho'), lat),
		h=(('eta_rho', 'xi_rho'), h),
		mask_rho=(('eta_rho', 'xi_rho'), mask),
		angle=(('eta_rho', 'xi_rho'), ang),
		dx=(('eta_rho', 'xi_rho'), dx),
		slope=(('eta_rho', 'xi_rho'), slope),
		break_lon=('break_pt', blon),
		break_lat=('break_pt', blat),
		break_dist=('break_pt', bdist),
		break_slope=('break_pt', slope_break),
		break_h=('break_pt', h_break),
		dzeta_rel_break=('break_pt', dzeta_rel_break),
		dubt_rel_break=('break_pt', dubt_rel_break),
		site_lon=('site', s_lon),
		site_lat=('site', s_lat),
		site_inside=('site', s_inside),
		site_h=('site', s_h),
		site_snap_km=('site', s_snapkm),
		site_bdy_km=('site', s_bdist),
		zeta_ref=('time', zeta),
		zeta_band=('time', band),
		springneap_env=('time', env),
	))

	ds_out = xr.Dataset(dv, coords=coords)
	ds_out.attrs.update(dict(
		realistic=REALISTIC, control=CONTROL,
		dt_hours=dt_s/3600.,
		record_start=str(t[0])[:19], record_end=str(t[-1])[:19],
		break_isobath=BREAK_ISOBATH,
		shelf_max_depth=SHELF_MAX_DEPTH,
		ppw_lo=PPW_TARGET[0], ppw_hi=PPW_TARGET[1], ppw_min=PPW_MIN,
		c1_min=C1_MIN, trap_ratio=TRAP_RATIO,
		ctrl_discharge=CTRL_Q,
		omega_M2_cpd=OMEGA_M2*86400/(2*np.pi),
		springneap_kind=env_kind,
		windows=';'.join(f'{k}:{v[0]}..{v[1]}' for k, v in WINDOWS.items()),
		gen_sites_main=','.join(GEN_SITES_MAIN),
		sections=';'.join(f'{k}:{v[0]},{v[1]},{v[2]},{v[3]}'
						  for k, v in SECTIONS.items()),
		moorings=';'.join(f'{k}:{v[0]},{v[1]}' for k, v in MOORINGS.items()),
		note='mode 1 is index 1 of the free surface eigenproblem; alpha uses '
			 'near bottom N2 by construction; res_both is the subdomain in '
			 'which the two experiments are compared quantitatively; no '
			 'discharge series is stored; the spring neap envelope is kept '
			 'for fig03 and drawn by no panel of this figure',
	))

	enc = {v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars}
	ds_out.to_netcdf(OUTFILE, encoding=enc)
	summary(ds_out, wins, h, wet, bt, brk, flg, out)


def summary(ds, wins, h, wet, bt, brk, flg, out):
	"""The numbers the Results section quotes from this figure."""
	print('\n' + '=' * 72)
	print('PREMISE 1  the two runs share the barotropic tide')
	print('=' * 72)
	band = wet & (h > 20.) & (h < 1000.)
	for nm, v, u in (('M2 elevation amplitude', bt['dzeta_rel'], '%'),
					 ('M2 depth averaged current', bt['dubt_rel'], '%'),
					 ('M2 elevation phase', bt['dzeta_pha'], 'deg')):
		x = np.abs(v[band])
		if np.isfinite(x).any():
			print(f'  {nm:28s} median |REF-CTRL| {np.nanmedian(x):6.2f} {u}, '
				  f'95th percentile {np.nanpercentile(x, 95):6.2f} {u}')
	print('  A small number here is what licenses the attribution of the')
	print('  conversion anomaly to the baroclinic pressure at the bed alone.')

	print('\n' + '=' * 72)
	print('PREMISE 2  the lens is surface trapped where alpha is evaluated')
	print('=' * 72)
	for iw, w in enumerate(wins):
		hr = out['hratio_ref'][iw]
		for nm, lo, hi in (('mid shelf, 20 to 100 m', 20., 100.),
						   ('outer shelf, 100 to 250 m', 100., 250.),
						   ('upper slope, 250 to 1000 m', 250., 1000.)):
			m = wet & (h >= lo) & (h < hi)
			f = np.nanmean(flg['trap_ok_ref'][iw][m])
			print(f'  {w:>11}  {nm:28s} h > 3 hp over {100*f:5.1f} % of '
				  f'the area, median h/hp {np.nanmedian(hr[m]):5.1f}')
	print('  The criticality argument is applied only where this fraction is')
	print('  near one, which excludes the mid shelf by construction.')

	print('\n' + '=' * 72)
	print('PREMISE 3  criticality is common to the two runs at the break')
	print('=' * 72)
	for iw, w in enumerate(wins):
		ar = brk['alpha_ref_break'][iw]
		ac = brk['alpha_ctrl_break'][iw]
		da = brk['dalpha_break'][iw]
		spread = np.nanstd(ar)
		flip = np.nanmean(((ar > 1.) != (ac > 1.)).astype(float))
		dn = brk['dn2bot_rel_break'][iw]
		print(f'  {w:>11}  alpha REF {np.nanmedian(ar):5.2f}  '
			  f'CTRL {np.nanmedian(ac):5.2f}  |d alpha| median '
			  f'{np.nanmedian(np.abs(da)):6.3f}  along break sd {spread:5.2f} '
			  f'  ratio {np.nanmedian(np.abs(da))/spread:6.3f}')
		print(f'{"":15}near bottom N2 differs by {np.nanmedian(np.abs(dn)):5.1f} '
			  f'% at the break, and the two runs fall on opposite sides of '
			  f'alpha = 1 over {100*flip:4.1f} % of it')

	print('\n' + '=' * 72)
	print('PREMISE 4  the subdomain in which the runs are compared')
	print('=' * 72)
	for iw, w in enumerate(wins):
		for nm, lo, hi in (('mid shelf', 20., 100.),
						   ('outer shelf', 100., 250.),
						   ('slope and deep', 250., 4000.)):
			m = wet & (h >= lo) & (h < hi)
			fr = np.nanmean(flg['res_ref'][iw][m])
			fc = np.nanmean(flg['res_ctrl'][iw][m])
			fb = np.nanmean(flg['res_both'][iw][m])
			print(f'  {w:>11}  {nm:16s} resolved  REF {100*fr:5.1f} %  '
				  f'CTRL {100*fc:5.1f} %  BOTH {100*fb:5.1f} %')
	print('  CTRL is the binding constraint because the plume lengthens the')
	print('  mode. Flux and modal quantities are quoted on the BOTH column')
	print('  only, and reported as bounds elsewhere.')


if __name__ == '__main__':
	main()