#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig01_compute.py
=================================================================
Figure 1  -  Forcing and the equatorial internal tide regime
Amazon shelf internal tide manuscript

Computes and stores everything panels (a) to (e) need:

  (a) bathymetry, isobaths, moorings, sections, virtual stations
  (b) points per mode one wavelength, REF and CTRL
  (c) Amazon discharge, hydrological windows, spring neap envelope
  (d) criticality alpha along the shelf break, REF and CTRL
  (e) rotary spectra at stations from the inner shelf to the slope

Output:  fig01_data.nc   (single dataset, read by fig01_plot.py)

Physics notes
-------------
* Vertical modes solve the Sturm-Liouville problem with the FREE SURFACE
  condition Phi(0) = (c^2/g) dPhi/dz, retained rather than the rigid lid
  because the shelf is shallow and the surface term is not negligible
  there (Tchilibou et al. 2022).  The weak form puts the boundary term on
  the mass matrix, so the problem stays symmetric and mode 0 comes out as
  the barotropic mode with c0 ~ sqrt(gH).  Mode 1 is therefore index 1.
* Modes are computed on the WINDOW MEAN stratification, which is the
  standard background-state definition used for modal decompositions of
  the internal tide.  N2 from mean(T,S) is not mean(N2); the difference is
  small for a background state and it is the convention in the references.
* Criticality uses NEAR BOTTOM N2 by construction (Baines 1982), which is
  exactly why the plume is expected to leave it almost untouched.  Do not
  read a null result here as a failed diagnostic, it is the control that
  removes geometry from the explanation.
=================================================================
"""

import os
import time as _time
import warnings
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from scipy.signal import welch, butter, filtfilt, hilbert

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
import matplotlib.pyplot as plt   # used headless, only to extract contours
from matplotlib.path import Path as MplPath


# =====================================================================
# CONFIGURATION
# =====================================================================
REALISTIC = '/data1/amasseds_run/new_mean/roms_am2_new_mean.nc'
CONTROL   = '/data1/amasseds_run/const/roms_am2_redo.nc'
GRIDFILE  = '/home/rafaela/ROMS/pyroms_tools_arian/data/grid/pca_grd_river_open.nc'

# ROMS river forcing file. Set to None if you prefer to plot the discharge
# from your own GloFAS csv in the plotting script instead.
RIVER_FRC = None          # e.g. '/home/rafaela/ROMS/.../pca_river.nc'
AMAZON_RIVER_IDX = None   # e.g. [0, 1] for North and South Channel, None = sum all

OUTFILE = 'fig01_data.nc'

# hydrological windows, following Nascimento et al. (2026)
WINDOWS = {
	'rising':     ('1990-01-01', '1990-02-28'),
	'peak':       ('1990-03-01', '1990-05-31'),
	'decreasing': ('1990-06-01', '1990-07-31'),
}

MOORINGS = {
	'M1': (-50.3133, 3.0752),
	'M2': (-49.9372, 3.3853),
	'M3': (-49.6225, 4.0715),
}

# ---------------------------------------------------------------------
# Internal tide generation sites A to F, first named by Magalhaes et al.
# (2016) and reproduced in the model bathymetric slope map of Tchilibou et
# al. (2022) and Assene et al. (2024).
#
# Digitised from that slope map, so each position carries roughly +/- 0.3
# degrees of reading error. That error is removed below by snapping every
# site onto the model 200 m isobath, which is where the sites physically
# sit, so only the along break position has to be approximately right.
# Sites A and B are the two most intense (A* and B* in the source figure).
# ---------------------------------------------------------------------
GEN_SITES = {
	'A': (-46.60,  0.45),
	'B': (-43.80, -1.98),
	'C': (-45.20, -0.23),
	'D': (-47.60,  1.55),
	'E': (-50.50,  4.23),
	'F': (-51.90,  5.47),
}
GEN_SITES_MAIN = ('A', 'B')     # the two dominant sites, drawn emphasised

# Snap each site to the nearest point of the extracted BREAK_ISOBATH path.
# Set to False to keep the digitised coordinates exactly as written above.
SNAP_SITES_TO_BREAK = True

# cross shelf sections, (lon0, lat0, lon1, lat1), from coast to deep ocean
SECTIONS = {
	'S1': (-50.90,  2.40, -47.60,  4.60),   # southern, through M1-M3
	'S2': (-48.60,  0.60, -45.40,  2.90),   # central
	'S3': (-45.90, -0.90, -43.00,  1.10),   # northern
}

# virtual stations, placed along SECTION_FOR_STATIONS at these target depths
SECTION_FOR_STATIONS = 'S1'
STATION_DEPTHS = [20., 50., 100., 200., 1000., 2000.]

# shelf break definition
BREAK_ISOBATH = 200.        # m, path along which alpha is reported
SHELF_MAX_DEPTH = 250.      # m, shelf / deep partition

# constants
G      = 9.81
RHO0   = 1025.0
OMEGA_M2 = 2*np.pi/(12.4206012*3600.)     # rad s-1
OMEGA_S2 = 2*np.pi/(12.0*3600.)
OMEGA_M4 = 2*OMEGA_M2
T_M2     = 12.4206012*3600.               # s
OMEGA_E  = 7.2921e-5                      # earth rotation rate

NMODES     = 4          # modes 0..3, mode 0 is barotropic with free surface
N2_FLOOR   = 1e-9       # s-2, floor for statically unstable columns
NBOT_METRES = 50.       # m, thickness over which near bottom N2 is averaged
CHUNK_COLS = 4000       # columns per batched eigen solve

PPW_TARGET = (8., 12.)  # Hallberg (2013) points per wavelength criterion

# Rotary spectra are I/O bound: a point time series forces a full pass over
# the variable, so cost scales with the number of passes. ('ref',) halves it
# and is enough for Figure 1, which only shows the realistic run.
SPECTRA_RUNS = ('ref', 'ctrl')
# Time chunk used when reading velocity. Larger means fewer, bigger reads.
SPECTRA_TIME_CHUNK = 400

# Below this mode 1 speed the column carries no resolvable internal wave, so
# the points per wavelength metric is undefined rather than small.
MASK_UNSTRATIFIED = True
C1_MIN = 0.05           # m s-1


# =====================================================================
# HELPERS
# =====================================================================
def open_run(path, chunks={'ocean_time': 50}):
	"""Open a ROMS history/average file, with or without xroms."""
	if HAS_XROMS:
		try:
			out = xroms.open_netcdf(path, chunks=chunks)
		except TypeError:
			out = xroms.open_netcdf(path)
		# newer xroms returns (ds, xgrid)
		if isinstance(out, tuple):
			ds = out[0]
		else:
			ds = out
	else:
		ds = xr.open_dataset(path, chunks=chunks)
	return ds


def get_z(ds):
	"""Return z_rho, z_w as DataArrays, whatever the xroms version did."""
	if 'z_rho' in ds and 'z_w' in ds:
		return ds['z_rho'], ds['z_w']
	raise KeyError('z_rho / z_w not attached. Open the file through xroms, '
				   'or build them with xroms.roms_dataset(ds).')


def compute_N2(temp, salt, z_rho, z_w, lon, lat):
	"""
	Buoyancy frequency squared at w points from window mean T and S.

	Returns N2 with the same vertical size as z_w (bottom and surface
	values extrapolated from the adjacent interior points, which the free
	surface mode solve needs).
	"""
	if HAS_GSW:
		p  = gsw.p_from_z(z_rho, lat)
		SA = gsw.SA_from_SP(salt, p, lon, lat)
		CT = gsw.CT_from_pt(SA, temp)
		rho = gsw.rho(SA, CT, 0.0)          # potential density, ref 0 dbar
	else:
		warnings.warn('gsw not available, using a linear equation of state')
		rho = RHO0*(1 - 1.7e-4*(temp - 10.) + 7.6e-4*(salt - 35.))

	rho = np.asarray(rho)
	zr  = np.asarray(z_rho)
	zw  = np.asarray(z_w)

	# d(rho)/dz between rho levels -> interior w points
	drho = np.diff(rho, axis=0)
	dz   = np.diff(zr,  axis=0)
	N2_int = -(G/RHO0)*drho/dz                     # (Nz-1, ...) = interior w

	# pad to the full w grid
	shp = (zw.shape[0],) + N2_int.shape[1:]
	N2 = np.empty(shp, dtype=float)
	N2[1:-1] = N2_int
	N2[0]    = N2_int[0]
	N2[-1]   = N2_int[-1]
	return N2


def vertical_modes(N2, zw, nmodes=NMODES, free_surface=True,
				   n2_floor=N2_FLOOR, chunk=CHUNK_COLS):
	"""
	Solve  d2Phi/dz2 + (N2/c2) Phi = 0,  Phi(-H) = 0,
	with the free surface condition  Phi(0) = (c2/g) dPhi/dz|_0.

	Weak form:  int Phi' Psi' dz = lambda ( int N2 Phi Psi dz + g Phi(0)Psi(0) )
	so both matrices stay symmetric and the eigenproblem is standard after
	a diagonal similarity transform.

	Parameters
	----------
	N2 : (Nw, M) array at w points, bottom first
	zw : (Nw, M) array of w point depths, negative, increasing to 0
	free_surface : if True the surface node is an unknown and mode 0 is the
				   barotropic mode with c0 ~ sqrt(gH). If False the rigid
				   lid is used and mode 0 is the first baroclinic mode.

	Returns
	-------
	c   : (nmodes, M) modal phase speeds, m s-1
	Phi : (Nw, nmodes, M) modal structure at w points, normalised so that
		  max|Phi| = 1 and Phi > 0 in the upper half of the column
	"""
	Nw, M = N2.shape
	n = Nw - 1 if free_surface else Nw - 2

	c   = np.full((nmodes, M), np.nan)
	Phi = np.full((Nw, nmodes, M), np.nan)

	N2c = np.maximum(N2, n2_floor)

	for s in range(0, M, chunk):
		e   = min(s + chunk, M)
		m   = e - s
		n2  = N2c[:, s:e]
		z   = zw[:, s:e]

		h = np.diff(z, axis=0)                       # (Nw-1, m) cell thickness
		bad = (~np.isfinite(h)).any(axis=0) | (h <= 0).any(axis=0) \
			  | (~np.isfinite(n2)).any(axis=0)
		h = np.where(np.isfinite(h) & (h > 0), h, 1.0)

		# ---- stiffness K (symmetric tridiagonal) and mass M (diagonal)
		K = np.zeros((m, n, n))
		Md = np.zeros((m, n))

		# interior nodes j = 0..(Nw-3) correspond to w index k = j+1
		nint = Nw - 2
		for j in range(nint):
			hb = h[j]        # spacing below node k=j+1
			ha = h[j + 1]    # spacing above
			K[:, j, j] = 1.0/hb + 1.0/ha
			if j > 0:
				K[:, j, j - 1] = -1.0/hb
			if j < n - 1:
				K[:, j, j + 1] = -1.0/ha
			Md[:, j] = n2[j + 1]*0.5*(hb + ha)

		if free_surface:
			j = n - 1                    # surface node, w index Nw-1
			hb = h[-1]
			K[:, j, j]     = 1.0/hb
			K[:, j, j - 1] = -1.0/hb
			Md[:, j] = n2[-1]*0.5*hb + G      # the boundary term

		Md = np.maximum(Md, 1e-12)
		D  = 1.0/np.sqrt(Md)                          # (m, n)
		A  = K*D[:, :, None]*D[:, None, :]
		A  = 0.5*(A + np.swapaxes(A, 1, 2))           # enforce symmetry

		lam, V = np.linalg.eigh(A)                    # ascending
		lam = np.maximum(lam, 1e-20)
		cc  = 1.0/np.sqrt(lam)                        # (m, n)

		Vphys = V*D[:, :, None]                       # back to Phi

		k = min(nmodes, n)
		c[:k, s:e] = cc[:, :k].T

		full = np.zeros((m, Nw, k))
		if free_surface:
			full[:, 1:, :] = Vphys[:, :, :k]
		else:
			full[:, 1:-1, :] = Vphys[:, :, :k]

		# normalise: unit maximum, positive in the upper half
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
	"""Average N2 over the lowest `thickness` metres of the column."""
	Nw = N2.shape[0]
	zb = zw[0]                                    # bottom depth (negative)
	w  = ((zw - zb) <= thickness) & np.isfinite(N2)
	w[0] = False                                  # skip the bed cell itself
	if not w.any():
		w[1:4] = True
	num = np.nansum(np.where(w, N2, 0.), axis=0)
	den = np.nansum(w, axis=0)
	out = np.where(den > 0, num/np.maximum(den, 1), np.nan)
	# fallback for very thin columns
	thin = den == 0
	if thin.any():
		out[thin] = np.nanmean(N2[1:4, thin], axis=0)
	return out


def criticality(gradh, N2bot, lat, omega=OMEGA_M2):
	"""
	alpha = |grad h| / s,  s = sqrt((omega^2 - f^2)/(N^2 - omega^2))

	At these latitudes omega >> f, so the numerator is essentially omega
	and alpha is controlled by near bottom N2 and the slope alone.
	"""
	f  = 2*OMEGA_E*np.sin(np.deg2rad(lat))
	num = omega**2 - f**2
	den = N2bot - omega**2
	s = np.where((num > 0) & (den > 0), np.sqrt(np.abs(num/den)), np.nan)
	return gradh/s, s, f


def extract_isobath(lon, lat, h, level, min_pts=40):
	"""Longest contour of h at `level`, returned as lon, lat and along path km."""
	fig = plt.figure()
	cs = plt.contour(lon, lat, h, levels=[level])
	segs = [s for s in cs.allsegs[0] if len(s) >= min_pts]
	plt.close(fig)
	if not segs:
		raise RuntimeError(f'no {level} m contour found')
	seg = max(segs, key=len)
	blon, blat = seg[:, 0], seg[:, 1]
	# order south to north for a monotonic axis
	if blat[0] > blat[-1]:
		blon, blat = blon[::-1], blat[::-1]
	dx = 111.2*np.cos(np.deg2rad(0.5*(blat[1:] + blat[:-1])))*np.diff(blon)
	dy = 111.2*np.diff(blat)
	dist = np.concatenate([[0.], np.cumsum(np.hypot(dx, dy))])
	return blon, blat, dist


def build_tree(lon, lat, mask=None):
	"""KD tree on the curvilinear grid for nearest neighbour extraction."""
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
	"""Return (eta_dim, xi_dim) actually present on a ROMS DataArray."""
	eta = [d for d in da.dims if d.startswith('eta')]
	xi  = [d for d in da.dims if d.startswith('xi')]
	if not eta or not xi:
		raise KeyError(f'no eta/xi dims on {da.name}, found {da.dims}')
	return eta[0], xi[0]


def uv_at_rho(ds, j, i, levels=(-1, 0)):
	"""
	Velocity time series at the rho point (j, i), averaged from the C grid
	faces rather than taken from a neighbouring face.

	Works whatever naming the file uses. xroms puts u on (eta_rho, xi_u) and
	v on (eta_v, xi_rho); a raw ROMS history file uses (eta_u, xi_u) and
	(eta_v, xi_v). Both are handled by inspecting the dims.

	Returns dict {level: (u_grid, v_grid)} in GRID coordinates, still to be
	rotated to east / north by the local angle.
	"""
	ue, xe = _hdims(ds.u)
	ve, xv = _hdims(ds.v)

	nxu = ds.sizes[xe]
	nev = ds.sizes[ve]

	# u sits on xi faces: the rho point i lies between xi_u i-1 and i
	iu0, iu1 = max(i - 1, 0), min(i, nxu - 1)
	# v sits on eta faces: the rho point j lies between eta_v j-1 and j
	jv0, jv1 = max(j - 1, 0), min(j, nev - 1)

	ju = min(j, ds.sizes[ue] - 1)      # u's eta index (eta_rho or eta_u)
	iv = min(i, ds.sizes[xv] - 1)      # v's xi index  (xi_rho or xi_v)

	ua = ds.u.isel({ue: ju, xe: iu0})
	ub = ds.u.isel({ue: ju, xe: iu1})
	va = ds.v.isel({ve: jv0, xv: iv})
	vb = ds.v.isel({ve: jv1, xv: iv})

	out = {}
	for lev in levels:
		uu = 0.5*(ua.isel(s_rho=lev).values + ub.isel(s_rho=lev).values)
		vv = 0.5*(va.isel(s_rho=lev).values + vb.isel(s_rho=lev).values)
		out[lev] = (np.asarray(uu).squeeze(), np.asarray(vv).squeeze())
	return out


def extract_uv_stations(ds, st_j, st_i, levels=(0, -1)):
	"""
	Pull u and v at every station in ONE pass over each variable.

	A point time series still forces the reader to touch every time record,
	so the cost is set by the number of passes over the file, not by the
	number of points wanted. Vectorised indexing puts all stations, both C
	grid faces and both levels into a single lazy selection, which is why
	this is roughly 36 times cheaper than looping station by station.

	Returns u, v with dims (ocean_time, s_rho, station), already averaged
	from the faces onto the rho point, still in grid coordinates.

	The s_rho axis of the RESULT follows the order of `levels`, not the model
	ordering, so levels=(0, -1) gives index 0 = bottom and index 1 = surface.
	"""
	ue, xe = _hdims(ds.u)
	ve, xv = _hdims(ds.v)

	j = np.asarray(st_j, int)
	i = np.asarray(st_i, int)
	nxu, nev = ds.sizes[xe], ds.sizes[ve]

	iu = np.stack([np.clip(i - 1, 0, nxu - 1), np.clip(i, 0, nxu - 1)], axis=1)
	jv = np.stack([np.clip(j - 1, 0, nev - 1), np.clip(j, 0, nev - 1)], axis=1)
	ju = np.clip(j, 0, ds.sizes[ue] - 1)
	iv = np.clip(i, 0, ds.sizes[xv] - 1)

	IU = xr.DataArray(iu, dims=('station', 'face'))
	JU = xr.DataArray(np.repeat(ju[:, None], 2, axis=1), dims=('station', 'face'))
	JV = xr.DataArray(jv, dims=('station', 'face'))
	IV = xr.DataArray(np.repeat(iv[:, None], 2, axis=1), dims=('station', 'face'))

	lev = list(levels)
	u = ds.u.isel({ue: JU, xe: IU}).isel(s_rho=lev).mean('face')
	v = ds.v.isel({ve: JV, xv: IV}).isel(s_rho=lev).mean('face')

	t0 = _time.time()
	# one graph, one pass per variable, works with or without dask
	both = xr.Dataset({'u': u, 'v': v}).compute()
	print(f'    read in {_time.time() - t0:.0f} s')
	return both['u'], both['v']


def compute_spectra(dsr, dsc, st_names, st_j, st_i, ang, dt_s, strict=True,
					runs=SPECTRA_RUNS):
	"""
	Rotary spectra at every station, surface and bottom.

	strict=True raises on the first extraction failure instead of quietly
	returning an empty array, which is how the C grid indexing bug in the
	first version of this script went unnoticed.
	Set runs=('ref',) to halve the cost if only the realistic run is shown.
	"""
	nst = len(st_names)
	freq_ref = None
	spec = {}

	for tag, ds in (('ref', dsr), ('ctrl', dsc)):
		if tag not in runs:
			continue
		print(f'  {tag}: extracting {nst} stations in one pass')
		try:
			# index 0 = bottom, index 1 = surface, set by the levels order
			U, V = extract_uv_stations(ds, st_j, st_i, levels=(0, -1))
		except Exception as err:
			msg = f'velocity extraction failed for {tag}: {err}'
			if strict:
				raise RuntimeError(msg) from err
			warnings.warn(msg)
			continue

		Uv = np.asarray(U.values)      # (nt, 2, nst), s_rho order = levels
		Vv = np.asarray(V.values)

		for lev_ix, sfx in ((0, 'bot'), (1, 'surf')):
			cwa  = np.full((nst, 0), np.nan)
			rows_cw, rows_ccw = [None]*nst, [None]*nst
			for n in range(nst):
				a = float(ang[int(st_j[n]), int(st_i[n])])
				uu = Uv[:, lev_ix, n]
				vv = Vv[:, lev_ix, n]
				ue_ = uu*np.cos(a) - vv*np.sin(a)      # grid -> east
				vn_ = uu*np.sin(a) + vv*np.cos(a)      # grid -> north
				fr, cw, ccw = rotary_spectra(ue_, vn_, dt_s)
				if fr is None:
					warnings.warn(f'no spectrum at {st_names[n]} {sfx} ({tag})')
					continue
				if freq_ref is None:
					freq_ref = fr
				rows_cw[n]  = np.interp(freq_ref, fr, cw)
				rows_ccw[n] = np.interp(freq_ref, fr, ccw)

			for key, rows in ((f'psd_cw_{sfx}_{tag}', rows_cw),
							  (f'psd_ccw_{sfx}_{tag}', rows_ccw)):
				arr = np.full((nst, freq_ref.size), np.nan)
				for n, x in enumerate(rows):
					if x is not None:
						arr[n] = x
				spec[key] = arr
				print(f'    {key}: {np.isfinite(arr).all(axis=1).sum()}/{nst}')

	if freq_ref is None:
		raise RuntimeError('no station produced a spectrum, nothing to save')

	# fill any run that was skipped so the file keeps a consistent shape
	for tag in ('ref', 'ctrl'):
		for sfx in ('surf', 'bot'):
			for pre in ('psd_cw', 'psd_ccw'):
				spec.setdefault(f'{pre}_{sfx}_{tag}',
								np.full((nst, freq_ref.size), np.nan))
	return freq_ref, spec


def rotary_spectra(u, v, dt_s, nperseg=None):
	"""
	Rotary spectra of the complex velocity w = u + iv.

	Positive frequency is counterclockwise, negative is clockwise, which in
	the Northern Hemisphere puts wind driven inertial motion in the CW band.
	Returns frequency in cycles per day, always positive, with the CW and
	CCW spectra separated.
	"""
	u = np.asarray(u, float); v = np.asarray(v, float)
	good = np.isfinite(u) & np.isfinite(v)
	if good.sum() < 64:
		return None, None, None
	u = np.interp(np.arange(u.size), np.where(good)[0], u[good])
	v = np.interp(np.arange(v.size), np.where(good)[0], v[good])
	w = (u - u.mean()) + 1j*(v - v.mean())

	fs = 1.0/dt_s
	if nperseg is None:
		nperseg = int(min(w.size, max(256, w.size//4)))
	f, P = welch(w, fs=fs, window='hann', nperseg=nperseg,
				 noverlap=nperseg//2, return_onesided=False,
				 detrend=False, scaling='density')
	f = np.fft.fftshift(f); P = np.fft.fftshift(P)
	pos = f > 0
	neg = f < 0
	fpos = f[pos]
	ccw = P[pos]
	cw  = np.interp(fpos, -f[neg][::-1], P[neg][::-1])
	# welch returns a density per Hz because fs was given in Hz. Variance is
	# conserved under the change of variable f' = 86400 f, so the density in
	# cycles per day is the density per Hz divided by 86400. Without this the
	# spectra are correct in shape but overstated by that factor, and the axis
	# label is wrong.
	return fpos*86400., cw/86400., ccw/86400.       # cpd, m2 s-2 cpd-1


def spring_neap_envelope(t_s, zeta, dt_s):
	"""
	Fortnightly envelope of the semidiurnal sea surface height.

	If the output interval resolves the semidiurnal band, the envelope comes
	from the Hilbert transform of the 11 to 13 h bandpassed signal. If it does
	not, the astronomical M2 and S2 beat is returned instead, which still marks
	spring and neap dates correctly but carries no model amplitude.
	"""
	if dt_s <= 3*3600. + 1:
		fs = 1.0/dt_s
		lo, hi = 1.0/(13*3600.), 1.0/(11*3600.)
		b, a = butter(3, [lo/(fs/2), hi/(fs/2)], btype='band')
		band = filtfilt(b, a, zeta - np.nanmean(zeta))
		env = np.abs(hilbert(band))
		return band, env, 'hilbert'
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
	dx  = 1.0/pm
	ny, nx = h.shape
	print(f'  grid {ny} x {nx}, {dsr.sizes.get("s_rho")} levels')

	t = dsr.ocean_time.values
	dt_s = float(np.median(np.diff(t))/np.timedelta64(1, 's'))
	print(f'  output interval {dt_s/3600.:.2f} h')
	if dt_s > 3*3600.:
		warnings.warn('the semidiurnal band is not resolved by this output '
					  'interval, panel (e) will be aliased. Use a history '
					  'file with hourly output for the rotary spectra.')

	slope = grad_h(h, pm, pn)
	flat  = mask.ravel() > 0

	wins = list(WINDOWS.keys())
	nwin = len(wins)
	out = {k: np.full((nwin, ny, nx), np.nan) for k in
		   ['c1_ref', 'c1_ctrl', 'ppw_ref', 'ppw_ctrl',
			'alpha_ref', 'alpha_ctrl', 'n2bot_ref', 'n2bot_ctrl',
			'hp_ref', 'hp_ctrl']}

	# -----------------------------------------------------------------
	# window loop: modes, criticality, plume thickness
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

			# flatten the horizontal for the batched eigen solve
			Nw = zw.shape[0]
			N2f = N2.reshape(Nw, -1)[:, flat]
			zwf = zw.reshape(Nw, -1)[:, flat]

			c, _ = vertical_modes(N2f, zwf, nmodes=NMODES, free_surface=True)
			c1 = np.full(ny*nx, np.nan); c1[flat] = c[1]      # mode 1
			c1 = c1.reshape(ny, nx)

			n2b = np.full(ny*nx, np.nan)
			n2b[flat] = near_bottom_N2(N2f, zwf)
			n2b = n2b.reshape(ny, nx)

			# plume thickness = depth of maximum N2 in the upper column
			zmid = -0.5*np.abs(zw[0])
			top = zw > np.maximum(zmid, -120.)
			N2top = np.where(top, N2, -np.inf)
			kmax = np.nanargmax(N2top, axis=0)
			hp = -np.take_along_axis(zw, kmax[None], axis=0)[0]

			lam1 = c1*T_M2
			ppw = lam1/dx
			# Where the column is well mixed there is no mode 1 to resolve,
			# so ppw is undefined rather than small. The tidally mixed inner
			# shelf falls here, especially in CTRL. Masking keeps panel (b)
			# an honest statement about resolution instead of a statement
			# about stratification.
			if MASK_UNSTRATIFIED:
				ppw = np.where(c1 >= C1_MIN, ppw, np.nan)
			nmask = int(np.sum((c1 < C1_MIN) & (mask > 0)))
			if nmask:
				print(f'    {nmask} wet cells with c1 < {C1_MIN} m/s '
					  f'(no resolvable mode 1), ppw masked')
			alpha, _, _ = criticality(slope, n2b, lat)

			out[f'c1_{tag}'][iw]    = c1
			out[f'ppw_{tag}'][iw]   = ppw
			out[f'alpha_{tag}'][iw] = alpha
			out[f'n2bot_{tag}'][iw] = n2b
			out[f'hp_{tag}'][iw]    = hp

	# -----------------------------------------------------------------
	# shelf break path and the along break profiles
	# -----------------------------------------------------------------
	print('extracting the shelf break')
	blon, blat, bdist = extract_isobath(lon, lat, np.where(mask > 0, h, np.nan),
										BREAK_ISOBATH)
	tree, jj, ii = build_tree(lon, lat, mask)
	bj, bi = nearest_ji(tree, jj, ii, blon, blat)
	nb = blon.size

	brk = {}
	for v in ['alpha_ref', 'alpha_ctrl', 'c1_ref', 'c1_ctrl',
			  'n2bot_ref', 'n2bot_ctrl']:
		brk[v + '_break'] = out[v][:, bj, bi]
	slope_break = slope[bj, bi]
	h_break = h[bj, bi]

	# -----------------------------------------------------------------
	# generation sites: snap to the break, test containment, check the
	# distance to the open boundaries
	# -----------------------------------------------------------------
	print('locating the generation sites')
	perim = np.vstack([
		np.column_stack([lon[0, :],   lat[0, :]]),
		np.column_stack([lon[:, -1],  lat[:, -1]]),
		np.column_stack([lon[-1, ::-1], lat[-1, ::-1]]),
		np.column_stack([lon[::-1, 0], lat[::-1, 0]]),
	])
	domain = MplPath(perim)

	site_names = list(GEN_SITES.keys())
	ns_ = len(site_names)
	s_lon = np.full(ns_, np.nan); s_lat = np.full(ns_, np.nan)
	s_lon0 = np.array([GEN_SITES[k][0] for k in site_names])
	s_lat0 = np.array([GEN_SITES[k][1] for k in site_names])
	s_inside = np.zeros(ns_, dtype='int8')
	s_bdist  = np.full(ns_, np.nan)      # km to the nearest open boundary
	s_h      = np.full(ns_, np.nan)
	s_snapkm = np.full(ns_, np.nan)

	for n, k in enumerate(site_names):
		p0 = np.array([[s_lon0[n], s_lat0[n]]])
		s_inside[n] = int(domain.contains_points(p0)[0])
		if SNAP_SITES_TO_BREAK and s_inside[n]:
			dlon = (blon - s_lon0[n])*111.2*np.cos(np.deg2rad(s_lat0[n]))
			dlat = (blat - s_lat0[n])*111.2
			m = int(np.argmin(np.hypot(dlon, dlat)))
			s_lon[n], s_lat[n] = blon[m], blat[m]
			s_snapkm[n] = float(np.hypot(dlon, dlat)[m])
		else:
			s_lon[n], s_lat[n] = s_lon0[n], s_lat0[n]

		if s_inside[n]:
			j, i = nearest_ji(tree, jj, ii, s_lon[n], s_lat[n])
			j, i = int(j[0]), int(i[0])
			s_h[n] = h[j, i]
			ncell = min(j, ny - 1 - j, i, nx - 1 - i)
			s_bdist[n] = ncell*float(dx[j, i])/1000.

	print(f'  {"site":>5}  {"lon":>8} {"lat":>7}  {"in dom":>6} '
		  f'{"h (m)":>7} {"snap km":>8} {"km to bdy":>10}')
	for n, k in enumerate(site_names):
		tag = 'yes' if s_inside[n] else 'NO'
		print(f'  {k:>5}  {s_lon[n]:8.2f} {s_lat[n]:7.2f}  {tag:>6} '
			  f'{s_h[n]:7.0f} {s_snapkm[n]:8.1f} {s_bdist[n]:10.0f}')
	if (s_inside == 0).any():
		miss = [k for n, k in enumerate(site_names) if not s_inside[n]]
		warnings.warn(f'generation sites outside the model domain: {miss}. '
					  'They are kept in the output with inside=0 so the plot '
					  'script can draw them as context, but no conversion '
					  'can be attributed to them in this configuration.')
	near = [k for n, k in enumerate(site_names)
			if s_inside[n] and np.isfinite(s_bdist[n]) and s_bdist[n] < 150.]
	if near:
		warnings.warn(f'generation sites within 150 km of an open boundary: '
					  f'{near}. Check the sponge and nudging width before '
					  'quoting their conversion, the baroclinic field there '
					  'is damped towards the boundary solution.')

	# -----------------------------------------------------------------
	# virtual stations
	# -----------------------------------------------------------------
	print('placing virtual stations')
	lo0, la0, lo1, la1 = SECTIONS[SECTION_FOR_STATIONS]
	npt = 400
	slon = np.linspace(lo0, lo1, npt); slat = np.linspace(la0, la1, npt)
	sj, si = nearest_ji(tree, jj, ii, slon, slat)
	sh = h[sj, si]

	st_names, st_lon, st_lat, st_h, st_j, st_i = [], [], [], [], [], []
	for name, (mlon, mlat) in MOORINGS.items():
		j, i = nearest_ji(tree, jj, ii, mlon, mlat)
		st_names.append(name); st_lon.append(lon[j, i][0]); st_lat.append(lat[j, i][0])
		st_h.append(h[j, i][0]); st_j.append(int(j[0])); st_i.append(int(i[0]))
	for d in STATION_DEPTHS:
		k = int(np.nanargmin(np.abs(sh - d)))
		if np.abs(sh[k] - d) > 0.5*d:
			continue
		st_names.append(f'V{int(d)}')
		st_lon.append(lon[sj[k], si[k]]); st_lat.append(lat[sj[k], si[k]])
		st_h.append(h[sj[k], si[k]]); st_j.append(int(sj[k])); st_i.append(int(si[k]))
	nst = len(st_names)
	print('  stations:', ', '.join(st_names))

	# -----------------------------------------------------------------
	# rotary spectra
	# -----------------------------------------------------------------
	print('rotary spectra')
	ang = np.asarray(dsr.angle) if 'angle' in dsr else np.zeros_like(h)
	freq_ref, spec = compute_spectra(dsr, dsc, st_names, st_j, st_i,
									 ang, dt_s, strict=True)

	# -----------------------------------------------------------------
	# discharge and the spring neap envelope
	# -----------------------------------------------------------------
	print('discharge and spring neap')
	q_time, q = None, None
	if RIVER_FRC and os.path.exists(RIVER_FRC):
		riv = xr.open_dataset(RIVER_FRC)
		tr = riv['river_transport']
		if AMAZON_RIVER_IDX is not None:
			tr = tr.isel(river=AMAZON_RIVER_IDX)
		q = np.abs(tr).sum('river').values
		tname = [d for d in tr.dims if 'time' in d][0]
		q_time = riv[tname].values
	else:
		warnings.warn('no river forcing file given, discharge left empty. '
					  'Set RIVER_FRC or plot your GloFAS series directly.')

	jb, ib = nearest_ji(tree, jj, ii, -48.0, 2.0)     # shelf break reference point
	zeta = np.asarray(dsr.zeta.isel(eta_rho=int(jb[0]), xi_rho=int(ib[0])).values).squeeze()
	tsec = (t - t[0])/np.timedelta64(1, 's')
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
		station=('station', st_names),
		site=('site', site_names),
		freq=('freq', freq_ref),
		time=('time', t),
	)
	dv = {}
	for k, v in out.items():
		dv[k] = (('window', 'eta_rho', 'xi_rho'), v)
	for k, v in brk.items():
		dv[k] = (('window', 'break_pt'), v)
	for k, v in spec.items():
		dv[k] = (('station', 'freq'), v)

	dv.update(dict(
		lon_rho=(('eta_rho', 'xi_rho'), lon),
		lat_rho=(('eta_rho', 'xi_rho'), lat),
		h=(('eta_rho', 'xi_rho'), h),
		mask_rho=(('eta_rho', 'xi_rho'), mask),
		dx=(('eta_rho', 'xi_rho'), dx),
		slope=(('eta_rho', 'xi_rho'), slope),
		break_lon=('break_pt', blon),
		break_lat=('break_pt', blat),
		break_dist=('break_pt', bdist),
		break_slope=('break_pt', slope_break),
		break_h=('break_pt', h_break),
		station_lon=('station', np.array(st_lon)),
		station_lat=('station', np.array(st_lat)),
		station_h=('station', np.array(st_h)),
		station_j=('station', np.array(st_j)),
		station_i=('station', np.array(st_i)),
		site_lon=('site', s_lon),
		site_lat=('site', s_lat),
		site_lon_raw=('site', s_lon0),
		site_lat_raw=('site', s_lat0),
		site_inside=('site', s_inside),
		site_h=('site', s_h),
		site_snap_km=('site', s_snapkm),
		site_bdy_km=('site', s_bdist),
		zeta_ref=('time', zeta),
		zeta_band=('time', band),
		springneap_env=('time', env),
	))

	ds_out = xr.Dataset(dv, coords=coords)

	if q is not None:
		ds_out = ds_out.assign_coords(qtime=('qtime', q_time))
		ds_out['discharge'] = ('qtime', q)

	# metadata the plot script reads
	ds_out.attrs.update(dict(
		realistic=REALISTIC, control=CONTROL,
		dt_hours=dt_s/3600.,
		break_isobath=BREAK_ISOBATH,
		shelf_max_depth=SHELF_MAX_DEPTH,
		ppw_lo=PPW_TARGET[0], ppw_hi=PPW_TARGET[1],
		omega_M2_cpd=OMEGA_M2*86400/(2*np.pi),
		omega_M4_cpd=OMEGA_M4*86400/(2*np.pi),
		springneap_kind=env_kind,
		windows=';'.join(f'{k}:{v[0]}..{v[1]}' for k, v in WINDOWS.items()),
		gen_sites_main=','.join(GEN_SITES_MAIN),
		gen_sites_snapped=str(bool(SNAP_SITES_TO_BREAK)),
		gen_sites_source='digitised from the model slope map of Tchilibou '
						 'et al. (2022) / Assene et al. (2024), sites first '
						 'named by Magalhaes et al. (2016)',
		sections=';'.join(f'{k}:{v[0]},{v[1]},{v[2]},{v[3]}'
						  for k, v in SECTIONS.items()),
		moorings=';'.join(f'{k}:{v[0]},{v[1]}' for k, v in MOORINGS.items()),
		note='mode 1 is index 1 of the free surface eigenproblem; '
			 'alpha uses near bottom N2 by construction',
	))

	ds_out['station_f'] = ('station',
						   2*OMEGA_E*np.sin(np.deg2rad(np.array(st_lat))))

	enc = {v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars}
	ds_out.to_netcdf(OUTFILE, encoding=enc)

	# -----------------------------------------------------------------
	# quick console summary, the numbers you will quote in the text
	# -----------------------------------------------------------------
	print('\n--- summary ---')
	shelf = (h > 50) & (h < SHELF_MAX_DEPTH) & (mask > 0)
	for iw, w in enumerate(wins):
		c1r = np.nanmean(out['c1_ref'][iw][shelf])
		c1c = np.nanmean(out['c1_ctrl'][iw][shelf])
		ar  = np.nanmedian(brk['alpha_ref_break'][iw])
		ac  = np.nanmedian(brk['alpha_ctrl_break'][iw])
		pr  = np.nanmedian(out['ppw_ref'][iw][shelf])
		pc  = np.nanmedian(out['ppw_ctrl'][iw][shelf])
		print(f'{w:>11}: shelf c1  REF {c1r:5.2f}  CTRL {c1c:5.2f} m/s  |  '
			  f'break alpha REF {ar:5.2f}  CTRL {ac:5.2f}  |  '
			  f'ppw REF {pr:5.1f}  CTRL {pc:5.1f}')
	print('\nIf the two alpha columns agree to within a few per cent, that is '
		  'the result panel (d) exists to show: criticality cannot explain a '
		  'plume driven difference, whatever its absolute value.')


def spectra_only(infile=OUTFILE):
	"""
	Recompute only the rotary spectra and patch them into an existing
	fig01_data.nc, reusing the station indices already stored there.

		python fig01_compute.py --spectra-only

	The vertical mode solves are the expensive part of this script, so this
	avoids repeating them when only panel (e) needs fixing.
	"""
	import shutil
	print('patching spectra into', infile)
	ds0 = xr.open_dataset(infile).load()
	ds0.close()

	dsr = open_run(REALISTIC, chunks={'ocean_time': SPECTRA_TIME_CHUNK})
	dsc = open_run(CONTROL,   chunks={'ocean_time': SPECTRA_TIME_CHUNK})

	t = dsr.ocean_time.values
	dt_s = float(np.median(np.diff(t))/np.timedelta64(1, 's'))
	print(f'  output interval {dt_s/3600.:.2f} h, '
		  f'Nyquist {12./(dt_s/3600.):.2f} cpd')

	# up front cost estimate, so a long run is expected rather than alarming
	try:
		nb = int(np.prod(dsr.u.shape))*dsr.u.dtype.itemsize
		npass = 2*len([r for r in SPECTRA_RUNS if r in ('ref', 'ctrl')])
		print(f'  velocity variable {nb/1e9:.1f} GB, {npass} passes, '
			  f'about {npass*nb/1e9:.0f} GB to read')
		print(f'  expect a few minutes on local disk, longer over a network '
			  f'mount. Set SPECTRA_RUNS = (\'ref\',) to halve it.')
	except Exception:
		pass

	h = np.asarray(dsr.h)
	ang = np.asarray(dsr.angle) if 'angle' in dsr else np.zeros_like(h)
	st_names = [str(s) for s in ds0.station.values]
	st_j = ds0.station_j.values
	st_i = ds0.station_i.values

	freq_ref, spec = compute_spectra(dsr, dsc, st_names, st_j, st_i,
									 ang, dt_s, strict=True)

	ds0 = ds0.drop_dims('freq', errors='ignore')
	ds0 = ds0.assign_coords(freq=('freq', freq_ref))
	for k, v in spec.items():
		ds0[k] = (('station', 'freq'), v)
	ds0.attrs['dt_hours'] = dt_s/3600.
	ds0.attrs['nyquist_cpd'] = 12./(dt_s/3600.)

	tmp = infile + '.tmp'
	enc = {v: {'zlib': True, 'complevel': 4} for v in ds0.data_vars}
	ds0.to_netcdf(tmp, encoding=enc)
	ds0.close()
	shutil.move(tmp, infile)
	print(f'  wrote {freq_ref.size} frequencies, '
		  f'{freq_ref.min():.3f} to {freq_ref.max():.3f} cpd')
	print(f'  M2 at {OMEGA_M2*86400/(2*np.pi):.4f} cpd, '
		  f'M4 at {OMEGA_M4*86400/(2*np.pi):.4f} cpd')


if __name__ == '__main__':
	import sys
	if '--spectra-only' in sys.argv:
		spectra_only()
	else:
		main()