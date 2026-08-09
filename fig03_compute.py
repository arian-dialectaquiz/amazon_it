###################
# -*- coding: utf-8 -*-
"""
fig03_compute.py
=================================================================
Figure 3  -  The M2 energy budget and its buoyancy control
Amazon shelf internal tide manuscript

Computes and stores everything panels (a) to (i) need:

  (a-c) barotropic to baroclinic conversion C for the three windows, REF
  (d-f) depth integrated baroclinic flux |F_bc| with propagation vectors
  (g)   time series of area averaged C over the shelf break, both runs,
		with the spring and neap phase and the fortnightly variability band
  (h,i) integrated budgets of C, div F_bc and D_bc for shelf and slope

Output:  fig03_data.nc   (read by fig03_plot.py)

Method
------
The whole budget is quadratic in the harmonic coefficients, so nothing
needs to be held as a time series. In ONE streaming pass per run we
accumulate the least squares normal equations for a harmonic fit at M2
and S2 at every point and level, for the baroclinic velocity, the
baroclinic pressure and the barotropic velocity. Then

	<x y> over a tidal cycle = 0.5 (a_x a_y + b_x b_y)

gives the conversion and the flux directly. Storing the semidiurnal time
series instead would need about 100 GB per run.

	C      = <p_bc(-H) w_bt(-H)>,   w_bt(-H) = -u_bt . grad H
	F_bc   = int <u_bc p_bc> dz
	D_bc   = C - div F_bc            (residual)

Sign convention: H is positive downward, the bottom is at z = -H, so the
kinematic bottom condition gives w = -u . grad H. Positive C is therefore
barotropic to baroclinic. The script checks the sign against the known
generation sites and warns if the slope integral comes out negative.

Equation of state: density comes from xroms, which implements ROMS' own
nonlinear equation of state, so the baroclinic pressure is consistent
with the pressure the model itself felt rather than an approximation to
it. The pressure is a linear functional of density, so p_bc is built from
the density ANOMALY relative to the window mean. That leaves the M2 part
untouched, since the mean is absorbed by the constant coefficient of the
fit, and it keeps p_bc near 1e3 Pa instead of 1e6, which is what makes
float32 accumulation safe. Fields are flattened to wet columns before the
arithmetic, since about a third of the grid is land.
=================================================================
"""

import os
import time as _time
import warnings
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from scipy.signal import butter, filtfilt, hilbert

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
	import xroms
	HAS_XROMS = True
except ImportError:
	HAS_XROMS = False

from fig01_compute import (
	REALISTIC, CONTROL, WINDOWS, MOORINGS, GEN_SITES, GEN_SITES_MAIN,
	BREAK_ISOBATH, SHELF_MAX_DEPTH, G, RHO0, OMEGA_M2, OMEGA_S2, T_M2,
	OMEGA_E, open_run, extract_isobath, build_tree, nearest_ji, _hdims,
	HAS_GSW,
)

if HAS_GSW:
	import gsw


# =====================================================================
# CONFIGURATION
# =====================================================================
OUTFILE = 'fig03_data.nc'

# 'xroms'      ROMS own nonlinear equation of state, the model's own rho,
#              so the baroclinic pressure is consistent with the pressure
#              the model actually felt. This is the default and the one to
#              use unless xroms is unavailable.
# 'model'      read rho straight from the history file if it carries one
# 'gsw'        TEOS-10, correct but slower and not the model's EOS
# 'linearized' expansion about the window mean, fastest fallback
EOS_MODE = 'xroms'
MEAN_STRIDE = 12             # stride used for the cheap window mean pass
BLOCK_HOURS = 50             # length of the blocks in the panel (g) series
TIME_CHUNK = 24              # reading chunk in records, caps the Y footprint

CONSTITUENTS = {'M2': OMEGA_M2, 'S2': OMEGA_S2}   # fitted per window
BLOCK_CONSTITUENTS = {'M2': OMEGA_M2}             # fitted per block

SHELF_MAX = SHELF_MAX_DEPTH  # m, shelf / slope partition
SLOPE_MAX = 3500.            # m, outer limit of the slope region
BREAK_BAND = (100., 1000.)   # m, the band used for the panel (g) average

RUNS = ('ref', 'ctrl')


# =====================================================================
# HELPERS
# =====================================================================
def uv_to_rho(u, v):
	"""
	C grid velocity to rho points, averaging the two adjacent faces.

	u : (..., ny, nx-1) on xi faces
	v : (..., ny-1, nx) on eta faces
	Returns u_rho, v_rho both (..., ny, nx).
	"""
	ny = v.shape[-2] + 1
	nx = u.shape[-1] + 1
	ur = np.empty(u.shape[:-2] + (ny, nx), dtype=u.dtype)
	vr = np.empty(v.shape[:-2] + (ny, nx), dtype=v.dtype)
	ur[..., 1:-1] = 0.5*(u[..., :-1] + u[..., 1:])
	ur[..., 0] = u[..., 0]
	ur[..., -1] = u[..., -1]
	vr[..., 1:-1, :] = 0.5*(v[..., :-1, :] + v[..., 1:, :])
	vr[..., 0, :] = v[..., 0, :]
	vr[..., -1, :] = v[..., -1, :]
	return ur, vr


def baroclinic_pressure(rho, zw):
	"""
	Baroclinic perturbation pressure on rho levels.

		P(z) = g * int_z^0 rho dz'
		p_bc = P - (1/H) int_{-H}^0 P dz

	The depth mean removal is what makes it baroclinic, and it also
	removes the free surface term, which is depth independent.

	rho : (nt, nz, M) density
	zw  : (nt, nz+1, M) w point depths
	Returns p_bc (nt, nz, M).
	"""
	dz = np.diff(zw, axis=1)                       # (nt, nz, M)
	# integral from each rho level up to the surface
	P = G*np.cumsum((rho*dz)[:, ::-1, :], axis=1)[:, ::-1, :]
	H = np.sum(dz, axis=1, keepdims=True)
	Pbar = np.sum(P*dz, axis=1, keepdims=True)/np.maximum(H, 1e-6)
	return P - Pbar


def design_matrix(t_s, consts):
	"""Least squares columns, a constant plus cos and sin per constituent."""
	cols = [np.ones_like(t_s)]
	for w in consts.values():
		cols.append(np.cos(w*t_s))
		cols.append(np.sin(w*t_s))
	return np.stack(cols, axis=1)          # (nt, ncoef)


def solve_normal(GtG, GtY, ridge=1e-10):
	"""
	Solve the accumulated normal equations for every point at once.

	GtG : (ncoef, ncoef)
	GtY : (ncoef, nfield, M)
	Returns coefficients (ncoef, nfield, M).
	"""
	A = GtG + ridge*np.trace(GtG)/GtG.shape[0]*np.eye(GtG.shape[0])
	shp = GtY.shape
	return np.linalg.solve(A, GtY.reshape(shp[0], -1)).reshape(shp)


def cycle_mean(a1, b1, a2, b2):
	"""<x y> over a tidal cycle from the cos and sin coefficients."""
	return 0.5*(a1*a2 + b1*b2)


def grad_h_components(h, pm, pn):
	"""dH/dx and dH/dy in grid directions, metres per metre."""
	dhdxi = np.gradient(h, axis=1)*pm
	dhdeta = np.gradient(h, axis=0)*pn
	return dhdxi, dhdeta


def divergence(Fxi, Feta, pm, pn):
	"""
	Curvilinear divergence of a vector given in GRID components.

		div F = pm pn [ d(F_xi / pn)/dxi + d(F_eta / pm)/deta ]
	"""
	a = np.gradient(Fxi/pn, axis=1)
	b = np.gradient(Feta/pm, axis=0)
	return pm*pn*(a + b)


def area_weights(pm, pn, mask):
	"""Cell area in m2, zero on land."""
	return np.where(mask > 0, 1.0/(pm*pn), 0.0)


def window_index(times, windows):
	"""Map every time record to a window index, or -1 if outside all."""
	idx = np.full(times.size, -1, dtype=int)
	for k, (name, (t0, t1)) in enumerate(windows.items()):
		sel = (times >= np.datetime64(t0)) & (times <= np.datetime64(t1))
		idx[sel] = k
	return idx


# =====================================================================
# EQUATION OF STATE
# =====================================================================
_DENSITY_FN = None


def _resolve_density_fn():
	"""
	Find xroms' implementation of the ROMS nonlinear equation of state.

	xroms exposes it as xroms.density in recent versions and as
	xroms.roms_seawater.density in older ones, so both are tried. The
	result is sanity checked against a known seawater value, and if the
	function returns sigma rather than density the offset is added.
	"""
	global _DENSITY_FN
	if _DENSITY_FN is not None:
		return _DENSITY_FN
	if not HAS_XROMS:
		raise RuntimeError('xroms not available, set EOS_MODE to gsw, '
						   'model or linearized')
	mods = [xroms, getattr(xroms, 'roms_seawater', None)]
	for mod in mods:
		if mod is None:
			continue
		f = getattr(mod, 'density', None)
		if f is None:
			continue
		try:
			probe = np.asarray(f(np.array([10.0]), np.array([35.0]),
								 np.array([0.0])))
		except Exception:
			continue
		if not np.isfinite(probe).all():
			continue
		off = 1000.0 if float(probe[0]) < 100.0 else 0.0
		if not (1000.0 < float(probe[0]) + off < 1100.0):
			continue

		def wrapped(T, S, Z, _f=f, _off=off):
			return np.asarray(_f(T, S, Z)) + _off

		print(f'  equation of state: xroms density, probe rho(10, 35, 0) '
			  f'= {float(probe[0]) + off:.2f} kg m-3')
		_DENSITY_FN = wrapped
		return _DENSITY_FN
	raise RuntimeError('could not resolve xroms.density, set EOS_MODE to '
					   'gsw, model or linearized')


def density_chunk(ds, sl, T, S, zr, mode=EOS_MODE, lon=None, lat=None,
				  Tm=None, Sm=None, rho_m=None, alpha=None, beta=None, k=None):
	"""
	Density on a chunk, flattened arrays in, flattened array out.

	T, S, zr are (nt, nz, M). Returns rho (nt, nz, M).
	"""
	if mode == 'model' and 'rho' in ds:
		r = ds.rho.isel(ocean_time=sl).values
		r = r.reshape(r.shape[0], r.shape[1], -1)
		return r + (1000.0 if np.nanmedian(r) < 100 else 0.0)
	if mode == 'xroms':
		return _resolve_density_fn()(T, S, zr)
	if mode == 'gsw' and HAS_GSW:
		p = gsw.p_from_z(zr, lat)
		SA = gsw.SA_from_SP(S, p, lon, lat)
		CT = gsw.CT_from_pt(SA, T)
		return np.asarray(gsw.rho(SA, CT, 0.0))
	return rho_m[k] + rho_m[k]*(beta[k]*(S - Sm[k]) - alpha[k]*(T - Tm[k]))


def window_mean_density(ds, wins, widx, flat, stride=MEAN_STRIDE):
	"""
	Window mean density at wet columns only, used purely as the reference
	state that the anomaly is taken from.

	Averaging the DENSITY rather than T and S is the right reference here,
	because the anomaly it defines is what enters the pressure integral.
	Stored flattened, so this is tens of megabytes rather than gigabytes.
	"""
	nz = ds.sizes['s_rho']
	nwet = int(flat.sum())
	lon = np.asarray(ds.lon_rho).ravel()[flat]
	lat = np.asarray(ds.lat_rho).ravel()[flat]

	rho_m = np.zeros((len(wins), nz, nwet))
	Tm = np.zeros_like(rho_m); Sm = np.zeros_like(rho_m)
	cnt = np.zeros(len(wins))

	sel = np.arange(0, widx.size, stride)
	print(f'    mean state from {sel.size} records (stride {stride})')
	for s in range(0, sel.size, 24):
		ix = sel[s:s+24]
		kk = widx[ix]
		good = kk >= 0
		if not good.any():
			continue
		ix = ix[good]; kk = kk[good]
		T = ds.temp.isel(ocean_time=ix).values
		S = ds.salt.isel(ocean_time=ix).values
		zr = ds.z_rho.isel(ocean_time=ix).values
		T = T.reshape(T.shape[0], nz, -1)[:, :, flat]
		S = S.reshape(S.shape[0], nz, -1)[:, :, flat]
		zr = zr.reshape(zr.shape[0], nz, -1)[:, :, flat]
		r = density_chunk(ds, None, T, S, zr, lon=lon, lat=lat)
		for m, k in enumerate(kk):
			rho_m[k] += r[m]; Tm[k] += T[m]; Sm[k] += S[m]; cnt[k] += 1
	cnt = np.maximum(cnt, 1)
	rho_m /= cnt[:, None, None]
	Tm /= cnt[:, None, None]
	Sm /= cnt[:, None, None]

	alpha = np.full_like(Tm, 1.7e-4)
	beta = np.full_like(Tm, 7.6e-4)
	if HAS_GSW and EOS_MODE == 'linearized':
		for k in range(len(wins)):
			zr0 = np.asarray(ds.z_rho.isel(ocean_time=0).values)
			zr0 = zr0.reshape(nz, -1)[:, flat]
			p = gsw.p_from_z(zr0, lat)
			SA = gsw.SA_from_SP(Sm[k], p, lon, lat)
			CT = gsw.CT_from_pt(SA, Tm[k])
			alpha[k] = gsw.alpha(SA, CT, p)
			beta[k] = gsw.beta(SA, CT, p)
	return Tm, Sm, rho_m, alpha, beta


# =====================================================================
# MAIN STREAMING PASS FOR ONE RUN
# =====================================================================
def process_run(ds, tag, wins, geom):
	"""
	One streaming pass. Returns the per window budget fields and the per
	block shelf break time series.
	"""
	lon, lat, h, mask, pm, pn, flat, ny, nx = (
		geom['lon'], geom['lat'], geom['h'], geom['mask'],
		geom['pm'], geom['pn'], geom['flat'], geom['ny'], geom['nx'])
	nz = ds.sizes['s_rho']
	nwet = int(flat.sum())
	nwin = len(wins)

	t = ds.ocean_time.values
	t_s = (t - t[0])/np.timedelta64(1, 's')
	widx = window_index(t, WINDOWS)
	dt_h = float(np.median(np.diff(t_s))/3600.)
	print(f'  {tag}: {t.size} records, {dt_h:.2f} h apart')

	print('  mean state pass')
	lon_f = np.asarray(ds.lon_rho).ravel()[flat]
	lat_f = np.asarray(ds.lat_rho).ravel()[flat]
	Tm, Sm, rho_m, alpha, beta = window_mean_density(ds, wins, widx, flat)

	ncoef = 1 + 2*len(CONSTITUENTS)
	nfield = 3*nz + 2                      # u_bc, v_bc, p_bc, u_bt, v_bt
	GtG = np.zeros((nwin, ncoef, ncoef))
	GtY = np.zeros((nwin, ncoef, nfield, nwet))
	print(f'  window accumulator {GtY.nbytes/1e9:.2f} GB')

	nbc = 1 + 2*len(BLOCK_CONSTITUENTS)
	blk = max(int(round(BLOCK_HOURS/dt_h)), 8)
	chunk = TIME_CHUNK or blk
	nblock = int(np.ceil(t.size/blk))
	Cblk = np.full(nblock, np.nan)
	tblk = np.empty(nblock, dtype=t.dtype)

	band = (h >= BREAK_BAND[0]) & (h <= BREAK_BAND[1]) & (mask > 0)
	aw = area_weights(pm, pn, mask)
	dHdx, dHdy = grad_h_components(h, pm, pn)

	bGtG = np.zeros((nblock, nbc, nbc))
	bGtY = np.zeros((nblock, nbc, 3, nwet), dtype=np.float32)
	print(f'  block accumulator {bGtY.nbytes/1e9:.2f} GB, '
		  f'{nblock} blocks of {blk} records')
	tic = _time.time()

	for s in range(0, t.size, chunk):
		e = min(s + chunk, t.size)
		sl = slice(s, e)
		nt_ = e - s

		u = ds.u.isel(ocean_time=sl).values
		v = ds.v.isel(ocean_time=sl).values
		ur, vr = uv_to_rho(u, v)
		del u, v

		T = ds.temp.isel(ocean_time=sl).values
		S = ds.salt.isel(ocean_time=sl).values
		zw = ds.z_w.isel(ocean_time=sl).values
		zr = ds.z_rho.isel(ocean_time=sl).values

		# flatten to wet columns before the arithmetic, roughly a third
		# of the grid is land and carries no budget
		ur = ur.reshape(nt_, nz, -1)[:, :, flat]
		vr = vr.reshape(nt_, nz, -1)[:, :, flat]
		T = T.reshape(nt_, nz, -1)[:, :, flat]
		S = S.reshape(nt_, nz, -1)[:, :, flat]
		zr = zr.reshape(nt_, nz, -1)[:, :, flat]
		zw = zw.reshape(nt_, nz + 1, -1)[:, :, flat]

		# Density ANOMALY relative to the window mean. p_bc is linear in
		# rho, so the mean part is constant in time and is absorbed by the
		# constant coefficient of the fit. Working with the anomaly keeps
		# p_bc near 1e3 Pa instead of 1e6, which is what makes float32
		# accumulation safe.
		k = np.where(widx[sl] >= 0, widx[sl], 0)
		rho = density_chunk(ds, sl, T, S, zr, lon=lon_f, lat=lat_f,
							Tm=Tm, Sm=Sm, rho_m=rho_m,
							alpha=alpha, beta=beta, k=k)
		rho_a = rho - rho_m[k]
		del T, S, zr, rho

		p_bc = baroclinic_pressure(rho_a, zw)
		del rho_a

		dz = np.diff(zw, axis=1)
		H = np.sum(dz, axis=1)
		u_bt = np.sum(ur*dz, axis=1)/np.maximum(H, 1e-6)
		v_bt = np.sum(vr*dz, axis=1)/np.maximum(H, 1e-6)
		u_bc = ur - u_bt[:, None, :]
		v_bc = vr - v_bt[:, None, :]
		del ur, vr

		Y = np.empty((nt_, nfield, nwet), dtype=np.float32)
		Y[:, 0:nz] = u_bc
		Y[:, nz:2*nz] = v_bc
		Y[:, 2*nz:3*nz] = p_bc
		Y[:, 3*nz] = u_bt
		Y[:, 3*nz+1] = v_bt
		del u_bc, v_bc, p_bc, u_bt, v_bt

		# window accumulation
		X = design_matrix(t_s[sl], CONSTITUENTS)
		for kk in range(nwin):
			m = widx[sl] == kk
			if not m.any():
				continue
			Xk = X[m]
			GtG[kk] += Xk.T @ Xk
			GtY[kk] += np.einsum('tc,tfm->cfm', Xk, Y[m], optimize=True)

		# block accumulation, for the panel (g) time series
		Xb = design_matrix(t_s[sl], BLOCK_CONSTITUENTS)
		Yb = np.stack([Y[:, 3*nz], Y[:, 3*nz+1], Y[:, 2*nz]], axis=1)
		bid = np.arange(s, e)//blk
		for g in np.unique(bid):
			m = bid == g
			Xg = Xb[m]
			bGtG[g] += Xg.T @ Xg
			bGtY[g] += np.einsum('tc,tfm->cfm', Xg, Yb[m],
								 optimize=True).astype(np.float32)
		del Y, Yb

		if (s//chunk) % 20 == 0:
			done = e/t.size
			el = _time.time() - tic
			print(f'    {100*done:5.1f}%  {el:6.0f} s elapsed, '
				  f'{el/max(done,1e-6)-el:6.0f} s to go', flush=True)

	print('  reducing the block series')
	for g in range(nblock):
		if bGtG[g, 0, 0] <= nbc:
			continue
		Cblk[g], tblk[g] = _block_conversion(
			bGtG[g], bGtY[g].astype(np.float64), flat, ny, nx,
			dHdx, dHdy, band, aw, t[min(g*blk, t.size - 1)])

	# -----------------------------------------------------------------
	# solve and form the budget
	# -----------------------------------------------------------------
	print('  solving the normal equations')
	out = {}
	for k, wname in enumerate(wins):
		if GtG[k, 0, 0] < ncoef:
			print(f'    {wname}: no records, skipped')
			continue
		c = solve_normal(GtG[k], GtY[k])
		# M2 is coefficients 1 and 2
		a, b = c[1], c[2]

		def unflat(v):
			out_ = np.full(ny*nx, np.nan); out_[flat] = v
			return out_.reshape(ny, nx)

		a_ubc, b_ubc = a[0:nz], b[0:nz]
		a_vbc, b_vbc = a[nz:2*nz], b[nz:2*nz]
		a_pbc, b_pbc = a[2*nz:3*nz], b[2*nz:3*nz]
		a_ubt, b_ubt = a[3*nz], b[3*nz]
		a_vbt, b_vbt = a[3*nz+1], b[3*nz+1]

		# depth integrated baroclinic flux, grid components
		zw_m = ds.z_w.isel(ocean_time=0).values.reshape(nz+1, -1)[:, flat]
		dz = np.diff(zw_m, axis=0)
		Fxi = np.sum(cycle_mean(a_ubc, b_ubc, a_pbc, b_pbc)*dz, axis=0)
		Feta = np.sum(cycle_mean(a_vbc, b_vbc, a_pbc, b_pbc)*dz, axis=0)

		# conversion, using the bottom baroclinic pressure
		upb = cycle_mean(a_ubt, b_ubt, a_pbc[0], b_pbc[0])
		vpb = cycle_mean(a_vbt, b_vbt, a_pbc[0], b_pbc[0])
		Fxi2, Feta2 = unflat(Fxi), unflat(Feta)
		C = -(unflat(upb)*dHdx + unflat(vpb)*dHdy)

		divF = divergence(np.nan_to_num(Fxi2), np.nan_to_num(Feta2), pm, pn)
		divF = np.where(mask > 0, divF, np.nan)
		D = C - divF

		out[wname] = dict(C=C, Fxi=Fxi2, Feta=Feta2, divF=divF, D=D,
						  pbc_bot=unflat(np.hypot(a_pbc[0], b_pbc[0])))

	return out, Cblk, tblk, blk


def _block_conversion(bGtG, bGtY, flat, ny, nx, dHdx, dHdy, band, aw, tstamp):
	"""Area averaged conversion over the shelf break band for one block."""
	c = solve_normal(bGtG, bGtY)
	a, b = c[1], c[2]
	upb = cycle_mean(a[0], b[0], a[2], b[2])
	vpb = cycle_mean(a[1], b[1], a[2], b[2])
	U = np.full(ny*nx, np.nan); U[flat] = upb
	V = np.full(ny*nx, np.nan); V[flat] = vpb
	C = -(U.reshape(ny, nx)*dHdx + V.reshape(ny, nx)*dHdy)
	w = np.where(band & np.isfinite(C), aw, 0.0)
	tot = w.sum()
	return (np.nansum(np.nan_to_num(C)*w)/tot if tot > 0 else np.nan), tstamp


# =====================================================================
# MAIN
# =====================================================================
def main():
	print('opening runs')
	dsr = open_run(REALISTIC, chunks={'ocean_time': BLOCK_HOURS})
	dsc = open_run(CONTROL, chunks={'ocean_time': BLOCK_HOURS})

	lon = np.asarray(dsr.lon_rho); lat = np.asarray(dsr.lat_rho)
	h = np.asarray(dsr.h)
	mask = np.asarray(dsr.mask_rho) if 'mask_rho' in dsr else np.ones_like(h)
	pm = np.asarray(dsr.pm); pn = np.asarray(dsr.pn)
	ny, nx = h.shape
	flat = mask.ravel() > 0
	geom = dict(lon=lon, lat=lat, h=h, mask=mask, pm=pm, pn=pn,
				flat=flat, ny=ny, nx=nx)
	wins = list(WINDOWS.keys())
	aw = area_weights(pm, pn, mask)

	res = {}
	for tag, ds in (('ref', dsr), ('ctrl', dsc)):
		if tag not in RUNS:
			continue
		tic = _time.time()
		res[tag] = process_run(ds, tag, wins, geom)
		print(f'  {tag} done in {(_time.time()-tic)/60:.1f} min')

	# -----------------------------------------------------------------
	# shelf break path and the onshore flux across it
	# -----------------------------------------------------------------
	blon, blat, bdist = extract_isobath(lon, lat,
										np.where(mask > 0, h, np.nan),
										BREAK_ISOBATH)
	tree, jj, ii = build_tree(lon, lat, mask)
	bj, bi = nearest_ji(tree, jj, ii, blon, blat)
	dHdx, dHdy = grad_h_components(h, pm, pn)
	gmag = np.hypot(dHdx, dHdy)
	nxo = np.where(gmag > 0, dHdx/np.maximum(gmag, 1e-12), 0.)   # offshore
	nyo = np.where(gmag > 0, dHdy/np.maximum(gmag, 1e-12), 0.)
	dl = np.gradient(bdist)*1000.                                 # m

	# -----------------------------------------------------------------
	# assemble
	# -----------------------------------------------------------------
	print('assembling')
	shelf = (h < SHELF_MAX) & (mask > 0)
	slope = (h >= SHELF_MAX) & (h <= SLOPE_MAX) & (mask > 0)
	nwin = len(wins)

	fields = ['C', 'Fxi', 'Feta', 'divF', 'D', 'pbc_bot']
	arr = {f'{f}_{tag}': np.full((nwin, ny, nx), np.nan)
		   for f in fields for tag in RUNS}
	budg = {f'{q}_{reg}_{tag}': np.full(nwin, np.nan)
			for q in ('C', 'divF', 'D') for reg in ('shelf', 'slope')
			for tag in RUNS}
	onsh = {f'onshore_{tag}': np.full(nwin, np.nan) for tag in RUNS}
	onshp = {f'onshore_prof_{tag}': np.full((nwin, blon.size), np.nan)
			 for tag in RUNS}

	for tag in RUNS:
		out, Cblk, tblk, blk = res[tag]
		for k, w in enumerate(wins):
			if w not in out:
				continue
			d = out[w]
			for f in fields:
				arr[f'{f}_{tag}'][k] = d[f]
			for reg, m in (('shelf', shelf), ('slope', slope)):
				for q, v in (('C', d['C']), ('divF', d['divF']), ('D', d['D'])):
					budg[f'{q}_{reg}_{tag}'][k] = np.nansum(
						np.where(m & np.isfinite(v), v*aw, 0.))/1e6   # MW
			# onshore flux across the break, W per metre then integrated
			fon = -(d['Fxi'][bj, bi]*nxo[bj, bi] + d['Feta'][bj, bi]*nyo[bj, bi])
			onshp[f'onshore_prof_{tag}'][k] = fon
			onsh[f'onshore_{tag}'][k] = np.nansum(fon*dl)/1e6        # MW

	# spring and neap, from the shelf break sea surface height
	jb, ib = nearest_ji(tree, jj, ii, -48.0, 2.0)
	zeta = np.asarray(dsr.zeta.isel(eta_rho=int(jb[0]),
									xi_rho=int(ib[0])).values).squeeze()
	t = dsr.ocean_time.values
	dt_s = float(np.median(np.diff(t))/np.timedelta64(1, 's'))
	fs = 1.0/dt_s
	b, a = butter(3, [ (1/(13*3600.))/(fs/2), (1/(11*3600.))/(fs/2) ],
				  btype='band')
	env = np.abs(hilbert(filtfilt(b, a, zeta - np.nanmean(zeta))))

	_, Cblk_r, tblk_r, blk = res['ref']
	_, Cblk_c, _, _ = res['ctrl'] if 'ctrl' in res else (None, None, None, None)

	# fortnightly variability band of the control run, the noise floor
	ok = np.isfinite(Cblk_c) if Cblk_c is not None else np.array([False])
	if ok.sum() > 8:
		nper = max(int(round(14.77*24/BLOCK_HOURS)), 2)
		cyc = [np.nanmean(Cblk_c[i:i+nper])
			   for i in range(0, ok.size - nper, nper)]
		noise = float(np.nanstd(cyc))
	else:
		noise = np.nan

	coords = dict(
		window=('window', wins),
		eta_rho=('eta_rho', np.arange(ny)),
		xi_rho=('xi_rho', np.arange(nx)),
		break_pt=('break_pt', np.arange(blon.size)),
		block=('block', np.arange(Cblk_r.size)),
		time=('time', t),
	)
	dv = {k: (('window', 'eta_rho', 'xi_rho'), v) for k, v in arr.items()}
	dv.update({k: ('window', v) for k, v in budg.items()})
	dv.update({k: ('window', v) for k, v in onsh.items()})
	dv.update({k: (('window', 'break_pt'), v) for k, v in onshp.items()})
	dv.update(dict(
		lon_rho=(('eta_rho', 'xi_rho'), lon),
		lat_rho=(('eta_rho', 'xi_rho'), lat),
		h=(('eta_rho', 'xi_rho'), h),
		mask_rho=(('eta_rho', 'xi_rho'), mask),
		angle=(('eta_rho', 'xi_rho'),
			   np.asarray(dsr.angle) if 'angle' in dsr else np.zeros_like(h)),
		break_lon=('break_pt', blon),
		break_lat=('break_pt', blat),
		break_dist=('break_pt', bdist),
		C_block_ref=('block', Cblk_r),
		C_block_ctrl=('block', Cblk_c if Cblk_c is not None
					  else np.full_like(Cblk_r, np.nan)),
		block_time=('block', tblk_r),
		springneap_env=('time', env),
	))

	ds_out = xr.Dataset(dv, coords=coords)
	ds_out.attrs.update(dict(
		realistic=REALISTIC, control=CONTROL,
		eos_mode=EOS_MODE, block_hours=BLOCK_HOURS,
		constituents=','.join(CONSTITUENTS),
		shelf_max=SHELF_MAX, slope_max=SLOPE_MAX,
		break_isobath=BREAK_ISOBATH,
		break_band_lo=BREAK_BAND[0], break_band_hi=BREAK_BAND[1],
		ctrl_fortnightly_sd=noise,
		windows=';'.join(f'{k}:{v[0]}..{v[1]}' for k, v in WINDOWS.items()),
		gen_sites_main=','.join(GEN_SITES_MAIN),
		note='C = <p_bc(-H) w_bt(-H)>, w_bt(-H) = -u_bt . grad H, so '
			 'positive C is barotropic to baroclinic. D_bc is the residual '
			 'C - div F_bc and aggregates dissipation with everything the '
			 'diagnostics cannot separate.',
	))
	enc = {v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars}
	ds_out.to_netcdf(OUTFILE, encoding=enc)
	print('wrote', OUTFILE)

	# -----------------------------------------------------------------
	# summary and the sign check
	# -----------------------------------------------------------------
	print('\n--- summary, all values in MW ---')
	hdr = f'{"window":>11} {"region":>6} ' + ' '.join(
		f'{q+"_"+tg:>12}' for q in ('C', 'divF', 'D') for tg in RUNS)
	print(hdr)
	for k, w in enumerate(wins):
		for reg in ('shelf', 'slope'):
			row = f'{w:>11} {reg:>6} '
			for q in ('C', 'divF', 'D'):
				for tg in RUNS:
					row += f'{budg[f"{q}_{reg}_{tg}"][k]:12.1f} '
			print(row)
	print()
	for k, w in enumerate(wins):
		for reg in ('shelf', 'slope'):
			line = f'{w:>11} {reg:>6}  exported fraction  '
			for tg in RUNS:
				C = budg[f'C_{reg}_{tg}'][k]
				Fd = budg[f'divF_{reg}_{tg}'][k]
				line += f'{tg} {100*Fd/C if C else np.nan:5.0f}%   '
			print(line)
	print()
	for k, w in enumerate(wins):
		line = f'{w:>11}  onshore flux across the {BREAK_ISOBATH:.0f} m isobath  '
		for tg in RUNS:
			line += f'{tg} {onsh[f"onshore_{tg}"][k]:7.1f} MW   '
		print(line)
	print(f'\ncontrol fortnightly standard deviation of the shelf break '
		  f'conversion: {noise:.3g} W m-2')

	for tg in RUNS:
		s = np.nansum(np.where(slope, arr[f'C_{tg}'][0]*aw, 0.))
		if s < 0:
			warnings.warn(f'slope integrated conversion is negative in {tg}. '
						  'Check the sign convention on grad H before using '
						  'these numbers.')
		else:
			print(f'sign check, {tg}: slope integrated C positive, '
				  'barotropic to baroclinic as expected')


if __name__ == '__main__':
	main()