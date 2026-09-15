#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig03_compute.py
=================================================================
Figure 3  -  Conversion, its amplitude and its alignment
Amazon shelf internal tide manuscript, version 3

Computes and stores everything panels (a) to (i) need:

  (a to c) conversion C for the three windows, REF
  (d)      the total log ratio between the runs at peak discharge
  (e)      its amplitude term
  (f)      its alignment term, on the same scale as (d) and (e)
  (g)      cos(dphi) against criticality along the 200 m isobath
  (h)      the bottom pressure amplitude against N_b on the same path
  (i)      signed decomposition of C by depth band with kappa

Outputs
	fig03_data.nc         the figure fields, small
	fig03_harmonics.nc    the M2 harmonic coefficients of u_bc, v_bc,
						  p_bc and u_bt, written once so that Figures 4
						  and 5 need no further pass over the model
	fig03_report.txt      written by fig03_report.py, called at the end

Method
------
Everything is quadratic in the harmonic coefficients, so nothing is held
as a time series. One streaming pass per run accumulates the least
squares normal equations at M2 and S2 for the baroclinic velocity, the
baroclinic pressure and the barotropic velocity, after which

	<x y> over a tidal cycle = 0.5 (a_x a_y + b_x b_y)

gives the conversion directly. Writing the two factors of

	C = <p_bc(-H) w_bt(-H)>,     w_bt(-H) = -u_bt . grad H

as harmonics turns that product into

	C = 0.5 P W cos(dphi)

with P the amplitude of the baroclinic pressure at the bed, W the
amplitude of the barotropic vertical velocity, which the two runs share,
and cos(dphi) the alignment between them. The ratio between the runs
then factorises exactly,

	ln(C_R / C_C) = ln(P_R / P_C) + ln(cos_R / cos_C)

with no residual, since W cancels. That identity is checked cell by cell
and the check is printed.

Sign convention: H is positive downward and the bed is at z = -H, so the
kinematic condition gives w = -u . grad H and positive C is barotropic
to baroclinic. The slope integral is checked for sign at the end.
=================================================================
"""

import os
import time as _time
import warnings
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use('Agg')

try:
	import xroms
	HAS_XROMS = True
except ImportError:
	HAS_XROMS = False

from fig01_compute import (
	REALISTIC, CONTROL, WINDOWS, BREAK_ISOBATH, SHELF_MAX_DEPTH,
	G, RHO0, OMEGA_M2, OMEGA_S2, T_M2,
	open_run, extract_isobath, build_tree, nearest_ji, HAS_GSW,
)

if HAS_GSW:
	import gsw


# =====================================================================
# CONFIGURATION
# =====================================================================
OUTFILE  = 'fig03_data.nc'
HARMFILE = 'fig03_harmonics.nc'
FIG01    = 'fig01_data.nc'          # criticality and N_b along the break

SAVE_HARMONICS = True    # False skips the companion file and forces a new
						 # pass over the model when Figures 4 and 5 are built

EOS_MODE = 'xroms'       # xroms, model, gsw or linearized
MEAN_STRIDE = 12         # stride of the cheap window mean pass
BLOCK_HOURS = 72         # three day blocks, matching Section 2.3
TIME_CHUNK = 24          # records read at once

CONSTITUENTS = {'M2': OMEGA_M2, 'S2': OMEGA_S2}
BLOCK_CONSTITUENTS = {'M2': OMEGA_M2}

GEN_BAND = (100., 1000.)          # m, the band the factorisation is quoted on
BANDS = [('shelf, h < 250 m',            0.,   250., 'shelf'),
		 ('upper slope, 250 to 1000 m',  250., 1000., 'slope'),
		 ('deep, 1000 to 3500 m',       1000., 3500., 'deep')]

ALPHA_BINS = np.linspace(0.4, 2.6, 23)          # panel (g)
NB_BINS = np.logspace(-5.2, -3.2, 21)           # panel (h), N_b^2 in s-2

RUNS = ('ref', 'ctrl')


# =====================================================================
# HELPERS
# =====================================================================
def uv_to_rho(u, v):
	"""C grid velocity to rho points, averaging the two adjacent faces."""
	ny = v.shape[-2] + 1
	nx = u.shape[-1] + 1
	ur = np.empty(u.shape[:-2] + (ny, nx), dtype=u.dtype)
	vr = np.empty(v.shape[:-2] + (ny, nx), dtype=v.dtype)
	ur[..., 1:-1] = 0.5*(u[..., :-1] + u[..., 1:])
	ur[..., 0] = u[..., 0]; ur[..., -1] = u[..., -1]
	vr[..., 1:-1, :] = 0.5*(v[..., :-1, :] + v[..., 1:, :])
	vr[..., 0, :] = v[..., 0, :]; vr[..., -1, :] = v[..., -1, :]
	return ur, vr


def baroclinic_pressure(rho, zw):
	"""
	Baroclinic perturbation pressure on rho levels,

		P(z) = g int_z^0 rho dz',      p_bc = P - (1/H) int_{-H}^0 P dz .

	The depth mean removal is what makes it baroclinic and it also removes
	the free surface term, which is depth independent.
	"""
	dz = np.diff(zw, axis=1)
	P = G*np.cumsum((rho*dz)[:, ::-1, :], axis=1)[:, ::-1, :]
	H = np.sum(dz, axis=1, keepdims=True)
	Pbar = np.sum(P*dz, axis=1, keepdims=True)/np.maximum(H, 1e-6)
	return P - Pbar


def design_matrix(t_s, consts):
	cols = [np.ones_like(t_s)]
	for w in consts.values():
		cols.append(np.cos(w*t_s))
		cols.append(np.sin(w*t_s))
	return np.stack(cols, axis=1)


def solve_normal(GtG, GtY, ridge=1e-10):
	A = GtG + ridge*np.trace(GtG)/GtG.shape[0]*np.eye(GtG.shape[0])
	shp = GtY.shape
	return np.linalg.solve(A, GtY.reshape(shp[0], -1)).reshape(shp)


def cycle_mean(a1, b1, a2, b2):
	return 0.5*(a1*a2 + b1*b2)


def grad_h_components(h, pm, pn):
	return np.gradient(h, axis=1)*pm, np.gradient(h, axis=0)*pn


def area_weights(pm, pn, mask):
	return np.where(mask > 0, 1.0/(pm*pn), 0.0)


def window_index(times, windows):
	idx = np.full(times.size, -1, dtype=int)
	for k, (name, (t0, t1)) in enumerate(windows.items()):
		sel = (times >= np.datetime64(t0)) & (times <= np.datetime64(t1))
		idx[sel] = k
	return idx


def wrap_pi(x):
	return (x + np.pi) % (2*np.pi) - np.pi


def binned(x, y, bins):
	"""Median and quartiles of y in bins of x."""
	ctr = 0.5*(bins[1:] + bins[:-1])
	med = np.full(ctr.size, np.nan)
	q25 = np.full(ctr.size, np.nan)
	q75 = np.full(ctr.size, np.nan)
	ok = np.isfinite(x) & np.isfinite(y)
	for k in range(ctr.size):
		s = ok & (x >= bins[k]) & (x < bins[k + 1])
		if s.sum() > 6:
			med[k] = np.nanmedian(y[s])
			q25[k] = np.nanpercentile(y[s], 25)
			q75[k] = np.nanpercentile(y[s], 75)
	return ctr, med, q25, q75


# =====================================================================
# EQUATION OF STATE
# =====================================================================
_DENSITY_FN = None


def _resolve_density_fn():
	"""xroms' implementation of the ROMS nonlinear equation of state."""
	global _DENSITY_FN
	if _DENSITY_FN is not None:
		return _DENSITY_FN
	if not HAS_XROMS:
		raise RuntimeError('xroms not available, set EOS_MODE to gsw, '
						   'model or linearized')
	for mod in (xroms, getattr(xroms, 'roms_seawater', None)):
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
	"""Window mean density at wet columns, the reference the anomaly uses."""
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
	return Tm, Sm, rho_m, alpha, beta


# =====================================================================
# STREAMING PASS FOR ONE RUN
# =====================================================================
def process_run(ds, tag, wins, geom):
	"""
	One streaming pass. Returns the per window harmonic coefficients, the
	window mean z_w, and the per block conversion over the generation band.
	"""
	h, mask, pm, pn = geom['h'], geom['mask'], geom['pm'], geom['pn']
	flat, ny, nx = geom['flat'], geom['ny'], geom['nx']
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
	nfield = 3*nz + 2                       # u_bc, v_bc, p_bc, u_bt, v_bt
	GtG = np.zeros((nwin, ncoef, ncoef))
	GtY = np.zeros((nwin, ncoef, nfield, nwet))
	zwm = np.zeros((nwin, nz + 1, nwet))
	zwc = np.zeros(nwin)
	print(f'  window accumulator {GtY.nbytes/1e9:.2f} GB')

	nbc = 1 + 2*len(BLOCK_CONSTITUENTS)
	blk = max(int(round(BLOCK_HOURS/dt_h)), 8)
	chunk = TIME_CHUNK or blk
	nblock = int(np.ceil(t.size/blk))
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

		ur = ur.reshape(nt_, nz, -1)[:, :, flat]
		vr = vr.reshape(nt_, nz, -1)[:, :, flat]
		T = T.reshape(nt_, nz, -1)[:, :, flat]
		S = S.reshape(nt_, nz, -1)[:, :, flat]
		zr = zr.reshape(nt_, nz, -1)[:, :, flat]
		zw = zw.reshape(nt_, nz + 1, -1)[:, :, flat]

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

		X = design_matrix(t_s[sl], CONSTITUENTS)
		for kk in range(nwin):
			m = widx[sl] == kk
			if not m.any():
				continue
			Xk = X[m]
			GtG[kk] += Xk.T @ Xk
			GtY[kk] += np.einsum('tc,tfm->cfm', Xk, Y[m], optimize=True)
			zwm[kk] += zw[m].sum(axis=0)
			zwc[kk] += int(m.sum())

		Xb = design_matrix(t_s[sl], BLOCK_CONSTITUENTS)
		Yb = np.stack([Y[:, 3*nz], Y[:, 3*nz+1], Y[:, 2*nz]], axis=1)
		bid = np.arange(s, e)//blk
		for g in np.unique(bid):
			m = bid == g
			Xg = Xb[m]
			bGtG[g] += Xg.T @ Xg
			bGtY[g] += np.einsum('tc,tfm->cfm', Xg, Yb[m],
								 optimize=True).astype(np.float32)
		del Y, Yb, zw

		if (s//chunk) % 20 == 0:
			done = e/t.size
			el = _time.time() - tic
			print(f'    {100*done:5.1f}%  {el:6.0f} s elapsed, '
				  f'{el/max(done, 1e-6) - el:6.0f} s to go', flush=True)

	zwm /= np.maximum(zwc, 1)[:, None, None]

	print('  solving the normal equations')
	coef = {}
	for k, wname in enumerate(wins):
		if GtG[k, 0, 0] < ncoef:
			warnings.warn(f'{wname}/{tag}: no records, skipped')
			continue
		c = solve_normal(GtG[k], GtY[k])
		coef[wname] = (c[1], c[2])          # M2 cosine and sine coefficients

	return coef, zwm, (bGtG, bGtY, blk, nblock, t)


def block_series(bGtG, bGtY, blk, nblock, t, flat, ny, nx, dHdx, dHdy,
				 band, aw):
	"""Area averaged conversion over the generation band, block by block."""
	Cblk = np.full(nblock, np.nan)
	tblk = np.empty(nblock, dtype=t.dtype)
	for g in range(nblock):
		tblk[g] = t[min(g*blk, t.size - 1)]
		if bGtG[g, 0, 0] <= bGtY.shape[1]:
			continue
		c = solve_normal(bGtG[g], bGtY[g].astype(np.float64))
		a, b = c[1], c[2]
		upb = cycle_mean(a[0], b[0], a[2], b[2])
		vpb = cycle_mean(a[1], b[1], a[2], b[2])
		U = np.full(ny*nx, np.nan); U[flat] = upb
		V = np.full(ny*nx, np.nan); V[flat] = vpb
		C = -(U.reshape(ny, nx)*dHdx + V.reshape(ny, nx)*dHdy)
		w = np.where(band & np.isfinite(C), aw, 0.0)
		tot = w.sum()
		Cblk[g] = np.nansum(np.nan_to_num(C)*w)/tot if tot > 0 else np.nan
	return Cblk, tblk


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
	nz = dsr.sizes['s_rho']
	flat = mask.ravel() > 0
	wet = mask > 0
	wins = list(WINDOWS.keys())
	nwin = len(wins)
	aw = area_weights(pm, pn, mask)
	dHdx, dHdy = grad_h_components(h, pm, pn)
	geom = dict(h=h, mask=mask, pm=pm, pn=pn, flat=flat, ny=ny, nx=nx)

	gband = (h >= GEN_BAND[0]) & (h <= GEN_BAND[1]) & wet

	def unflat(v):
		a = np.full(ny*nx, np.nan); a[flat] = v
		return a.reshape(ny, nx)

	# -----------------------------------------------------------------
	# the streaming pass
	# -----------------------------------------------------------------
	raw = {}
	for tag, ds in (('ref', dsr), ('ctrl', dsc)):
		tic = _time.time()
		raw[tag] = process_run(ds, tag, wins, geom)
		print(f'  {tag} done in {(_time.time() - tic)/60:.1f} min')

	# -----------------------------------------------------------------
	# the factorisation, cell by cell
	# -----------------------------------------------------------------
	print('forming C, P, W and the alignment')
	F = {f'{q}_{tag}': np.full((nwin, ny, nx), np.nan)
		 for q in ('C', 'P', 'W', 'cosdphi', 'dphi') for tag in RUNS}
	harm = {}

	for tag in RUNS:
		coef, zwm, _ = raw[tag]
		harm[tag] = dict(zwm=zwm, a={}, b={})
		for k, wname in enumerate(wins):
			if wname not in coef:
				continue
			a, b = coef[wname]
			a_pb, b_pb = a[2*nz], b[2*nz]              # bottom rho level
			a_ub, b_ub = a[3*nz], b[3*nz]
			a_vb, b_vb = a[3*nz+1], b[3*nz+1]

			dhx = dHdx.ravel()[flat]
			dhy = dHdy.ravel()[flat]
			a_w = -(a_ub*dhx + a_vb*dhy)
			b_w = -(b_ub*dhx + b_vb*dhy)

			P = np.hypot(a_pb, b_pb)
			W = np.hypot(a_w, b_w)
			dphi = wrap_pi(np.arctan2(b_pb, a_pb) - np.arctan2(b_w, a_w))
			C = 0.5*(a_pb*a_w + b_pb*b_w)

			F[f'C_{tag}'][k] = unflat(C)
			F[f'P_{tag}'][k] = unflat(P)
			F[f'W_{tag}'][k] = unflat(W)
			F[f'cosdphi_{tag}'][k] = unflat(np.cos(dphi))
			F[f'dphi_{tag}'][k] = unflat(np.degrees(dphi))

			harm[tag]['a'][wname] = a
			harm[tag]['b'][wname] = b

	# identity check, C against 0.5 P W cos(dphi)
	for tag in RUNS:
		lhs = F[f'C_{tag}']
		rhs = 0.5*F[f'P_{tag}']*F[f'W_{tag}']*F[f'cosdphi_{tag}']
		den = np.nanmax(np.abs(lhs))
		err = np.nanmax(np.abs(lhs - rhs))/max(den, 1e-30)
		print(f'  identity check {tag}: max relative error {err:.2e}')
		if err > 1e-8:
			warnings.warn(f'the factorisation does not close in {tag}')

	# -----------------------------------------------------------------
	# log ratios between the runs, valid where C is positive in both
	# -----------------------------------------------------------------
	ok = (F['C_ref'] > 0) & (F['C_ctrl'] > 0) \
		 & (F['cosdphi_ref'] > 0) & (F['cosdphi_ctrl'] > 0)
	with np.errstate(invalid='ignore', divide='ignore'):
		ln_tot = np.where(ok, np.log(F['C_ref']/F['C_ctrl']), np.nan)
		ln_amp = np.where(ok, np.log(F['P_ref']/F['P_ctrl']), np.nan)
		ln_ali = np.where(ok, np.log(F['cosdphi_ref']/F['cosdphi_ctrl']),
						  np.nan)

	# -----------------------------------------------------------------
	# band integrals, signed decomposition and the weighted factorisation
	# -----------------------------------------------------------------
	print('integrating by band')
	nband = len(BANDS)
	bnames = [b[3] for b in BANDS]
	integ = {f'{q}_{tag}': np.full((nwin, nband), np.nan)
			 for q in ('Cpos', 'Cneg', 'Cnet', 'kappa') for tag in RUNS}
	fac = {q: np.full((nwin, nband), np.nan)
		   for q in ('ln_tot', 'ln_amp', 'ln_ali', 'ratio_int', 'amp_share',
					 'area_frac')}

	for k in range(nwin):
		for ib, (nm, lo, hi, short) in enumerate(BANDS):
			m = wet & (h >= lo) & (h < hi)
			for tag in RUNS:
				C = F[f'C_{tag}'][k]
				w = np.where(m & np.isfinite(C), aw, 0.0)
				c = np.nan_to_num(C)
				pos = float(np.nansum(np.where(c > 0, c, 0.)*w)/1e6)
				neg = float(np.nansum(np.where(c < 0, c, 0.)*w)/1e6)
				tot = pos - neg
				integ[f'Cpos_{tag}'][k, ib] = pos
				integ[f'Cneg_{tag}'][k, ib] = neg
				integ[f'Cnet_{tag}'][k, ib] = pos + neg
				integ[f'kappa_{tag}'][k, ib] = 1 - abs(pos + neg)/tot \
					if tot else np.nan

			s = m & ok[k] & np.isfinite(ln_tot[k])
			if s.sum() > 10:
				wt = F['C_ref'][k][s]*aw[s]
				wt = np.where(np.isfinite(wt) & (wt > 0), wt, 0.0)
				tw = wt.sum()
				if tw > 0:
					fac['ln_tot'][k, ib] = float(np.sum(wt*ln_tot[k][s])/tw)
					fac['ln_amp'][k, ib] = float(np.sum(wt*ln_amp[k][s])/tw)
					fac['ln_ali'][k, ib] = float(np.sum(wt*ln_ali[k][s])/tw)
					fac['amp_share'][k, ib] = (
						fac['ln_amp'][k, ib]/fac['ln_tot'][k, ib]
						if abs(fac['ln_tot'][k, ib]) > 1e-9 else np.nan)
			fac['area_frac'][k, ib] = float(
				np.sum(np.where(s, aw, 0.))/max(np.sum(np.where(m, aw, 0.)), 1e-9))
			cr = integ['Cpos_ref'][k, ib]; cc = integ['Cpos_ctrl'][k, ib]
			fac['ratio_int'][k, ib] = cr/cc if cc else np.nan

	# seasonal factorisation within each run, relative to the rising window
	seas = {f'{q}_{tag}': np.full((nwin, nband), np.nan)
			for q in ('ln_tot', 'ln_amp', 'ln_ali') for tag in RUNS}
	for tag in RUNS:
		C0, P0, X0 = (F[f'C_{tag}'][0], F[f'P_{tag}'][0],
					  F[f'cosdphi_{tag}'][0])
		for k in range(nwin):
			Ck, Pk, Xk = (F[f'C_{tag}'][k], F[f'P_{tag}'][k],
						  F[f'cosdphi_{tag}'][k])
			good = (C0 > 0) & (Ck > 0) & (X0 > 0) & (Xk > 0)
			for ib, (nm, lo, hi, short) in enumerate(BANDS):
				m = wet & (h >= lo) & (h < hi) & good
				if m.sum() < 10:
					continue
				wt = Ck[m]*aw[m]
				tw = wt.sum()
				if tw <= 0:
					continue
				seas[f'ln_tot_{tag}'][k, ib] = float(
					np.sum(wt*np.log(Ck[m]/C0[m]))/tw)
				seas[f'ln_amp_{tag}'][k, ib] = float(
					np.sum(wt*np.log(Pk[m]/P0[m]))/tw)
				seas[f'ln_ali_{tag}'][k, ib] = float(
					np.sum(wt*np.log(Xk[m]/X0[m]))/tw)

	# -----------------------------------------------------------------
	# along break sampling and the two binned relations
	# -----------------------------------------------------------------
	print('sampling along the shelf break')
	blon, blat, bdist = extract_isobath(lon, lat, np.where(wet, h, np.nan),
										BREAK_ISOBATH)
	tree, jj, ii = build_tree(lon, lat, mask)
	bj, bi = nearest_ji(tree, jj, ii, blon, blat)
	nb = blon.size

	alpha_b = {t: np.full((nwin, nb), np.nan) for t in RUNS}
	nb2_b = {t: np.full((nwin, nb), np.nan) for t in RUNS}
	if os.path.exists(FIG01):
		d1 = xr.open_dataset(FIG01)
		for t in RUNS:
			alpha_b[t] = d1[f'alpha_{t}_break'].values
			nb2_b[t] = d1[f'n2bot_{t}_break'].values
		d1.close()
		print(f'  criticality and N_b read from {FIG01}')
	else:
		warnings.warn(f'{FIG01} not found, panels (g) and (h) will be empty')

	brk = {}
	for q in ('C', 'P', 'W', 'cosdphi', 'dphi'):
		for t in RUNS:
			brk[f'{q}_{t}_break'] = F[f'{q}_{t}'][:, bj, bi]
	for t in RUNS:
		brk[f'alpha_{t}_break'] = alpha_b[t]
		brk[f'nb2_{t}_break'] = nb2_b[t]

	acen = 0.5*(ALPHA_BINS[1:] + ALPHA_BINS[:-1])
	ncen = np.sqrt(NB_BINS[1:]*NB_BINS[:-1])
	bing = {f'{q}_{t}': np.full((nwin, acen.size), np.nan)
			for q in ('med', 'q25', 'q75') for t in RUNS}
	binh = {f'{q}_{t}': np.full((nwin, ncen.size), np.nan)
			for q in ('med', 'q25', 'q75') for t in RUNS}
	for k in range(nwin):
		for t in RUNS:
			_, m1, l1, u1 = binned(alpha_b[t][k], brk[f'cosdphi_{t}_break'][k],
								   ALPHA_BINS)
			bing[f'med_{t}'][k] = m1; bing[f'q25_{t}'][k] = l1
			bing[f'q75_{t}'][k] = u1
			_, m2, l2, u2 = binned(nb2_b[t][k], brk[f'P_{t}_break'][k],
								   NB_BINS)
			binh[f'med_{t}'][k] = m2; binh[f'q25_{t}'][k] = l2
			binh[f'q75_{t}'][k] = u2

	# -----------------------------------------------------------------
	# block series and the detection threshold
	# -----------------------------------------------------------------
	print('reducing the block series')
	blkC, blkT = {}, None
	for tag in RUNS:
		_, _, (bGtG, bGtY, blk, nblock, t) = raw[tag]
		blkC[tag], tb = block_series(bGtG, bGtY, blk, nblock, t, flat, ny, nx,
									 dHdx, dHdy, gband, aw)
		blkT = tb
	widx_b = window_index(blkT, WINDOWS)
	noise = np.full(nwin, np.nan)
	for k in range(nwin):
		v = blkC['ctrl'][widx_b == k]
		v = v[np.isfinite(v)]
		if v.size > 3:
			noise[k] = float(np.percentile(v, 75) - np.percentile(v, 25))

	# =================================================================
	# write the figure file
	# =================================================================
	print('writing', OUTFILE)
	coords = dict(
		window=('window', wins),
		band=('band', bnames),
		eta_rho=('eta_rho', np.arange(ny)),
		xi_rho=('xi_rho', np.arange(nx)),
		break_pt=('break_pt', np.arange(nb)),
		alpha_bin=('alpha_bin', acen),
		nb2_bin=('nb2_bin', ncen),
		block=('block', np.arange(blkT.size)),
	)
	dv = {k: (('window', 'eta_rho', 'xi_rho'), v) for k, v in F.items()}
	dv.update(dict(
		ln_total=(('window', 'eta_rho', 'xi_rho'), ln_tot),
		ln_amplitude=(('window', 'eta_rho', 'xi_rho'), ln_amp),
		ln_alignment=(('window', 'eta_rho', 'xi_rho'), ln_ali),
	))
	dv.update({k: (('window', 'band'), v) for k, v in integ.items()})
	dv.update({f'fac_{k}': (('window', 'band'), v) for k, v in fac.items()})
	dv.update({f'seas_{k}': (('window', 'band'), v) for k, v in seas.items()})
	dv.update({k: (('window', 'break_pt'), v) for k, v in brk.items()})
	dv.update({f'bing_{k}': (('window', 'alpha_bin'), v)
			   for k, v in bing.items()})
	dv.update({f'binh_{k}': (('window', 'nb2_bin'), v)
			   for k, v in binh.items()})
	dv.update(dict(
		lon_rho=(('eta_rho', 'xi_rho'), lon),
		lat_rho=(('eta_rho', 'xi_rho'), lat),
		h=(('eta_rho', 'xi_rho'), h),
		mask_rho=(('eta_rho', 'xi_rho'), mask),
		break_lon=('break_pt', blon),
		break_lat=('break_pt', blat),
		break_dist=('break_pt', bdist),
		C_block_ref=('block', blkC['ref']),
		C_block_ctrl=('block', blkC['ctrl']),
		block_time=('block', blkT),
		ctrl_block_iqr=('window', noise),
	))

	ds_out = xr.Dataset(dv, coords=coords)
	ds_out.attrs.update(dict(
		realistic=REALISTIC, control=CONTROL, eos_mode=EOS_MODE,
		block_hours=BLOCK_HOURS,
		constituents=','.join(CONSTITUENTS),
		break_isobath=BREAK_ISOBATH,
		gen_band_lo=GEN_BAND[0], gen_band_hi=GEN_BAND[1],
		bands=';'.join(f'{b[3]}:{b[0]}' for b in BANDS),
		windows=';'.join(f'{k}:{v[0]}..{v[1]}' for k, v in WINDOWS.items()),
		note='C = 0.5 P W cos(dphi) with P the amplitude of p_bc at the bed, '
			 'W the amplitude of w_bt = -u_bt . grad H, common to the two '
			 'runs, and cos(dphi) the alignment. ln_total = ln_amplitude + '
			 'ln_alignment exactly, cell by cell, wherever C and cos(dphi) '
			 'are positive in both runs. The fac_ fields are those terms '
			 'averaged over each band with C of the realistic run as weight.',
	))
	enc = {v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars}
	ds_out.to_netcdf(OUTFILE, encoding=enc)
	ds_out.close()

	# =================================================================
	# write the harmonics, so Figures 4 and 5 need no further pass
	# =================================================================
	if SAVE_HARMONICS:
		print('writing', HARMFILE)
		hv = {}
		for tag in RUNS:
			coef, zwm, _ = raw[tag]
			a3 = np.full((nwin, nz, ny, nx), np.nan, dtype=np.float32)
			b3 = np.full_like(a3, np.nan)
			au = np.full((nwin, nz, ny, nx), np.nan, dtype=np.float32)
			bu = np.full_like(au, np.nan)
			av = np.full_like(au, np.nan); bv = np.full_like(au, np.nan)
			zz = np.full((nwin, nz + 1, ny, nx), np.nan, dtype=np.float32)
			ab = np.full((nwin, ny, nx), np.nan, dtype=np.float32)
			bb = np.full_like(ab, np.nan)
			ab2 = np.full_like(ab, np.nan); bb2 = np.full_like(ab, np.nan)
			for k, wname in enumerate(wins):
				if wname not in coef:
					continue
				a, b = coef[wname]
				for lev in range(nz):
					au[k, lev] = unflat(a[lev]); bu[k, lev] = unflat(b[lev])
					av[k, lev] = unflat(a[nz+lev]); bv[k, lev] = unflat(b[nz+lev])
					a3[k, lev] = unflat(a[2*nz+lev])
					b3[k, lev] = unflat(b[2*nz+lev])
				for lev in range(nz + 1):
					zz[k, lev] = unflat(zwm[k, lev])
				ab[k] = unflat(a[3*nz]); bb[k] = unflat(b[3*nz])
				ab2[k] = unflat(a[3*nz+1]); bb2[k] = unflat(b[3*nz+1])
			hv.update({
				f'a_ubc_{tag}': (('window', 's_rho', 'eta_rho', 'xi_rho'), au),
				f'b_ubc_{tag}': (('window', 's_rho', 'eta_rho', 'xi_rho'), bu),
				f'a_vbc_{tag}': (('window', 's_rho', 'eta_rho', 'xi_rho'), av),
				f'b_vbc_{tag}': (('window', 's_rho', 'eta_rho', 'xi_rho'), bv),
				f'a_pbc_{tag}': (('window', 's_rho', 'eta_rho', 'xi_rho'), a3),
				f'b_pbc_{tag}': (('window', 's_rho', 'eta_rho', 'xi_rho'), b3),
				f'zw_mean_{tag}': (('window', 's_w', 'eta_rho', 'xi_rho'), zz),
				f'a_ubt_{tag}': (('window', 'eta_rho', 'xi_rho'), ab),
				f'b_ubt_{tag}': (('window', 'eta_rho', 'xi_rho'), bb),
				f'a_vbt_{tag}': (('window', 'eta_rho', 'xi_rho'), ab2),
				f'b_vbt_{tag}': (('window', 'eta_rho', 'xi_rho'), bb2),
			})
		hv.update(dict(
			lon_rho=(('eta_rho', 'xi_rho'), lon),
			lat_rho=(('eta_rho', 'xi_rho'), lat),
			h=(('eta_rho', 'xi_rho'), h),
			mask_rho=(('eta_rho', 'xi_rho'), mask),
			pm=(('eta_rho', 'xi_rho'), pm),
			pn=(('eta_rho', 'xi_rho'), pn),
			angle=(('eta_rho', 'xi_rho'),
				   np.asarray(dsr.angle) if 'angle' in dsr
				   else np.zeros_like(h)),
		))
		dh = xr.Dataset(hv, coords=dict(
			window=('window', wins),
			s_rho=('s_rho', np.arange(nz)),
			s_w=('s_w', np.arange(nz + 1)),
			eta_rho=('eta_rho', np.arange(ny)),
			xi_rho=('xi_rho', np.arange(nx)),
		))
		dh.attrs.update(dict(
			note='M2 cosine and sine coefficients of the baroclinic velocity '
				 'and pressure and of the depth averaged velocity, per '
				 'window and run, with the window mean z_w. Written by '
				 'fig03_compute.py so that Figures 4 and 5 need no further '
				 'pass over the model output.',
			constituents=','.join(CONSTITUENTS),
			windows=';'.join(f'{k}:{v[0]}..{v[1]}' for k, v in WINDOWS.items()),
		))
		ench = {v: {'zlib': True, 'complevel': 4} for v in dh.data_vars}
		dh.to_netcdf(HARMFILE, encoding=ench)
		dh.close()

	# sign check
	for tag in RUNS:
		s = np.nansum(np.where(wet & (h >= 250.) & (h <= 1000.),
							   F[f'C_{tag}'][0]*aw, 0.))
		if s < 0:
			warnings.warn(f'slope integrated conversion is negative in {tag}, '
						  'check the sign of grad H before using the numbers')
		else:
			print(f'  sign check {tag}: slope integrated C positive')

	import fig03_report
	fig03_report.main()


if __name__ == '__main__':
	main()