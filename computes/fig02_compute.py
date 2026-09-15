#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig02_compute.py
=================================================================
Figure 2  -  The medium the discharge builds
Amazon shelf internal tide manuscript, version 3

Computes and stores everything panels (a) to (g) need:

  (a to c) mode one phase speed c1 in the realistic run, three windows
  (d)      the difference in c1 between the runs at peak discharge
  (e)      the vertical structure of mode one at M1, M2 and M3
  (f)      the bottom value of the mode one structure function against
           the surface trapping index
  (g)      near bottom stratification along the 200 m isobath

Output   fig02_data.nc, and the report written by fig02_report.py

Why this figure carries no wave
-------------------------------
Every quantity here is a property of the background stratification and
of the vertical eigenproblem it defines. None of them requires the
model to resolve a propagating mode one wave, so all of them are valid
over the whole domain including the shelf, where Figure 1 shows the
wave itself is under resolved in the constant discharge run.

The quantity the paper needs from this figure is phi_1(-H), the value
of the mode one structure function at the bed under the normalisation
of Equation (phi). Each mode receives energy from the topography in
proportion to that value, so it is the channel through which a surface
trapped lens reaches the generation.

Two definitions changed from version 2
--------------------------------------
* The plume base hp is now the depth at which salinity first reaches
  35, following Lentz (1995) and Geyer et al. (1996). The depth of the
  maximum of N2 is retained alongside it as the pycnocline core, and
  the two differ by a factor of several over the shelf.
* The surface trapping index is measured above that plume base.
=================================================================
"""

import warnings
import numpy as np
import xarray as xr

from fig01_compute import (
	REALISTIC, CONTROL, GRIDFILE, WINDOWS, MOORINGS, BREAK_ISOBATH,
	NMODES, N2_FLOOR, CHUNK_COLS, C1_MIN, T_M2, G, RHO0,
	open_run, compute_N2, vertical_modes, near_bottom_N2,
	extract_isobath, build_tree, nearest_ji, plume_thickness,
)


# =====================================================================
# CONFIGURATION
# =====================================================================
OUTFILE = 'fig02_data.nc'

S_LENS = 35.0            # isohaline defining the base of the freshwater lens
H_MIN_STATS = 50.0       # m, shallowest depth entering the binned relations
H_MAX_STATS = 1000.0     # m, deepest depth entering the binned relations

T1_BINS = np.linspace(0.20, 1.00, 17)     # surface trapping index bins
N_SUB = 4000             # points kept per window and run for the scatter
SEED = 20260825

RUN_TAGS = ('ref', 'ctrl')

# depth bands used by the report
BANDS = [('mid shelf, 20 to 100 m',      20.,  100.),
		 ('outer shelf, 100 to 250 m',  100.,  250.),
		 ('upper slope, 250 to 1000 m', 250., 1000.)]


# =====================================================================
# HELPERS SPECIFIC TO THIS FIGURE
# =====================================================================
def phi_from_Phi(Phi_w, zw, c):
	"""
	Structure function for horizontal velocity and pressure,

		phi_n = rho0 c_n^2 dPhi_n/dz,     (1/H) int phi_n^2 dz = 1,

	evaluated at rho points from Phi at w points and normalised to unit
	depth mean square, which is the normalisation of Equation (phi) and
	the one that makes phi comparable between columns of different depth
	and between the two experiments.

	Phi_w  (Nw, M) structure function at w points, bottom first
	zw     (Nw, M) w point depths, negative, increasing to the surface
	c      (M,)    modal phase speed

	Returns phi (Nz, M) at rho points and the column depth H (M,).
	"""
	dz = np.diff(zw, axis=0)                       # (Nz, M)
	dPhi = np.diff(Phi_w, axis=0)
	with np.errstate(invalid='ignore', divide='ignore'):
		phi = RHO0*(c[None, :]**2)*(dPhi/dz)
	H = zw[-1] - zw[0]
	with np.errstate(invalid='ignore', divide='ignore'):
		msq = np.nansum(phi**2*dz, axis=0)/H
		norm = np.sqrt(np.where(msq > 0, msq, np.nan))
		phi = phi/norm[None, :]
	return phi, H


def bottom_value(phi):
	"""|phi| at the deepest rho point, the amplitude the topography forces."""
	return np.abs(phi[0])


def trapping_index(phi, zw, hp):
	"""
	Fraction of the mode one energy above the plume base,

		T1 = int_{-hp}^{0} phi^2 dz / int_{-H}^{0} phi^2 dz .

	phi (Nz, M) at rho points, zw (Nw, M) at w points, hp (M,) positive.
	"""
	dz = np.diff(zw, axis=0)
	zr = 0.5*(zw[1:] + zw[:-1])
	e = phi**2*dz
	tot = np.nansum(e, axis=0)
	above = np.nansum(np.where(zr > -hp[None, :], e, 0.0), axis=0)
	with np.errstate(invalid='ignore', divide='ignore'):
		return np.where(tot > 0, above/tot, np.nan)


def plume_base(salt, zr, thresh=S_LENS):
	"""
	Depth at which the salinity first reaches `thresh`, scanning down
	from the surface, by linear interpolation between the two levels
	that straddle it.

	salt, zr  (Nz, M), index 0 at the bed and index -1 at the surface

	Returns hp (M,) positive downward, together with two flags. Where
	the surface is already saltier than the threshold there is no lens
	and hp is not defined. Where no level reaches the threshold the lens
	fills the column and hp is the local depth.
	"""
	s = salt[::-1]
	z = zr[::-1]
	M = s.shape[1]
	cols = np.arange(M)

	reaches = s >= thresh
	any_r = reaches.any(axis=0)
	k = np.argmax(reaches, axis=0)

	s1 = s[k, cols]; z1 = z[k, cols]
	km1 = np.maximum(k - 1, 0)
	s0 = s[km1, cols]; z0 = z[km1, cols]
	with np.errstate(invalid='ignore', divide='ignore'):
		frac = np.where(np.abs(s1 - s0) > 1e-9, (thresh - s0)/(s1 - s0), 0.0)
	frac = np.clip(frac, 0.0, 1.0)
	hp = -(z0 + frac*(z1 - z0))

	no_lens = reaches[0]                       # surface already salty
	full_col = ~any_r                          # fresh all the way down
	hp = np.where(full_col, -z[-1], hp)        # lens fills the column
	hp = np.where(no_lens, np.nan, hp)
	return hp, no_lens.astype('int8'), full_col.astype('int8')


def binned(x, y, bins):
	"""Median and quartiles of y in bins of x."""
	ctr = 0.5*(bins[1:] + bins[:-1])
	med = np.full(ctr.size, np.nan)
	q25 = np.full(ctr.size, np.nan)
	q75 = np.full(ctr.size, np.nan)
	ok = np.isfinite(x) & np.isfinite(y)
	for k in range(ctr.size):
		sel = ok & (x >= bins[k]) & (x < bins[k + 1])
		if sel.sum() > 25:
			med[k] = np.nanmedian(y[sel])
			q25[k] = np.nanpercentile(y[sel], 25)
			q75[k] = np.nanpercentile(y[sel], 75)
	return ctr, med, q25, q75


def linfit(x, y):
	"""Least squares slope, intercept and correlation of y on x."""
	ok = np.isfinite(x) & np.isfinite(y)
	if ok.sum() < 50:
		return np.nan, np.nan, np.nan
	a, b = np.polyfit(x[ok], y[ok], 1)
	r = float(np.corrcoef(x[ok], y[ok])[0, 1])
	return float(a), float(b), r


# =====================================================================
# MAIN
# =====================================================================
def main():
	print('opening runs')
	dsr = open_run(REALISTIC)
	dsc = open_run(CONTROL)

	lon = np.asarray(dsr.lon_rho); lat = np.asarray(dsr.lat_rho)
	h = np.asarray(dsr.h)
	mask = np.asarray(dsr.mask_rho) if 'mask_rho' in dsr else np.ones_like(h)
	pm = np.asarray(dsr.pm)
	dx = 1.0/pm
	ny, nx = h.shape
	wet = mask > 0
	flat = wet.ravel()
	nwet = int(flat.sum())
	print(f'  grid {ny} x {nx}, {int(np.sum(wet))} wet columns')

	wins = list(WINDOWS.keys())
	nwin = len(wins)

	# -----------------------------------------------------------------
	# shelf break path and mooring columns, needed inside the loop
	# -----------------------------------------------------------------
	blon, blat, bdist = extract_isobath(lon, lat, np.where(wet, h, np.nan),
										BREAK_ISOBATH)
	tree, jj, ii = build_tree(lon, lat, mask)
	bj, bi = nearest_ji(tree, jj, ii, blon, blat)
	nb = blon.size

	st_names = list(MOORINGS.keys())
	st_j, st_i = [], []
	for nm, (mlo, mla) in MOORINGS.items():
		j, i = nearest_ji(tree, jj, ii, mlo, mla)
		st_j.append(int(j[0])); st_i.append(int(i[0]))
	nst = len(st_names)
	print('  moorings at depths ' +
		  ', '.join(f'{nm} {h[j, i]:.0f} m'
					for nm, j, i in zip(st_names, st_j, st_i)))

	# -----------------------------------------------------------------
	# allocation
	# -----------------------------------------------------------------
	keys2d = ['c1', 'c2', 'phib1', 'trap1', 'hp', 'zn2max', 'nb2', 'salt_surf']
	out = {f'{k}_{t}': np.full((nwin, ny, nx), np.nan)
		   for k in keys2d for t in RUN_TAGS}
	flag = {f'{k}_{t}': np.zeros((nwin, ny, nx), dtype='int8')
			for k in ('nolens', 'fulllens') for t in RUN_TAGS}

	prof = {}
	rng = np.random.default_rng(SEED)
	sub = {f'{k}_{t}': np.full((nwin, N_SUB), np.nan)
		   for k in ('trap1', 'phib1', 'h', 'c1') for t in RUN_TAGS}
	binstat = {f'{k}_{t}': np.full((nwin, T1_BINS.size - 1), np.nan)
			   for k in ('med', 'q25', 'q75') for t in RUN_TAGS}
	fitpar = {f'{k}_{t}': np.full(nwin, np.nan)
			  for k in ('slope', 'icept', 'r') for t in RUN_TAGS}

	# -----------------------------------------------------------------
	# window loop
	# -----------------------------------------------------------------
	for iw, wname in enumerate(wins):
		t0, t1 = WINDOWS[wname]
		for tag, ds in (('ref', dsr), ('ctrl', dsc)):
			sel = ds.sel(ocean_time=slice(t0, t1))
			if sel.sizes['ocean_time'] == 0:
				warnings.warn(f'{wname}/{tag}: no records, skipped')
				continue
			print(f'  {wname}/{tag}: {sel.sizes["ocean_time"]} records')

			temp = sel.temp.mean('ocean_time').values
			salt = sel.salt.mean('ocean_time').values
			zr = sel.z_rho.mean('ocean_time').values
			zw = sel.z_w.mean('ocean_time').values
			Nw, Nz = zw.shape[0], zr.shape[0]

			N2 = compute_N2(temp, salt, zr, zw, lon, lat)

			N2f = N2.reshape(Nw, -1)[:, flat]
			zwf = zw.reshape(Nw, -1)[:, flat]
			zrf = zr.reshape(Nz, -1)[:, flat]
			saltf = salt.reshape(Nz, -1)[:, flat]

			c, Phi = vertical_modes(N2f, zwf, nmodes=NMODES,
									free_surface=True)
			c1f, c2f = c[1], c[2]

			phi1f, _ = phi_from_Phi(Phi[:, 1, :], zwf, c1f)
			phib1f = bottom_value(phi1f)

			hpf, nolens, fullens = plume_base(saltf, zrf)
			trap1f = trapping_index(phi1f, zwf, np.nan_to_num(hpf, nan=0.0))
			trap1f = np.where(np.isfinite(hpf), trap1f, np.nan)

			nb2f = near_bottom_N2(N2f, zwf)

			def scatter(v):
				a = np.full(ny*nx, np.nan)
				a[flat] = v
				return a.reshape(ny, nx)

			out[f'c1_{tag}'][iw] = scatter(c1f)
			out[f'c2_{tag}'][iw] = scatter(c2f)
			out[f'phib1_{tag}'][iw] = scatter(phib1f)
			out[f'trap1_{tag}'][iw] = scatter(trap1f)
			out[f'hp_{tag}'][iw] = scatter(hpf)
			out[f'nb2_{tag}'][iw] = scatter(nb2f)
			out[f'salt_surf_{tag}'][iw] = np.where(wet, salt[-1], np.nan)
			out[f'zn2max_{tag}'][iw] = np.where(wet, plume_thickness(N2, zw),
												np.nan)
			flag[f'nolens_{tag}'][iw] = scatter(nolens.astype(float)) > 0.5
			flag[f'fulllens_{tag}'][iw] = scatter(fullens.astype(float)) > 0.5

			# ---- mooring profiles
			for n, (j, i) in enumerate(zip(st_j, st_i)):
				key = f'{tag}_{n}'
				col = j*nx + i
				m = int(np.sum(flat[:col]))          # index within the wet set
				if not flat[col]:
					continue
				phi_col = phi1f[:, m]
				prof.setdefault(f'N2_{tag}', np.full((nst, nwin, Nw), np.nan))
				prof.setdefault(f'Phi1_{tag}', np.full((nst, nwin, Nw), np.nan))
				prof.setdefault(f'phi1_{tag}', np.full((nst, nwin, Nz), np.nan))
				prof.setdefault(f'zw_{tag}', np.full((nst, nwin, Nw), np.nan))
				prof.setdefault(f'zr_{tag}', np.full((nst, nwin, Nz), np.nan))
				prof.setdefault(f'c1_{tag}', np.full((nst, nwin), np.nan))
				prof.setdefault(f'c2_{tag}', np.full((nst, nwin), np.nan))
				prof.setdefault(f'hp_{tag}', np.full((nst, nwin), np.nan))
				prof.setdefault(f'trap1_{tag}', np.full((nst, nwin), np.nan))
				prof[f'N2_{tag}'][n, iw] = N2f[:, m]
				prof[f'Phi1_{tag}'][n, iw] = Phi[:, 1, m]
				prof[f'phi1_{tag}'][n, iw] = phi_col
				prof[f'zw_{tag}'][n, iw] = zwf[:, m]
				prof[f'zr_{tag}'][n, iw] = zrf[:, m]
				prof[f'c1_{tag}'][n, iw] = c1f[m]
				prof[f'c2_{tag}'][n, iw] = c2f[m]
				prof[f'hp_{tag}'][n, iw] = hpf[m]
				prof[f'trap1_{tag}'][n, iw] = trap1f[m]

			# ---- binned relation and subsample for panel (f)
			hh = h[wet]
			keep = (hh >= H_MIN_STATS) & (hh <= H_MAX_STATS) \
				   & np.isfinite(trap1f) & np.isfinite(phib1f) \
				   & (c1f >= C1_MIN)
			x, y = trap1f[keep], phib1f[keep]
			ctr, med, q25, q75 = binned(x, y, T1_BINS)
			binstat[f'med_{tag}'][iw] = med
			binstat[f'q25_{tag}'][iw] = q25
			binstat[f'q75_{tag}'][iw] = q75
			a, b, r = linfit(x, y)
			fitpar[f'slope_{tag}'][iw] = a
			fitpar[f'icept_{tag}'][iw] = b
			fitpar[f'r_{tag}'][iw] = r
			if x.size:
				pick = rng.choice(x.size, size=min(N_SUB, x.size),
								  replace=False)
				sub[f'trap1_{tag}'][iw, :pick.size] = x[pick]
				sub[f'phib1_{tag}'][iw, :pick.size] = y[pick]
				sub[f'h_{tag}'][iw, :pick.size] = hh[keep][pick]
				sub[f'c1_{tag}'][iw, :pick.size] = c1f[keep][pick]
			print(f'    phi1(-H) on T1, slope {a:+.3f}, r {r:+.2f}, '
				  f'{x.size} columns')

	# -----------------------------------------------------------------
	# along break profiles
	# -----------------------------------------------------------------
	brk = {}
	for k in ('c1', 'nb2', 'phib1', 'trap1', 'hp'):
		for t in RUN_TAGS:
			brk[f'{k}_{t}_break'] = out[f'{k}_{t}'][:, bj, bi]

	# =================================================================
	# write
	# =================================================================
	print('writing', OUTFILE)
	Nw = prof['zw_ref'].shape[-1]
	Nz = prof['zr_ref'].shape[-1]

	coords = dict(
		window=('window', wins),
		eta_rho=('eta_rho', np.arange(ny)),
		xi_rho=('xi_rho', np.arange(nx)),
		break_pt=('break_pt', np.arange(nb)),
		station=('station', st_names),
		s_w=('s_w', np.arange(Nw)),
		s_rho=('s_rho', np.arange(Nz)),
		t1_bin=('t1_bin', 0.5*(T1_BINS[1:] + T1_BINS[:-1])),
		sub=('sub', np.arange(N_SUB)),
	)

	dv = {}
	for k, v in out.items():
		dv[k] = (('window', 'eta_rho', 'xi_rho'), v)
	for k, v in flag.items():
		dv[k] = (('window', 'eta_rho', 'xi_rho'), v.astype('int8'))
	for k, v in brk.items():
		dv[k] = (('window', 'break_pt'), v)
	for k, v in binstat.items():
		dv[f'bin_{k}'] = (('window', 't1_bin'), v)
	for k, v in fitpar.items():
		dv[f'fit_{k}'] = ('window', v)
	for k, v in sub.items():
		dv[f'sub_{k}'] = (('window', 'sub'), v)
	for k, v in prof.items():
		dim = 's_w' if v.ndim == 3 and v.shape[-1] == Nw else 's_rho'
		if v.ndim == 3:
			dv[f'prof_{k}'] = (('station', 'window', dim), v)
		else:
			dv[f'prof_{k}'] = (('station', 'window'), v)

	dv.update(dict(
		lon_rho=(('eta_rho', 'xi_rho'), lon),
		lat_rho=(('eta_rho', 'xi_rho'), lat),
		h=(('eta_rho', 'xi_rho'), h),
		mask_rho=(('eta_rho', 'xi_rho'), mask),
		dx=(('eta_rho', 'xi_rho'), dx),
		break_lon=('break_pt', blon),
		break_lat=('break_pt', blat),
		break_dist=('break_pt', bdist),
		break_h=('break_pt', h[bj, bi]),
		station_lon=('station', np.array([lon[j, i]
										  for j, i in zip(st_j, st_i)])),
		station_lat=('station', np.array([lat[j, i]
										  for j, i in zip(st_j, st_i)])),
		station_h=('station', np.array([h[j, i]
										for j, i in zip(st_j, st_i)])),
	))

	ds_out = xr.Dataset(dv, coords=coords)
	ds_out.attrs.update(dict(
		realistic=REALISTIC, control=CONTROL,
		break_isobath=BREAK_ISOBATH,
		s_lens=S_LENS, c1_min=C1_MIN,
		h_min_stats=H_MIN_STATS, h_max_stats=H_MAX_STATS,
		windows=';'.join(f'{k}:{v[0]}..{v[1]}' for k, v in WINDOWS.items()),
		moorings=';'.join(f'{k}:{v[0]},{v[1]}' for k, v in MOORINGS.items()),
		note='phi_1 is normalised to unit depth mean square, so phib1 is the '
			 'dimensionless bottom amplitude through which the topography '
			 'forces mode one; hp is the depth at which salinity first '
			 'reaches the s_lens attribute, and zn2max is the depth of the '
			 'maximum of N2, retained for comparison; nothing in this file '
			 'requires the mode one wave to be resolved',
	))

	enc = {v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars}
	ds_out.to_netcdf(OUTFILE, encoding=enc)
	ds_out.close()

	# -----------------------------------------------------------------
	# report, written straight after the data so the two never diverge
	# -----------------------------------------------------------------
	import fig02_report
	fig02_report.main()


if __name__ == '__main__':
	main()