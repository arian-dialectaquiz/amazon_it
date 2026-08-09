#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig04_compute.py
=================================================================
Figure 4  -  The shelf break gate and the frontal scatterer
Amazon shelf internal tide manuscript

Computes and stores everything panels (a) to (d) need:

  (a) cross shelf sections of the mode 1, 2 and 3 baroclinic flux at peak
	  discharge, REF against CTRL
  (b) modal partition of the flux crossing the 200 m isobath in time,
	  separated into its onshore and offshore parts
  (c) the onshore fraction against the modal impedance contrast Gamma,
	  coloured by plume thickness, with the plane wave expectation
  (d) map of the mode 1 to higher mode scattering rate, with mode 1 rays
	  traced through the c1 field and their turning points marked

Output:  fig04_data.nc   (read by fig04_plot.py)

Method
------
Two passes per run. A cheap strided pass gives the window mean
stratification, from which the vertical modes are solved on the full
grid and normalised so that (1/H) int phi_n^2 dz = 1. A full streaming
pass then projects the baroclinic velocity and pressure onto those modes
AT EACH TIME STEP, and accumulates the harmonic normal equations for the
modal amplitudes alone. That is 9 scalars per point, u_n, v_n and p_n for
n = 1 to 3, instead of the 92 that fig03 carried, so the accumulator is a
few tens of megabytes. The modal flux then follows from

	F_n = H <u_n p_n>,   <x y> = 0.5 (a_x a_y + b_x b_y)

WHY THE DIRECTIONS ARE SEPARATED
--------------------------------
The generation sits on the shelf break itself, so there is no well
defined incident wave and the textbook transmission and reflection
coefficients cannot be measured. Instead the modal flux normal to the
break is split into its onshore and offshore parts before integration,

	P_on  = -int min(F.n_off, 0) dl,   P_off = int max(F.n_off, 0) dl

and the onshore fraction P_on/(P_on + P_off) is used as the
transmission-like quantity. This distinction matters, because fig03
showed the NET flux across the break is seaward and becomes more so with
the plume, which is compatible with the onshore part rising as well if
the source strengthened. Only the directional split can tell the two
apart.
=================================================================
"""

import os
import time as _time
import warnings
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fig01_compute import (
	REALISTIC, CONTROL, WINDOWS, SECTIONS, GEN_SITES, GEN_SITES_MAIN,
	BREAK_ISOBATH, SHELF_MAX_DEPTH, G, RHO0, OMEGA_M2, OMEGA_S2,
	open_run, compute_N2, vertical_modes, extract_isobath,
	build_tree, nearest_ji, C1_MIN, roms_density, eos_provenance,
)
from fig03_compute import (
	uv_to_rho, baroclinic_pressure, design_matrix, solve_normal,
	cycle_mean, window_index, density_chunk, window_mean_density,
	grad_h_components, EOS_MODE, MEAN_STRIDE, BLOCK_HOURS, TIME_CHUNK,
)


# =====================================================================
# CONFIGURATION
# =====================================================================
OUTFILE = 'fig04_data.nc'

NMODE = 3                      # modes retained, 1 to NMODE
CONSTITUENTS = {'M2': OMEGA_M2, 'S2': OMEGA_S2}
BLOCK_CONSTITUENTS = {'M2': OMEGA_M2}

GAMMA_DIST = 25.0e3            # m, offset each side of the break for Gamma
SECTION_NPT = 260              # points along each cross shelf section

# ray tracing through the c1 field
RAY_SITES = ['A', 'C', 'D', 'E']
RAY_FAN = np.deg2rad(np.arange(-70., 71., 10.))   # about the shoreward normal
RAY_DS = 3.0e3                 # m, integration step
RAY_NSTEP = 220                # steps, about 660 km of path
C_REF = 1.0                    # m s-1, reference for the refractive index

RUNS = ('ref', 'ctrl')
PLOT_WINDOW = 'peak'

# Checkpoint each run's streaming pass to fig04_cache_<tag>.npz and
# reuse it if present. The pass costs over an hour per run, so this
# makes any later failure cheap. Delete the caches to force a redo.
USE_CACHE = True


# =====================================================================
# HELPERS
# =====================================================================
def normalised_modes(N2, zw, nmode=NMODE):
	"""
	Vertical modes with phi_n normalised so that (1/H) int phi_n^2 dz = 1,
	which makes F_n = H <u_n p_n> and makes the modal amplitudes directly
	comparable between columns.

	Returns c (nmode, M), phi (nz, nmode, M) on rho points, dz, H.
	"""
	nz = zw.shape[0] - 1
	c, Phi = vertical_modes(N2, zw, nmodes=nmode + 1, free_surface=True)
	dz = np.diff(zw, axis=0)
	H = np.sum(dz, axis=0)

	phi = np.full((nz, nmode, zw.shape[1]), np.nan)
	cn = np.full((nmode, zw.shape[1]), np.nan)
	for n in range(1, nmode + 1):                 # mode 0 is barotropic
		dPhi = np.diff(Phi[:, n, :], axis=0)
		with np.errstate(divide='ignore', invalid='ignore'):
			p = RHO0*(c[n][None, :]**2)*dPhi/dz
		norm = np.sqrt(np.maximum(np.sum(p*p*dz, axis=0)/np.maximum(H, 1e-6),
								  1e-30))
		phi[:, n - 1, :] = p/norm[None, :]
		cn[n - 1] = c[n]
	return cn, phi, dz, H


def unit_normal_offshore(h, pm, pn, mask):
	"""Unit vector pointing offshore, from the bathymetric gradient."""
	dHdx, dHdy = grad_h_components(h, pm, pn)
	g = np.hypot(dHdx, dHdy)
	with np.errstate(divide='ignore', invalid='ignore'):
		nx = np.where(g > 0, dHdx/g, 0.)
		ny = np.where(g > 0, dHdy/g, 0.)
	return np.where(mask > 0, nx, np.nan), np.where(mask > 0, ny, np.nan)


def sample_offset(tree, jj, ii, blon, blat, nx, ny, dist_m):
	"""
	Grid indices a fixed distance along the local offshore normal.
	dist_m positive is offshore, negative is onshore.
	"""
	dlon = (nx*dist_m/111.2e3)/np.cos(np.deg2rad(blat))
	dlat = ny*dist_m/111.2e3
	return nearest_ji(tree, jj, ii, blon + dlon, blat + dlat)


def trace_rays(c1, lon, lat, mask, x0, y0, angles, ds=RAY_DS,
			   nstep=RAY_NSTEP, c_ref=C_REF):
	"""
	Mode one rays through the horizontal c1 field, using the eikonal
	equation d/ds (n t) = grad n with the refractive index n = c_ref/c1.

	Refraction is towards slower water, so a ray entering the shelf from
	the break bends towards the cross shelf direction. A ray whose cross
	shelf progress reverses has turned and is recorded as such.

	Returns paths (nray, nstep, 2) in lon, lat with NaN after the ray
	leaves the domain, and a turning flag per ray.
	"""
	ok = np.isfinite(c1) & (mask > 0) & (c1 > C1_MIN)
	jj, ii = np.where(ok)
	tree = cKDTree(np.column_stack([lon[ok], lat[ok]]))
	cv = c1[ok]

	# refractive index on the grid, and its gradient in metres
	n_grid = np.where(ok, c_ref/np.maximum(c1, C1_MIN), np.nan)
	coslat = np.cos(np.deg2rad(lat))
	dx = np.gradient(lon, axis=1)*111.2e3*coslat
	dy = np.gradient(lat, axis=0)*111.2e3
	with np.errstate(invalid='ignore', divide='ignore'):
		gnx = np.gradient(np.nan_to_num(n_grid), axis=1)/np.where(np.abs(dx) > 1, dx, np.nan)
		gny = np.gradient(np.nan_to_num(n_grid), axis=0)/np.where(np.abs(dy) > 1, dy, np.nan)
	gnx = np.nan_to_num(gnx); gny = np.nan_to_num(gny)

	def at(plon, plat):
		"""Refractive index, its gradient and the distance to the nearest
		wet point, all as plain floats."""
		d, k = tree.query(np.array([[float(plon), float(plat)]]))
		j, i = int(jj[k[0]]), int(ii[k[0]])
		return (float(n_grid[j, i]), float(gnx[j, i]), float(gny[j, i]),
				float(d[0]))

	nray = len(angles)
	path = np.full((nray, nstep, 2), np.nan)
	turned = np.zeros(nray, dtype=bool)

	for r, a in enumerate(angles):
		plon, plat = float(x0), float(y0)
		n0, _, _, _ = at(plon, plat)
		if not np.isfinite(n0):
			continue
		tx, ty = float(np.cos(a)), float(np.sin(a))
		px, py = n0*tx, n0*ty
		prev_proj = None
		for s in range(nstep):
			path[r, s] = (plon, plat)
			nl, gx, gy, dist = at(plon, plat)
			if not np.isfinite(nl) or dist > 0.6:
				path[r, s:] = np.nan
				break
			px += gx*ds
			py += gy*ds
			m = np.hypot(px, py)
			if m <= 0 or not np.isfinite(m):
				path[r, s:] = np.nan
				break
			px, py = px*nl/m, py*nl/m
			tx, ty = px/nl, py/nl
			proj = tx*float(np.cos(a)) + ty*float(np.sin(a))
			if prev_proj is not None and proj < 0 <= prev_proj:
				turned[r] = True
			prev_proj = proj
			plat_new = plat + ty*ds/111.2e3
			plon = plon + (tx*ds/111.2e3)/max(np.cos(np.deg2rad(plat)), 0.1)
			plat = plat_new
	return path, turned


# =====================================================================
# STREAMING PASS FOR ONE RUN
# =====================================================================
def process_run(ds, tag, wins, geom, brk):
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

	print('  mean state pass and vertical modes')
	lon_f = lon.ravel()[flat]; lat_f = lat.ravel()[flat]
	Tm, Sm, rho_m, alpha, beta = window_mean_density(ds, wins, widx, flat)

	phi_w = np.full((nwin, nz, NMODE, nwet), np.nan, dtype=np.float32)
	cn_w = np.full((nwin, NMODE, nwet), np.nan)
	proj_w = np.full((nwin, nz, NMODE, nwet), np.nan, dtype=np.float32)
	Hm_w = np.full((nwin, nwet), np.nan)

	for k, wname in enumerate(wins):
		t0, t1 = WINDOWS[wname]
		sub = ds.sel(ocean_time=slice(t0, t1))
		if sub.sizes['ocean_time'] == 0:
			continue
		temp = sub.temp.mean('ocean_time').values
		salt = sub.salt.mean('ocean_time').values
		zr = sub.z_rho.mean('ocean_time').values
		zw = sub.z_w.mean('ocean_time').values
		N2 = compute_N2(temp, salt, zr, zw, lon, lat)
		N2f = N2.reshape(nz + 1, -1)[:, flat]
		zwf = zw.reshape(nz + 1, -1)[:, flat]
		cn, phi, dz, H = normalised_modes(N2f, zwf, NMODE)
		cn_w[k] = cn
		phi_w[k] = phi.astype(np.float32)
		Hm_w[k] = H
		# weights that turn a profile into a modal amplitude
		proj_w[k] = (phi*dz[:, None, :]/np.maximum(H, 1e-6)[None, None, :]
					 ).astype(np.float32)
		print(f'    {wname}: c1 median {np.nanmedian(cn[0]):.2f}, '
			  f'c2 {np.nanmedian(cn[1]):.2f}, c3 {np.nanmedian(cn[2]):.2f} m/s')

	# accumulators, 9 modal amplitudes per point instead of 92 profiles
	ncoef = 1 + 2*len(CONSTITUENTS)
	nfield = 3*NMODE
	GtG = np.zeros((nwin, ncoef, ncoef))
	GtY = np.zeros((nwin, ncoef, nfield, nwet))
	print(f'  window accumulator {GtY.nbytes/1e6:.0f} MB')

	nbc = 1 + 2*len(BLOCK_CONSTITUENTS)
	blk = max(int(round(BLOCK_HOURS/dt_h)), 8)
	chunk = TIME_CHUNK or blk
	nblock = int(np.ceil(t.size/blk))
	bj, bi, bwet = brk['j'], brk['i'], brk['wet']
	nb = bj.size
	bGtG = np.zeros((nblock, nbc, nbc))
	bGtY = np.zeros((nblock, nbc, nfield, nb), dtype=np.float32)
	tblk = np.empty(nblock, dtype=t.dtype)
	for g in range(nblock):
		tblk[g] = t[min(g*blk, t.size - 1)]

	tic = _time.time()
	nan_total = 0
	for s in range(0, t.size, chunk):
		e = min(s + chunk, t.size)
		sl = slice(s, e); nt_ = e - s
		kk = np.where(widx[sl] >= 0, widx[sl], 0)

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

		rho = density_chunk(ds, sl, T, S, zr, lon=lon_f, lat=lat_f,
							Tm=Tm, Sm=Sm, rho_m=rho_m,
							alpha=alpha, beta=beta, k=kk)
		p_bc = baroclinic_pressure(rho - rho_m[kk], zw)
		del T, S, zr, rho

		dz = np.diff(zw, axis=1)
		H = np.sum(dz, axis=1)
		u_bt = np.sum(ur*dz, axis=1)/np.maximum(H, 1e-6)
		v_bt = np.sum(vr*dz, axis=1)/np.maximum(H, 1e-6)
		u_bc = ur - u_bt[:, None, :]
		v_bc = vr - v_bt[:, None, :]
		del ur, vr, u_bt, v_bt

		# project onto the modes of the window each record belongs to
		Y = np.empty((nt_, nfield, nwet), dtype=np.float32)
		for k in np.unique(kk):
			m = kk == k
			P = proj_w[k]                       # (nz, NMODE, nwet)
			Y[m, 0:NMODE] = np.einsum('tzm,znm->tnm', u_bc[m], P,
									  optimize=True)
			Y[m, NMODE:2*NMODE] = np.einsum('tzm,znm->tnm', v_bc[m], P,
											optimize=True)
			Y[m, 2*NMODE:3*NMODE] = np.einsum('tzm,znm->tnm', p_bc[m], P,
											  optimize=True)
		del u_bc, v_bc, p_bc

		nbad = int(np.count_nonzero(~np.isfinite(Y)))
		if nbad:
			nan_total += nbad
			np.nan_to_num(Y, copy=False)

		X = design_matrix(t_s[sl], CONSTITUENTS)
		for k in range(nwin):
			m = widx[sl] == k
			if not m.any():
				continue
			Xk = X[m]
			GtG[k] += Xk.T @ Xk
			GtY[k] += np.einsum('tc,tfm->cfm', Xk, Y[m], optimize=True)

		Xb = design_matrix(t_s[sl], BLOCK_CONSTITUENTS)
		Yb = Y[:, :, bwet]
		bid = np.arange(s, e)//blk
		for g in np.unique(bid):
			m = bid == g
			Xg = Xb[m]
			bGtG[g] += Xg.T @ Xg
			bGtY[g] += np.einsum('tc,tfm->cfm', Xg, Yb[m],
								 optimize=True).astype(np.float32)
		del Y, Yb

		if (s//chunk) % 10 == 0 or e == t.size:
			done = e/t.size; el = _time.time() - tic
			print(f'    {100*done:5.1f}%   {el/60:6.1f} min elapsed   '
				  f'{el/max(done,1e-6)-el:6.0f} s to go', flush=True)

	if nan_total:
		frac = nan_total/(t.size*nfield*nwet)
		print(f'  WARNING {nan_total} non finite values in the projected '
			  f'fields ({100*frac:.4f}% of samples), zeroed before '
			  f'accumulation. These come from non finite salinity reaching '
			  f'the equation of state.')

	# ---- solve and form the modal fluxes
	print('  solving')
	out = {}
	for k, wname in enumerate(wins):
		if GtG[k, 0, 0] < ncoef:
			continue
		c = solve_normal(GtG[k], GtY[k])
		a, b = c[1], c[2]
		Fxi = np.full((NMODE, nwet), np.nan)
		Feta = np.full((NMODE, nwet), np.nan)
		for n in range(NMODE):
			Fxi[n] = Hm_w[k]*cycle_mean(a[n], b[n], a[2*NMODE + n],
										b[2*NMODE + n])
			Feta[n] = Hm_w[k]*cycle_mean(a[NMODE + n], b[NMODE + n],
										 a[2*NMODE + n], b[2*NMODE + n])
		out[wname] = dict(Fxi=Fxi, Feta=Feta, c=cn_w[k])

	# ---- block resolved modal flux along the break
	Fb = np.full((nblock, NMODE, 2, nb), np.nan)
	for g in range(nblock):
		if bGtG[g, 0, 0] <= nbc:
			continue
		c = solve_normal(bGtG[g], bGtY[g].astype(np.float64))
		a, b = c[1], c[2]
		Hb = np.nanmean(Hm_w[:, bwet], axis=0)
		for n in range(NMODE):
			Fb[g, n, 0] = Hb*cycle_mean(a[n], b[n], a[2*NMODE + n],
										b[2*NMODE + n])
			Fb[g, n, 1] = Hb*cycle_mean(a[NMODE + n], b[NMODE + n],
										a[2*NMODE + n], b[2*NMODE + n])
	return out, Fb, tblk, cn_w, Hm_w


# =====================================================================
# GEOMETRY, STREAMING, CACHE
# =====================================================================
def build_geometry():
	"""Grid, shelf break path and its offshore normal. Cheap, no reads."""
	dsr = open_run(REALISTIC, chunks={'ocean_time': BLOCK_HOURS})
	dsc = open_run(CONTROL, chunks={'ocean_time': BLOCK_HOURS})

	lon = np.asarray(dsr.lon_rho); lat = np.asarray(dsr.lat_rho)
	h = np.asarray(dsr.h)
	mask = np.asarray(dsr.mask_rho) if 'mask_rho' in dsr else np.ones_like(h)
	pm = np.asarray(dsr.pm); pn = np.asarray(dsr.pn)
	ang = np.asarray(dsr.angle) if 'angle' in dsr else np.zeros_like(h)
	ny, nx = h.shape
	flat = mask.ravel() > 0
	geom = dict(lon=lon, lat=lat, h=h, mask=mask, pm=pm, pn=pn, angle=ang,
				flat=flat, ny=ny, nx=nx)

	blon, blat, bdist = extract_isobath(lon, lat,
										np.where(mask > 0, h, np.nan),
										BREAK_ISOBATH)
	tree, jj, ii = build_tree(lon, lat, mask)
	bj, bi = nearest_ji(tree, jj, ii, blon, blat)
	colidx = np.full(ny*nx, -1, dtype=int)
	colidx[flat] = np.arange(int(flat.sum()))
	colidx = colidx.reshape(ny, nx)
	bwet = colidx[bj, bi]
	good = bwet >= 0
	blon, blat, bdist = blon[good], blat[good], bdist[good]
	bj, bi, bwet = bj[good], bi[good], bwet[good]
	nox, noy = unit_normal_offshore(h, pm, pn, mask)
	brk = dict(j=bj, i=bi, wet=bwet, lon=blon, lat=blat, dist=bdist,
			   dl=np.gradient(bdist)*1000., nox=nox, noy=noy,
			   tree=tree, jj=jj, ii=ii, nb=blon.size)
	print(f'  shelf break path, {blon.size} points, {bdist[-1]:.0f} km long')
	return dsr, dsc, geom, brk


def _cache_path(tag):
	return f'fig04_cache_{tag}.npz'


def save_cache(tag, r):
	"""
	Checkpoint one run's streaming result. The pass costs over an hour, so
	it is written to disk the moment it finishes and any later failure in
	assembly or writing costs seconds rather than the whole pass.
	"""
	out, Fb, tblk, cn_w, Hm_w = r
	d = dict(Fb=Fb, tblk=tblk.astype('datetime64[s]').astype('int64'),
			 cn_w=cn_w, Hm_w=Hm_w, wins=np.array(list(out.keys())))
	for w, v in out.items():
		d[f'Fxi__{w}'] = v['Fxi']; d[f'Feta__{w}'] = v['Feta']
		d[f'c__{w}'] = v['c']
	np.savez_compressed(_cache_path(tag), **d)
	print(f'  cached {tag} to {_cache_path(tag)} '
		  f'({os.path.getsize(_cache_path(tag))/1e6:.0f} MB)')


def load_cache(tag):
	p = _cache_path(tag)
	if not os.path.exists(p):
		return None
	z = np.load(p, allow_pickle=False)
	out = {}
	for w in [str(x) for x in z['wins']]:
		out[w] = dict(Fxi=z[f'Fxi__{w}'], Feta=z[f'Feta__{w}'],
					  c=z[f'c__{w}'])
	tblk = z['tblk'].astype('datetime64[s]')
	print(f'  loaded {tag} from {p}, streaming pass skipped')
	return out, z['Fb'], tblk, z['cn_w'], z['Hm_w']


# =====================================================================
# ASSEMBLY, everything after the streaming passes
# =====================================================================
def assemble(res, geom, brk, wins, outfile=OUTFILE, do_rays=True):
	lon, lat, h, mask = geom['lon'], geom['lat'], geom['h'], geom['mask']
	ang, flat, ny, nx = geom['angle'], geom['flat'], geom['ny'], geom['nx']
	bj, bi, bwet = brk['j'], brk['i'], brk['wet']
	blon, blat, bdist = brk['lon'], brk['lat'], brk['dist']
	dl, nox, noy, nb = brk['dl'], brk['nox'], brk['noy'], brk['nb']
	tree, jj, ii = brk['tree'], brk['jj'], brk['ii']
	nwin = len(wins)

	print('assembling')
	NM = NMODE
	Fx = {t: np.full((nwin, NM, ny, nx), np.nan) for t in RUNS}
	Fy = {t: np.full((nwin, NM, ny, nx), np.nan) for t in RUNS}
	Cn = {t: np.full((nwin, NM, ny, nx), np.nan) for t in RUNS}

	def unflat(v):
		a = np.full(ny*nx, np.nan); a[flat] = v
		return a.reshape(ny, nx)

	for tag in RUNS:
		out, Fb, tblk, cn_w, Hm_w = res[tag]
		for k, w in enumerate(wins):
			if w not in out:
				continue
			for n in range(NM):
				Fx[tag][k, n] = unflat(out[w]['Fxi'][n])
				Fy[tag][k, n] = unflat(out[w]['Feta'][n])
				Cn[tag][k, n] = unflat(out[w]['c'][n])

	# ---- directional split of the break crossing flux, per mode
	P_on = {t: np.full((nwin, NM), np.nan) for t in RUNS}
	P_off = {t: np.full((nwin, NM), np.nan) for t in RUNS}
	Fperp = {t: np.full((nwin, NM, nb), np.nan) for t in RUNS}
	for tag in RUNS:
		for k in range(nwin):
			for n in range(NM):
				fp = (Fx[tag][k, n][bj, bi]*nox[bj, bi]
					  + Fy[tag][k, n][bj, bi]*noy[bj, bi])
				Fperp[tag][k, n] = fp
				P_off[tag][k, n] = np.nansum(np.maximum(fp, 0)*dl)/1e6
				P_on[tag][k, n] = -np.nansum(np.minimum(fp, 0)*dl)/1e6

	# ---- block resolved directional split
	nblock = res[RUNS[0]][1].shape[0]
	Pon_b = {t: np.full((nblock, NM), np.nan) for t in RUNS}
	Poff_b = {t: np.full((nblock, NM), np.nan) for t in RUNS}
	for tag in RUNS:
		Fb = res[tag][1]
		for g in range(nblock):
			for n in range(NM):
				fp = Fb[g, n, 0]*nox[bj, bi] + Fb[g, n, 1]*noy[bj, bi]
				if not np.isfinite(fp).any():
					continue
				Poff_b[tag][g, n] = np.nansum(np.maximum(fp, 0)*dl)/1e6
				Pon_b[tag][g, n] = -np.nansum(np.minimum(fp, 0)*dl)/1e6

	# ---- impedance contrast across the break
	js, is_ = sample_offset(tree, jj, ii, blon, blat, nox[bj, bi],
							noy[bj, bi], -GAMMA_DIST)
	jd, id_ = sample_offset(tree, jj, ii, blon, blat, nox[bj, bi],
							noy[bj, bi], +GAMMA_DIST)
	Gam = {t: np.full((nwin, nb), np.nan) for t in RUNS}
	c1s = {t: np.full((nwin, nb), np.nan) for t in RUNS}
	c1d = {t: np.full((nwin, nb), np.nan) for t in RUNS}
	for tag in RUNS:
		for k in range(nwin):
			cs = Cn[tag][k, 0][js, is_]
			cd = Cn[tag][k, 0][jd, id_]
			c1s[tag][k] = cs; c1d[tag][k] = cd
			with np.errstate(invalid='ignore', divide='ignore'):
				Gam[tag][k] = (cd - cs)/(cd + cs)

	# ---- local onshore fraction along the break, for the regression
	phi_on = {t: np.full((nwin, nb), np.nan) for t in RUNS}
	for tag in RUNS:
		for k in range(nwin):
			f1 = Fperp[tag][k, 0]
			tot = np.abs(f1)
			with np.errstate(invalid='ignore', divide='ignore'):
				phi_on[tag][k] = np.where(tot > 0,
										  np.maximum(-f1, 0)/tot, np.nan)

	# ---- plume thickness along the break, from fig02 if available
	hp_break = np.full((nwin, nb), np.nan)
	if os.path.exists('fig02_data.nc'):
		d2 = xr.open_dataset('fig02_data.nc')
		for k in range(nwin):
			hp_break[k] = d2.hp_ref.isel(window=k).values[bj, bi]
		d2.close()
	else:
		warnings.warn('fig02_data.nc not found, plume thickness left empty')

	# ---- scattering rate, mode 1 into modes 2 and 3
	scat = {t: np.full((nwin, ny, nx), np.nan) for t in RUNS}
	coslat = np.cos(np.deg2rad(lat))
	dxm = np.gradient(lon, axis=1)*111.2e3*coslat
	dym = np.gradient(lat, axis=0)*111.2e3
	for tag in RUNS:
		for k in range(nwin):
			F1 = np.hypot(Fx[tag][k, 0], Fy[tag][k, 0])
			Fhi = sum(np.hypot(Fx[tag][k, n], Fy[tag][k, n])
					  for n in range(1, NM))
			tot = F1 + Fhi
			with np.errstate(invalid='ignore', divide='ignore'):
				fhi = np.where(tot > 0, Fhi/tot, np.nan)
				ex = np.where(F1 > 0, Fx[tag][k, 0]/np.maximum(F1, 1e-12), 0.)
				ey = np.where(F1 > 0, Fy[tag][k, 0]/np.maximum(F1, 1e-12), 0.)
				gx = np.gradient(np.nan_to_num(fhi), axis=1)/np.where(np.abs(dxm) > 1, dxm, np.nan)
				gy = np.gradient(np.nan_to_num(fhi), axis=0)/np.where(np.abs(dym) > 1, dym, np.nan)
			scat[tag][k] = np.where(mask > 0,
									(np.nan_to_num(gx)*ex
									 + np.nan_to_num(gy)*ey)*1e3, np.nan)

	# ---- cross shelf sections of the modal flux
	sec_names = list(SECTIONS.keys())
	sec_d = np.full((len(sec_names), SECTION_NPT), np.nan)
	sec_h = np.full_like(sec_d, np.nan)
	sec_F = {t: np.full((len(sec_names), NM, SECTION_NPT), np.nan)
			 for t in RUNS}
	kpw = wins.index(PLOT_WINDOW) if PLOT_WINDOW in wins else 0
	for si, sname in enumerate(sec_names):
		lo0, la0, lo1, la1 = SECTIONS[sname]
		sl_ = np.linspace(lo0, lo1, SECTION_NPT)
		sa_ = np.linspace(la0, la1, SECTION_NPT)
		sj, si_ = nearest_ji(tree, jj, ii, sl_, sa_)
		d0 = np.hypot((sl_ - lo0)*111.2*np.cos(np.deg2rad(sa_)),
					  (sa_ - la0)*111.2)
		sec_d[si] = d0
		sec_h[si] = h[sj, si_]
		for tag in RUNS:
			for n in range(NM):
				sec_F[tag][si, n] = np.hypot(Fx[tag][kpw, n][sj, si_],
											 Fy[tag][kpw, n][sj, si_])

	# ---- ray tracing through the peak window c1 field of REF
	ray_names = [s for s in RAY_SITES if s in GEN_SITES]
	npath = RAY_FAN.size
	rays = np.full((len(ray_names), npath, RAY_NSTEP, 2), np.nan)
	ray_turn = np.zeros((len(ray_names), npath), dtype='int8')
	c1map = Cn['ref'][kpw, 0]
	for r, sname in enumerate(ray_names):
		slon, slat_ = GEN_SITES[sname]
		j0, i0 = nearest_ji(tree, jj, ii, slon, slat_)
		j0, i0 = int(j0[0]), int(i0[0])
		base = np.arctan2(-noy[j0, i0], -nox[j0, i0])   # shoreward normal
		p, tn = trace_rays(c1map, lon, lat, mask, slon, slat_, base + RAY_FAN)
		rays[r] = p
		ray_turn[r] = tn.astype('int8')
		print(f'  rays from {sname}: {tn.sum()}/{npath} turned')

	# ---- write
	print('writing', outfile)
	coords = dict(
		window=('window', wins),
		mode=('mode', np.arange(1, NM + 1)),
		eta_rho=('eta_rho', np.arange(ny)),
		xi_rho=('xi_rho', np.arange(nx)),
		break_pt=('break_pt', np.arange(nb)),
		block=('block', np.arange(nblock)),
		section=('section', sec_names),
		spt=('spt', np.arange(SECTION_NPT)),
		site=('site', ray_names),
		ray=('ray', np.arange(npath)),
		step=('step', np.arange(RAY_NSTEP)),
	)
	dv = {}
	for tag in RUNS:
		dv[f'Fx_{tag}'] = (('window', 'mode', 'eta_rho', 'xi_rho'), Fx[tag])
		dv[f'Fy_{tag}'] = (('window', 'mode', 'eta_rho', 'xi_rho'), Fy[tag])
		dv[f'c_{tag}'] = (('window', 'mode', 'eta_rho', 'xi_rho'), Cn[tag])
		dv[f'scat_{tag}'] = (('window', 'eta_rho', 'xi_rho'), scat[tag])
		dv[f'Fperp_{tag}'] = (('window', 'mode', 'break_pt'), Fperp[tag])
		dv[f'Pon_{tag}'] = (('window', 'mode'), P_on[tag])
		dv[f'Poff_{tag}'] = (('window', 'mode'), P_off[tag])
		dv[f'Pon_block_{tag}'] = (('block', 'mode'), Pon_b[tag])
		dv[f'Poff_block_{tag}'] = (('block', 'mode'), Poff_b[tag])
		dv[f'gamma_{tag}'] = (('window', 'break_pt'), Gam[tag])
		dv[f'c1_shelf_{tag}'] = (('window', 'break_pt'), c1s[tag])
		dv[f'c1_deep_{tag}'] = (('window', 'break_pt'), c1d[tag])
		dv[f'phi_on_{tag}'] = (('window', 'break_pt'), phi_on[tag])
		dv[f'secF_{tag}'] = (('section', 'mode', 'spt'), sec_F[tag])
	dv.update(dict(
		lon_rho=(('eta_rho', 'xi_rho'), lon),
		lat_rho=(('eta_rho', 'xi_rho'), lat),
		h=(('eta_rho', 'xi_rho'), h),
		mask_rho=(('eta_rho', 'xi_rho'), mask),
		angle=(('eta_rho', 'xi_rho'), ang),
		break_lon=('break_pt', blon),
		break_lat=('break_pt', blat),
		break_dist=('break_pt', bdist),
		hp_break=(('window', 'break_pt'), hp_break),
		block_time=('block', res[RUNS[0]][2]),
		sec_dist=(('section', 'spt'), sec_d),
		sec_h=(('section', 'spt'), sec_h),
		rays=(('site', 'ray', 'step', 'xy'), rays),
		ray_turned=(('site', 'ray'), ray_turn),
	))
	coords['xy'] = ('xy', ['lon', 'lat'])

	ds_out = xr.Dataset(dv, coords=coords)
	ds_out.attrs.update(dict(
		realistic=REALISTIC, control=CONTROL,
		plot_window=PLOT_WINDOW, nmode=NM,
		break_isobath=BREAK_ISOBATH, gamma_dist_km=GAMMA_DIST/1e3,
		eos_mode=EOS_MODE,
		eos=eos_provenance(),
		windows=';'.join(f'{k}:{v[0]}..{v[1]}' for k, v in WINDOWS.items()),
		gen_sites_main=','.join(GEN_SITES_MAIN),
		note='Modes are normalised so (1/H) int phi_n^2 dz = 1, hence '
			 'F_n = H <u_n p_n>. The break crossing flux is split into '
			 'onshore and offshore parts BEFORE integration, because the '
			 'generation sits on the break and no incident wave is '
			 'defined. Gamma = (c1_deep - c1_shelf)/(c1_deep + c1_shelf) '
			 f'sampled {GAMMA_DIST/1e3:.0f} km either side along the '
			 'bathymetric normal.',
	))
	try:
		enc = {v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars}
		ds_out.to_netcdf(outfile, encoding=enc)
	except ValueError as err:
		# some backends do not accept compression, better an uncompressed
		# file than losing the whole assembly at the final step
		warnings.warn(f'compressed write failed ({err}), writing plain')
		ds_out.to_netcdf(outfile)

	# ---- summary
	print('\n--- summary, break crossing modal power in MW ---')
	for k, w in enumerate(wins):
		for tag in RUNS:
			on = P_on[tag][k]; off = P_off[tag][k]
			tot = on.sum() + off.sum()
			print(f'{w:>11} {tag:>5}  onshore ' +
				  ' '.join(f'M{n+1} {on[n]:6.1f}' for n in range(NM)) +
				  '   offshore ' +
				  ' '.join(f'M{n+1} {off[n]:6.1f}' for n in range(NM)) +
				  f'   onshore fraction {on.sum()/tot if tot else np.nan:5.3f}')
	print()
	for k, w in enumerate(wins):
		line = f'{w:>11}  mode 1 onshore fraction  '
		for tag in RUNS:
			on = P_on[tag][k, 0]; off = P_off[tag][k, 0]
			line += f'{tag} {on/(on+off) if (on+off) else np.nan:5.3f}   '
		line += '  median Gamma  '
		for tag in RUNS:
			line += f'{tag} {np.nanmedian(Gam[tag][k]):5.3f}   '
		print(line)
	print()
	for k, w in enumerate(wins):
		line = f'{w:>11}  higher mode share of the break flux  '
		for tag in RUNS:
			tot = P_on[tag][k].sum() + P_off[tag][k].sum()
			hi = (P_on[tag][k, 1:].sum() + P_off[tag][k, 1:].sum())
			line += f'{tag} {100*hi/tot if tot else np.nan:5.1f}%   '
		print(line)


def main():
	print('opening runs')
	dsr, dsc, geom, brk = build_geometry()
	wins = list(WINDOWS.keys())

	res = {}
	for tag, ds in (('ref', dsr), ('ctrl', dsc)):
		if tag not in RUNS:
			continue
		cached = load_cache(tag) if USE_CACHE else None
		if cached is not None:
			res[tag] = cached
			continue
		tic = _time.time()
		res[tag] = process_run(ds, tag, wins, geom, brk)
		print(f'  {tag} done in {(_time.time()-tic)/60:.1f} min')
		try:
			save_cache(tag, res[tag])
		except Exception as err:
			warnings.warn(f'could not cache {tag}: {err}')

	assemble(res, geom, brk, wins)


if __name__ == '__main__':
	import sys
	if '--assemble-only' in sys.argv:
		print('assemble only, using the cached streaming results')
		dsr, dsc, geom, brk = build_geometry()
		wins = list(WINDOWS.keys())
		res = {t: load_cache(t) for t in RUNS}
		missing = [t for t, v in res.items() if v is None]
		if missing:
			raise SystemExit(f'no cache for {missing}, run without the flag')
		assemble(res, geom, brk, wins)
	else:
		main()