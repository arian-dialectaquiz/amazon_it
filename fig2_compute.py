################
"""
fig02_compute.py
=================================================================
Figure 2  -  The plume as a waveguide
Amazon shelf internal tide manuscript

Computes and stores everything panels (a) to (f) need:

  (a-c) maps of the mode one phase speed c1 for the three hydrological
		windows in the realistic run
  (d)   dc1 = REF - CTRL at peak discharge, with the surface 35 isohaline
  (e)   mode one structure Phi1 and phi1 at inner shelf, mid shelf and
		shelf break stations, REF against CTRL
  (f)   mode one deformation radius against local shelf width, and the
		surface trapping index against plume thickness

Output:  fig02_data.nc   (read by fig02_plot.py)

Requires fig01_compute.py in the same directory, whose solver and helpers
are imported rather than duplicated, so there is one source of truth for
the eigenproblem.

Physics notes
-------------
* THE EQUATOR MATTERS HERE. The domain runs from 4.5 S to 6 N, so f
  changes sign inside it and the mid latitude radius c1/f is unbounded
  near the equator. Within a few degrees the correct meridional trapping
  scale is the equatorial radius Req = (c1/beta)^(1/2). Both are computed
  and the effective radius is taken as the smaller of the two, which
  matches the standard result that the equatorial regime applies inside
  |y| < Req. Sites A, B and C all sit inside that band, so panel (f) must
  be read with Req, not with c1/f.
* The surface trapping index uses phi1, the structure function for
  horizontal velocity and pressure, because that is what carries the
  kinetic energy. Phi1, the vertical velocity structure, is stored too
  for panel (e) but is not used for the index.
* The two layer estimate c1 = sqrt(g' hp (H-hp)/H) is computed alongside
  the full Sturm-Liouville solution as an interpretive scaling. Where the
  two agree the plume is behaving as a clean two layer waveguide, where
  they diverge the continuous stratification matters.
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

# one source of truth for the solver and the grid helpers
from fig01_compute import (
	REALISTIC, CONTROL, GRIDFILE, WINDOWS, MOORINGS, SECTIONS,
	GEN_SITES, GEN_SITES_MAIN, BREAK_ISOBATH, SHELF_MAX_DEPTH,
	G, RHO0, OMEGA_M2, T_M2, OMEGA_E, NMODES, N2_FLOOR, C1_MIN,
	open_run, compute_N2, vertical_modes, extract_isobath,
	build_tree, nearest_ji, HAS_GSW,
)

if HAS_GSW:
	import gsw


# =====================================================================
# CONFIGURATION
# =====================================================================
OUTFILE = 'fig02_data.nc'

# stations for the mode structure panel, inner shelf to shelf break.
# M1, M2 and M3 sit on the 16, 60 and 103 m isobaths in the AMASSEDS array.
PROFILE_STATIONS = ['M1', 'M2', 'M3', 'V200']
STATION_DEPTHS = [200.]          # virtual stations added to the mooring set
SECTION_FOR_STATIONS = 'S1'

PLOT_WINDOW = 'peak'             # window used for panel (d)
ISOHALINE = 35.0                 # surface contour drawn on panel (d)

BETA_EQ = 2*OMEGA_E/6.371e6      # df/dy at the equator, s-1 m-1
R_EARTH = 6.371e6

MAX_R = 5000.                    # km, cap on the reported radius


# =====================================================================
# HELPERS
# =====================================================================
def phi_from_Phi(Phi, zw, c):
	"""
	phi_n = rho0 c_n^2 dPhi_n/dz, the structure function for horizontal
	velocity and pressure, evaluated on rho points.

	Phi : (Nw, M) mode structure at w points
	zw  : (Nw, M) w point depths
	c   : (M,) modal phase speed
	Returns phi (Nz, M) with Nz = Nw - 1.
	"""
	dPhi = np.diff(Phi, axis=0)
	dz = np.diff(zw, axis=0)
	with np.errstate(divide='ignore', invalid='ignore'):
		phi = RHO0*(c[None, :]**2)*dPhi/dz
	return phi


def surface_trapping(phi, zr, zw, hp):
	"""
	Fraction of the mode one horizontal kinetic energy held above the
	plume base, T = int_{-hp}^{0} phi^2 dz / int_{-H}^{0} phi^2 dz.

	Normalisation of phi cancels, so this is independent of how the mode
	was scaled.
	"""
	dz = np.diff(zw, axis=0)
	e = (phi**2)*dz
	tot = np.nansum(e, axis=0)
	top = np.nansum(np.where(zr >= -hp[None, :], e, 0.0), axis=0)
	with np.errstate(divide='ignore', invalid='ignore'):
		return np.where(tot > 0, top/tot, np.nan)


def deformation_radii(c1, lat):
	"""
	Mid latitude and equatorial deformation radii, and the effective one.

	R_f  = c1/|f|                 valid away from the equator
	R_eq = sqrt(c1/beta)          valid within |y| < R_eq of the equator
	R_eff = min(R_f, R_eq)        the standard matching between the two

	All returned in kilometres.
	"""
	f = 2*OMEGA_E*np.sin(np.deg2rad(lat))
	beta = 2*OMEGA_E*np.cos(np.deg2rad(lat))/R_EARTH
	with np.errstate(divide='ignore', invalid='ignore'):
		Rf = np.where(np.abs(f) > 0, c1/np.abs(f), np.inf)/1000.
	Req = np.sqrt(np.maximum(c1, 0)/beta)/1000.
	Reff = np.minimum(Rf, Req)
	Rf = np.where(np.isfinite(Rf), np.minimum(Rf, MAX_R), MAX_R)
	return Rf, Req, Reff


def density_from_TS(temp, salt, z_rho, lon, lat):
	"""Potential density referenced to the surface."""
	if HAS_GSW:
		p = gsw.p_from_z(z_rho, lat)
		SA = gsw.SA_from_SP(salt, p, lon, lat)
		CT = gsw.CT_from_pt(SA, temp)
		return np.asarray(gsw.rho(SA, CT, 0.0))
	return RHO0*(1 - 1.7e-4*(temp - 10.) + 7.6e-4*(salt - 35.))


def two_layer_c1(rho, z_rho, zw, hp, H):
	"""
	Two layer estimate c1 = sqrt(g' hp (H - hp)/H), with g' formed from
	the mean density above and below the plume base.
	"""
	above = z_rho >= -hp[None, :]
	below = (z_rho < -hp[None, :]) & (z_rho >= -np.minimum(3*hp, H)[None, :])
	nb = below.sum(axis=0)
	below = np.where(nb[None, :] > 0, below, z_rho < -hp[None, :])

	ru = np.nansum(np.where(above, rho, 0.), axis=0)/np.maximum(above.sum(axis=0), 1)
	rl = np.nansum(np.where(below, rho, 0.), axis=0)/np.maximum(below.sum(axis=0), 1)
	gp = G*(rl - ru)/RHO0
	with np.errstate(invalid='ignore'):
		c = np.sqrt(np.maximum(gp, 0)*hp*np.maximum(H - hp, 0)/np.maximum(H, 1))
	return c, gp


def coastline_tree(lon, lat, mask):
	"""KD tree on the model coastline, for shelf width by nearest distance."""
	fig = plt.figure()
	cs = plt.contour(lon, lat, mask.astype(float), levels=[0.5])
	segs = [s for s in cs.allsegs[0] if len(s) > 5]
	plt.close(fig)
	if not segs:
		raise RuntimeError('no coastline found in the land mask')
	pts = np.vstack(segs)
	return cKDTree(pts), pts


def km_between(lon0, lat0, lon1, lat1):
	dx = 111.2*np.cos(np.deg2rad(0.5*(lat0 + lat1)))*(lon1 - lon0)
	dy = 111.2*(lat1 - lat0)
	return np.hypot(dx, dy)


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
	ny, nx = h.shape
	Nz = dsr.sizes['s_rho']; Nw = dsr.sizes['s_w']
	flat = mask.ravel() > 0
	nwet = int(flat.sum())
	print(f'  grid {ny} x {nx}, {Nz} levels, {nwet} wet columns')

	wins = list(WINDOWS.keys())
	nwin = len(wins)

	# -----------------------------------------------------------------
	# stations for the mode structure panel
	# -----------------------------------------------------------------
	tree, jj, ii = build_tree(lon, lat, mask)
	lo0, la0, lo1, la1 = SECTIONS[SECTION_FOR_STATIONS]
	slon = np.linspace(lo0, lo1, 400); slat = np.linspace(la0, la1, 400)
	sj, si = nearest_ji(tree, jj, ii, slon, slat)
	sh = h[sj, si]

	st_names, st_j, st_i = [], [], []
	for name, (mlo, mla) in MOORINGS.items():
		j, i = nearest_ji(tree, jj, ii, mlo, mla)
		st_names.append(name); st_j.append(int(j[0])); st_i.append(int(i[0]))
	for d in STATION_DEPTHS:
		k = int(np.nanargmin(np.abs(sh - d)))
		st_names.append(f'V{int(d)}')
		st_j.append(int(sj[k])); st_i.append(int(si[k]))
	keep = [n for n, s in enumerate(st_names) if s in PROFILE_STATIONS]
	st_names = [st_names[n] for n in keep]
	st_j = np.array([st_j[n] for n in keep])
	st_i = np.array([st_i[n] for n in keep])
	nst = len(st_names)
	st_h = h[st_j, st_i]
	print('  profile stations:',
		  ', '.join(f'{s} ({d:.0f} m)' for s, d in zip(st_names, st_h)))

	# -----------------------------------------------------------------
	# shelf break, coastline and local shelf width
	# -----------------------------------------------------------------
	print('shelf break and shelf width')
	blon, blat, bdist = extract_isobath(lon, lat,
										np.where(mask > 0, h, np.nan),
										BREAK_ISOBATH)
	ctree, cpts = coastline_tree(lon, lat, mask)

	# width at each break point, distance to the nearest coastline point
	d, k = ctree.query(np.column_stack([blon, blat]))
	W_break = km_between(cpts[k, 0], cpts[k, 1], blon, blat)

	# map every wet point to its nearest break point, so each shelf point
	# inherits the shelf width of its own alongshore position
	btree = cKDTree(np.column_stack([blon, blat]))
	_, kb = btree.query(np.column_stack([lon.ravel(), lat.ravel()]))
	shelf_width = W_break[kb].reshape(ny, nx)
	shelf_width = np.where(mask > 0, shelf_width, np.nan)
	print(f'  shelf width {np.nanmin(W_break):.0f} to '
		  f'{np.nanmax(W_break):.0f} km, median {np.nanmedian(W_break):.0f} km')

	# -----------------------------------------------------------------
	# window loop
	# -----------------------------------------------------------------
	keys2d = ['c1', 'hp', 'trap', 'gprime', 'c1_2lay', 'sss',
			  'Rf', 'Req', 'Reff']
	out = {f'{k}_{tag}': np.full((nwin, ny, nx), np.nan)
		   for k in keys2d for tag in ('ref', 'ctrl')}

	prof = {f'{k}_{tag}': np.full((nwin, nst, n), np.nan)
			for k, n in (('Phi1', Nw), ('phi1', Nz), ('N2', Nw),
						 ('zw', Nw), ('zr', Nz), ('salt', Nz), ('rho', Nz))
			for tag in ('ref', 'ctrl')}
	prof.update({f'c1st_{tag}': np.full((nwin, nst), np.nan)
				 for tag in ('ref', 'ctrl')})

	for iw, wname in enumerate(wins):
		t0, t1 = WINDOWS[wname]
		for tag, ds in (('ref', dsr), ('ctrl', dsc)):
			sub = ds.sel(ocean_time=slice(t0, t1))
			if sub.sizes['ocean_time'] == 0:
				print(f'  {wname}/{tag}: no times in window, skipped')
				continue
			tic = _time.time()
			print(f'  {wname}/{tag}: {sub.sizes["ocean_time"]} records', end='')

			temp = sub.temp.mean('ocean_time').values
			salt = sub.salt.mean('ocean_time').values
			zr = sub.z_rho.mean('ocean_time').values
			zw = sub.z_w.mean('ocean_time').values

			N2 = compute_N2(temp, salt, zr, zw, lon, lat)
			rho = density_from_TS(temp, salt, zr, lon, lat)

			N2f = N2.reshape(Nw, -1)[:, flat]
			zwf = zw.reshape(Nw, -1)[:, flat]
			zrf = zr.reshape(Nz, -1)[:, flat]
			rhof = rho.reshape(Nz, -1)[:, flat]
			Hf = np.abs(zwf[0])

			c, Phi = vertical_modes(N2f, zwf, nmodes=NMODES, free_surface=True)
			c1f = c[1]
			Phi1f = Phi[:, 1, :]
			phi1f = phi_from_Phi(Phi1f, zwf, c1f)

			# plume thickness, depth of the maximum N2 in the upper column
			zmid = -0.5*Hf
			top = zwf > np.maximum(zmid, -120.)
			kmax = np.nanargmax(np.where(top, N2f, -np.inf), axis=0)
			hpf = -np.take_along_axis(zwf, kmax[None], axis=0)[0]

			trapf = surface_trapping(phi1f, zrf, zwf, hpf)
			c2lf, gpf = two_layer_c1(rhof, zrf, zwf, hpf, Hf)

			def unflat(v):
				a = np.full(ny*nx, np.nan); a[flat] = v
				return a.reshape(ny, nx)

			out[f'c1_{tag}'][iw] = unflat(c1f)
			out[f'hp_{tag}'][iw] = unflat(hpf)
			out[f'trap_{tag}'][iw] = unflat(trapf)
			out[f'gprime_{tag}'][iw] = unflat(gpf)
			out[f'c1_2lay_{tag}'][iw] = unflat(c2lf)
			out[f'sss_{tag}'][iw] = np.where(mask > 0, salt[-1], np.nan)

			c1map = out[f'c1_{tag}'][iw]
			Rf, Req, Reff = deformation_radii(c1map, lat)
			out[f'Rf_{tag}'][iw] = np.where(mask > 0, Rf, np.nan)
			out[f'Req_{tag}'][iw] = np.where(mask > 0, Req, np.nan)
			out[f'Reff_{tag}'][iw] = np.where(mask > 0, Reff, np.nan)

			# profiles at the chosen stations
			colidx = np.full(ny*nx, -1, dtype=int)
			colidx[flat] = np.arange(nwet)
			colidx = colidx.reshape(ny, nx)
			for n in range(nst):
				m = colidx[st_j[n], st_i[n]]
				if m < 0:
					continue
				prof[f'Phi1_{tag}'][iw, n] = Phi1f[:, m]
				prof[f'phi1_{tag}'][iw, n] = phi1f[:, m]
				prof[f'N2_{tag}'][iw, n] = N2f[:, m]
				prof[f'zw_{tag}'][iw, n] = zwf[:, m]
				prof[f'zr_{tag}'][iw, n] = zrf[:, m]
				prof[f'salt_{tag}'][iw, n] = salt[:, st_j[n], st_i[n]]
				prof[f'rho_{tag}'][iw, n] = rhof[:, m]
				prof[f'c1st_{tag}'][iw, n] = c1f[m]

			print(f'   ({_time.time()-tic:.0f} s)')

	# difference fields
	dc1 = out['c1_ref'] - out['c1_ctrl']
	dhp = out['hp_ref'] - out['hp_ctrl']
	with np.errstate(divide='ignore', invalid='ignore'):
		rc1 = out['c1_ref']/out['c1_ctrl']

	# -----------------------------------------------------------------
	# assemble and write
	# -----------------------------------------------------------------
	print('writing', OUTFILE)
	coords = dict(
		window=('window', wins),
		eta_rho=('eta_rho', np.arange(ny)),
		xi_rho=('xi_rho', np.arange(nx)),
		station=('station', st_names),
		s_rho=('s_rho', np.arange(Nz)),
		s_w=('s_w', np.arange(Nw)),
		break_pt=('break_pt', np.arange(blon.size)),
	)
	dv = {}
	for k, v in out.items():
		dv[k] = (('window', 'eta_rho', 'xi_rho'), v)
	dv['dc1'] = (('window', 'eta_rho', 'xi_rho'), dc1)
	dv['dhp'] = (('window', 'eta_rho', 'xi_rho'), dhp)
	dv['rc1'] = (('window', 'eta_rho', 'xi_rho'), rc1)

	for k, v in prof.items():
		if v.ndim == 3:
			dim = 's_w' if v.shape[2] == Nw else 's_rho'
			dv[k] = (('window', 'station', dim), v)
		else:
			dv[k] = (('window', 'station'), v)

	dv.update(dict(
		lon_rho=(('eta_rho', 'xi_rho'), lon),
		lat_rho=(('eta_rho', 'xi_rho'), lat),
		h=(('eta_rho', 'xi_rho'), h),
		mask_rho=(('eta_rho', 'xi_rho'), mask),
		shelf_width=(('eta_rho', 'xi_rho'), shelf_width),
		break_lon=('break_pt', blon),
		break_lat=('break_pt', blat),
		break_width=('break_pt', W_break),
		station_lon=('station', lon[st_j, st_i]),
		station_lat=('station', lat[st_j, st_i]),
		station_h=('station', st_h),
		station_j=('station', st_j),
		station_i=('station', st_i),
	))

	ds_out = xr.Dataset(dv, coords=coords)
	ds_out.attrs.update(dict(
		realistic=REALISTIC, control=CONTROL,
		plot_window=PLOT_WINDOW,
		isohaline=ISOHALINE,
		break_isobath=BREAK_ISOBATH,
		shelf_max_depth=SHELF_MAX_DEPTH,
		c1_min=C1_MIN,
		beta_eq=BETA_EQ,
		windows=';'.join(f'{k}:{v[0]}..{v[1]}' for k, v in WINDOWS.items()),
		gen_sites_main=','.join(GEN_SITES_MAIN),
		note='Reff = min(c1/|f|, sqrt(c1/beta)). The domain crosses the '
			 'equator, so the equatorial radius is the relevant one for '
			 'sites A, B and C.',
	))

	enc = {v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars}
	ds_out.to_netcdf(OUTFILE, encoding=enc)

	# -----------------------------------------------------------------
	# summary, the numbers for the text
	# -----------------------------------------------------------------
	print('\n--- summary ---')
	outer = (h > 50) & (h < SHELF_MAX_DEPTH) & (mask > 0)
	mid = (h > 20) & (h <= 50) & (mask > 0)
	for iw, w in enumerate(wins):
		r = np.nanmedian(out['c1_ref'][iw][outer])
		c = np.nanmedian(out['c1_ctrl'][iw][outer])
		rm = np.nanmedian(out['c1_ref'][iw][mid])
		cm = np.nanmedian(out['c1_ctrl'][iw][mid])
		tr = np.nanmedian(out['trap_ref'][iw][outer])
		tc = np.nanmedian(out['trap_ctrl'][iw][outer])
		hr = np.nanmedian(out['hp_ref'][iw][outer])
		Rr = np.nanmedian(out['Reff_ref'][iw][outer])
		print(f'{w:>11}  outer shelf c1 REF {r:5.2f} CTRL {c:5.2f} '
			  f'(x{r/c:4.2f})  |  mid shelf x{rm/cm:4.2f}  |  '
			  f'trap REF {tr:4.2f} CTRL {tc:4.2f}  |  hp {hr:4.1f} m  |  '
			  f'Reff {Rr:5.0f} km')
	print(f'\nmedian shelf width {np.nanmedian(W_break):.0f} km')
	iwp = wins.index(PLOT_WINDOW)
	ratio = np.nanmedian(out['Reff_ref'][iwp][outer])/np.nanmedian(W_break)
	print(f'Reff / shelf width at {PLOT_WINDOW} = {ratio:.2f}')
	print('A ratio above one means the shelf is narrower than a deformation '
		  'radius, so cross shelf propagation is not rotationally controlled.')

	agree = np.nanmedian(np.abs(out['c1_2lay_ref'][iwp][outer]
								- out['c1_ref'][iwp][outer])
						 / out['c1_ref'][iwp][outer])
	print(f'two layer estimate departs from the full solution by '
		  f'{100*agree:.0f}% on the outer shelf at {PLOT_WINDOW}')


if __name__ == '__main__':
	main()

####---> Summary
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig02_summary.py
=================================================================
Print the Figure 2 summary table from an existing fig02_data.nc.

Everything below is already stored in the file, so this does NOT
recompute the vertical modes. Use it whenever the statistics change
but the underlying fields have not.

	python fig02_summary.py
=================================================================
"""

import numpy as np
import xarray as xr

DATA = 'fig02_data.nc'

ds = xr.open_dataset(DATA)
h = ds.h.values
mask = ds.mask_rho.values
wins = [str(w) for w in ds.window.values]
C1_MIN = float(ds.attrs.get('c1_min', 0.05))
SHELF_MAX = float(ds.attrs.get('shelf_max_depth', 250.))
W_break = ds.break_width.values

outer = (h > 50) & (h < SHELF_MAX) & (mask > 0)
mid = (h > 20) & (h <= 50) & (mask > 0)

print(f'--- Figure 2 summary, from {DATA} ---')
print(f'ratios use only cells where BOTH runs support a mode one, '
	  f'c1 > {C1_MIN} m/s\n')

for iw, w in enumerate(wins):
	cr = ds.c1_ref.isel(window=iw).values
	cc = ds.c1_ctrl.isel(window=iw).values
	both = np.isfinite(cr) & np.isfinite(cc) & (cr > C1_MIN) & (cc > C1_MIN)

	def stat(m):
		s = m & both
		if s.sum() < 20:
			return np.nan, np.nan, np.nan, 0.
		return (np.nanmedian(cr[s]), np.nanmedian(cc[s]),
				np.nanmedian(cr[s]) - np.nanmedian(cc[s]),
				100.*s.sum()/max((m & np.isfinite(cr)).sum(), 1))

	ro, co, do, fo = stat(outer)
	rm, cm, dm, fm = stat(mid)
	tr = np.nanmedian(ds.trap_ref.isel(window=iw).values[outer])
	tc = np.nanmedian(ds.trap_ctrl.isel(window=iw).values[outer])
	hp = np.nanmedian(ds.hp_ref.isel(window=iw).values[outer])
	Rr = np.nanmedian(ds.Reff_ref.isel(window=iw).values[outer])
	Rc = np.nanmedian(ds.Reff_ctrl.isel(window=iw).values[outer])

	print(f'{w:>11}  outer c1 REF {ro:5.2f} CTRL {co:5.2f} '
		  f'(x{ro/co:4.2f}, +{do:4.2f} m/s, {fo:3.0f}% of cells)')
	print(f'{"":>11}  mid   c1 REF {rm:5.2f} CTRL {cm:5.2f} '
		  f'(x{rm/cm:4.2f}, +{dm:4.2f} m/s, {fm:3.0f}% of cells)')
	print(f'{"":>11}  trapping index REF {tr:4.2f} CTRL {tc:4.2f}   '
		  f'plume thickness {hp:4.1f} m   '
		  f'Reff REF {Rr:5.0f} CTRL {Rc:5.0f} km')

	dead = mid & np.isfinite(cr) & (cc <= C1_MIN)
	live = mid & np.isfinite(cr)
	if dead.sum():
		print(f'{"":>11}  CTRL supports NO mode one on '
			  f'{100.*dead.sum()/max(live.sum(), 1):4.0f}% of the mid shelf, '
			  f'where REF does')
	print()

Wmed = np.nanmedian(W_break)
print(f'median shelf width {Wmed:.0f} km, range {np.nanmin(W_break):.0f} '
	  f'to {np.nanmax(W_break):.0f} km\n')
for iw, w in enumerate(wins):
	Rr = np.nanmedian(ds.Reff_ref.isel(window=iw).values[outer])
	Rc = np.nanmedian(ds.Reff_ctrl.isel(window=iw).values[outer])
	print(f'  Reff / shelf width, {w:>11}   REF {Rr/Wmed:4.2f}   '
		  f'CTRL {Rc/Wmed:4.2f}')
print('\nBelow one means the shelf is WIDER than a deformation radius, so')
print('mode one is rotationally trapped within Reff of the coast and the')
print('cross shelf propagation is not a free non rotating problem.')

iwp = wins.index(ds.attrs.get('plot_window', 'peak'))
c2 = ds.c1_2lay_ref.isel(window=iwp).values
c1 = ds.c1_ref.isel(window=iwp).values
s = outer & np.isfinite(c1) & np.isfinite(c2) & (c1 > C1_MIN)
print(f'\ntwo layer estimate departs from the full solution by '
	  f'{100*np.nanmedian(np.abs(c2[s]-c1[s])/c1[s]):.0f}% on the outer shelf')