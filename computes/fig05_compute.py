#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig05_compute.py
=================================================================
Figure 5  -  Radiation and where the energy is lost
Amazon shelf internal tide manuscript, version 3

Computes and stores everything panels (a) to (f) need:

  (a to c) depth integrated baroclinic flux with propagation vectors,
		   three windows, REF
  (d)      control volume budget by band, both runs
  (e)      cross isobath power against isobath depth, total and by mode
  (f)      radiated fraction and the share lost within the margin

Outputs  fig05_data.nc, and the report written by fig05_report.py

Two changes from the original plan, both forced by earlier figures
-----------------------------------------------------------------
1. The budget is closed with LINE INTEGRALS across isobaths rather than
   with the pointwise divergence of the flux. The two are the same
   quantity by the divergence theorem, but the divergence is a
   horizontal derivative taken over two or three grid cells, and
   Figure 1 places the mode one wavelength at eight to twelve cells
   over the outer shelf. A line integral of F along a contour needs no
   derivative at all, so it survives where the divergence does not.

   The control volume of a depth band is bounded by two isobaths and,
   because the configuration cuts the margin alongshore, by the domain
   edges. Those edge segments are integrated explicitly and reported,
   which turns the caveat carried in the earlier draft into a number.

2. The modal flux is restricted to modes one and two and to h > 250 m.
   Unlike the modal partition of the conversion in Figure 4, which
   needs vertical resolution alone, a modal FLUX represents a
   propagating wave and needs the horizontal wavelength resolved as
   well. The points per wavelength of each mode are computed and the
   area clearing eight is reported alongside every modal number.

Method
------
	F_bc    = int <u_bc p_bc> dz                    depth integrated
	F_n     = H <u_n p_n>                           modal, u_n, p_n by
	                                                projection onto phi_n
	Phi(h)  = closed line integral of F . n_offshore along an isobath
	E       = Phi(1000 m) / int_{gen} C over 50 to 1000 m

The harmonic coefficients come from fig03_harmonics.nc and the
conversion from fig03_data.nc, so no pass over the model is needed. The
vertical modes use the window mean stratification cached in MEANCACHE by
fig04_compute.py.
=================================================================
"""

import os
import functools
import warnings
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fig01_compute import (
	REALISTIC, CONTROL, WINDOWS, BREAK_ISOBATH, T_M2,
	open_run, compute_N2, vertical_modes, extract_isobath,
	build_tree, nearest_ji,
)
from fig02_compute import phi_from_Phi
from fig04_compute import mean_state

print = functools.partial(print, flush=True)


# =====================================================================
# CONFIGURATION
# =====================================================================
HARMFILE = 'fig03_harmonics.nc'
FIG03    = 'fig03_data.nc'          # the conversion, already computed
OUTFILE  = 'fig05_data.nc'

NMODE = 4                 # modes solved
NMODE_FLUX = 2            # modes quoted in the radiated partition
PPW_MIN = 8.              # points per wavelength required of a modal flux

# isobaths at which the cross shelf power is evaluated, shallow to deep
ISOBATHS = [250., 500., 750., 1000., 1500., 2000., 2500., 3000.]
E_ISOBATH = 1000.         # the isobath at which the radiated fraction is
                          # quoted, seaward of every generation site
GEN_BAND = (50., 1000.)   # the band whose generation E is measured against
BANDS = [('shelf, 50 to 250 m',      50.,  250., 'shelf'),
		 ('upper slope, 250 to 1000 m', 250., 1000., 'slope'),
		 ('deep, beyond 1000 m',     1000., 1e9,  'deep')]

RUNS = ('ref', 'ctrl')


# =====================================================================
# HELPERS
# =====================================================================
def grad_h_components(h, pm, pn):
	return np.gradient(h, axis=1)*pm, np.gradient(h, axis=0)*pn


def area_weights(pm, pn, mask):
	return np.where(mask > 0, 1.0/(pm*pn), 0.0)


def cycle_mean(a1, b1, a2, b2):
	return 0.5*(a1*a2 + b1*b2)


def project(a, phi, dz, H):
	"""Modal amplitude of a field at rho points."""
	return np.nansum(a*phi*dz, axis=0)/np.maximum(H, 1e-9)


def isobath_power(Fxi, Feta, lon, lat, h, mask, pm, pn, level,
				  tree, jj, ii):
	"""
	Power crossing an isobath, positive offshore.

	The contour is extracted from the bathymetry, the offshore normal is
	taken from the bathymetric gradient, and the flux is sampled at the
	nearest wet column. No horizontal derivative of the flux is taken,
	which is the point of doing it this way.

	Returns the integrated power in W, the along contour profile in
	W per metre, and the contour coordinates.
	"""
	try:
		blon, blat, bdist = extract_isobath(lon, lat,
											np.where(mask > 0, h, np.nan),
											level)
	except RuntimeError:
		return np.nan, None, None, None, None
	bj, bi = nearest_ji(tree, jj, ii, blon, blat)

	dhx, dhy = grad_h_components(h, pm, pn)
	g = np.hypot(dhx, dhy)
	nxi = np.where(g > 0, dhx/np.maximum(g, 1e-12), 0.)[bj, bi]
	neta = np.where(g > 0, dhy/np.maximum(g, 1e-12), 0.)[bj, bi]

	fn = Fxi[bj, bi]*nxi + Feta[bj, bi]*neta          # W per metre
	dl = np.gradient(bdist)*1000.                     # m
	tot = float(np.nansum(np.nan_to_num(fn)*dl))
	return tot, fn, blon, blat, bdist


def edge_power(Fxi, Feta, h, mask, pm, pn, lo, hi):
	"""
	Power leaving through the four domain edges within a depth band,
	positive outward. The configuration cuts the margin alongshore, so
	this is the part of the export that never crosses an isobath.
	"""
	tot = 0.0
	sel = (h >= lo) & (h < hi) & (mask > 0)
	# eta = 0 edge, outward normal along minus eta
	m = sel[0, :]
	tot += float(np.nansum(np.nan_to_num(-Feta[0, :])[m]/pm[0, :][m]))
	# eta = -1 edge, outward normal along plus eta
	m = sel[-1, :]
	tot += float(np.nansum(np.nan_to_num(Feta[-1, :])[m]/pm[-1, :][m]))
	# xi = 0 edge, outward normal along minus xi
	m = sel[:, 0]
	tot += float(np.nansum(np.nan_to_num(-Fxi[:, 0])[m]/pn[:, 0][m]))
	# xi = -1 edge, outward normal along plus xi
	m = sel[:, -1]
	tot += float(np.nansum(np.nan_to_num(Fxi[:, -1])[m]/pn[:, -1][m]))
	return tot


# =====================================================================
# MAIN
# =====================================================================
def main():
	for f in (HARMFILE, FIG03):
		if not os.path.exists(f):
			raise SystemExit(f'{f} not found, run fig03_compute.py first')

	print('opening the harmonic coefficients and the conversion')
	dh = xr.open_dataset(HARMFILE)
	d3 = xr.open_dataset(FIG03)

	lon = dh.lon_rho.values; lat = dh.lat_rho.values
	h = dh.h.values; mask = dh.mask_rho.values
	pm = dh.pm.values; pn = dh.pn.values
	ang = dh.angle.values if 'angle' in dh else np.zeros_like(h)
	ny, nx = h.shape
	wet = mask > 0
	flat = wet.ravel()
	nwet = int(flat.sum())
	nz = dh.sizes['s_rho']
	wins = [str(w) for w in dh.window.values]
	nwin = len(wins)
	nband = len(BANDS)
	niso = len(ISOBATHS)
	aw = area_weights(pm, pn, mask)
	hb = h.ravel()[flat]
	dx = 1.0/pm
	tree, jj, ii = build_tree(lon, lat, mask)
	print(f'  grid {ny} x {nx}, {nwet} wet columns')

	dsr = open_run(REALISTIC)
	dsc = open_run(CONTROL)

	def unflat(v):
		a = np.full(ny*nx, np.nan); a[flat] = v
		return a.reshape(ny, nx)

	# -----------------------------------------------------------------
	# allocation
	# -----------------------------------------------------------------
	fld = {f'{k}_{t}': np.full((nwin, ny, nx), np.nan)
		   for k in ('Fxi', 'Feta', 'Fmag') for t in RUNS}
	fldm = {f'{k}_{t}': np.full((nwin, NMODE, ny, nx), np.nan)
			for k in ('Fnxi', 'Fneta', 'ppw') for t in RUNS}

	iso = {f'{k}_{t}': np.full((nwin, niso), np.nan)
		   for k in ('P_tot', 'area_res') for t in RUNS}
	isom = {f'P_mode_{t}': np.full((nwin, niso, NMODE), np.nan)
			for t in RUNS}

	bud = {f'{k}_{t}': np.full((nwin, nband), np.nan)
		   for k in ('C', 'Cpos', 'F_out', 'F_in', 'F_edge', 'loss')
		   for t in RUNS}
	rad = {f'{k}_{t}': np.full(nwin, np.nan)
		   for k in ('E', 'Cgen', 'P_E') for t in RUNS}

	prof = {f'P_prof_{t}': np.full((nwin, niso, 400), np.nan) for t in RUNS}
	prof_lat = np.full((niso, 400), np.nan)

	# -----------------------------------------------------------------
	# window loop
	# -----------------------------------------------------------------
	for iw, wname in enumerate(wins):
		t0, t1 = WINDOWS[wname]
		for tag, ds in (('ref', dsr), ('ctrl', dsc)):
			print(f'  {wname}/{tag}')
			temp, salt, zr, zw = mean_state(ds, tag, wname, t0, t1)

			N2 = compute_N2(temp, salt, zr, zw, lon, lat)
			Nw = zw.shape[0]
			N2f = N2.reshape(Nw, -1)[:, flat]
			zwf = zw.reshape(Nw, -1)[:, flat]
			c, Phi = vertical_modes(N2f, zwf, nmodes=NMODE + 1,
									free_surface=True)
			dzf = np.diff(zwf, axis=0)
			Hf = zwf[-1] - zwf[0]

			a_u = dh[f'a_ubc_{tag}'].isel(window=iw).values
			b_u = dh[f'b_ubc_{tag}'].isel(window=iw).values
			a_v = dh[f'a_vbc_{tag}'].isel(window=iw).values
			b_v = dh[f'b_vbc_{tag}'].isel(window=iw).values
			a_p = dh[f'a_pbc_{tag}'].isel(window=iw).values
			b_p = dh[f'b_pbc_{tag}'].isel(window=iw).values
			a_u = a_u.reshape(nz, -1)[:, flat].astype(float)
			b_u = b_u.reshape(nz, -1)[:, flat].astype(float)
			a_v = a_v.reshape(nz, -1)[:, flat].astype(float)
			b_v = b_v.reshape(nz, -1)[:, flat].astype(float)
			a_p = a_p.reshape(nz, -1)[:, flat].astype(float)
			b_p = b_p.reshape(nz, -1)[:, flat].astype(float)

			# ---- depth integrated baroclinic flux, grid components
			Fxi = np.nansum(cycle_mean(a_u, b_u, a_p, b_p)*dzf, axis=0)
			Feta = np.nansum(cycle_mean(a_v, b_v, a_p, b_p)*dzf, axis=0)
			fld[f'Fxi_{tag}'][iw] = unflat(Fxi)
			fld[f'Feta_{tag}'][iw] = unflat(Feta)
			fld[f'Fmag_{tag}'][iw] = unflat(np.hypot(Fxi, Feta))

			# ---- modal flux, and the points per wavelength of each mode
			Fnx = np.full((NMODE, nwet), np.nan)
			Fny = np.full((NMODE, nwet), np.nan)
			ppw = np.full((NMODE, nwet), np.nan)
			for n in range(1, NMODE + 1):
				phin, _ = phi_from_Phi(Phi[:, n, :], zwf, c[n])
				a_un = project(a_u, phin, dzf, Hf)
				b_un = project(b_u, phin, dzf, Hf)
				a_vn = project(a_v, phin, dzf, Hf)
				b_vn = project(b_v, phin, dzf, Hf)
				a_pn = project(a_p, phin, dzf, Hf)
				b_pn = project(b_p, phin, dzf, Hf)
				Fnx[n - 1] = Hf*cycle_mean(a_un, b_un, a_pn, b_pn)
				Fny[n - 1] = Hf*cycle_mean(a_vn, b_vn, a_pn, b_pn)
				ppw[n - 1] = c[n]*T_M2/dx.ravel()[flat]
				fldm[f'Fnxi_{tag}'][iw, n - 1] = unflat(Fnx[n - 1])
				fldm[f'Fneta_{tag}'][iw, n - 1] = unflat(Fny[n - 1])
				fldm[f'ppw_{tag}'][iw, n - 1] = unflat(ppw[n - 1])

			# ---- cross isobath power, total and by mode
			Fxi2, Feta2 = unflat(Fxi), unflat(Feta)
			for k, lev in enumerate(ISOBATHS):
				tot, fn, blon, blat, bdist = isobath_power(
					Fxi2, Feta2, lon, lat, h, mask, pm, pn, lev, tree, jj, ii)
				iso[f'P_tot_{tag}'][iw, k] = tot/1e6 if np.isfinite(tot) \
					else np.nan
				if fn is not None:
					m = min(fn.size, prof[f'P_prof_{tag}'].shape[-1])
					prof[f'P_prof_{tag}'][iw, k, :m] = fn[:m]
					if np.all(np.isnan(prof_lat[k])):
						prof_lat[k, :m] = blat[:m]
					bj, bi = nearest_ji(tree, jj, ii, blon, blat)
					pk = ppw[:, :][:, np.searchsorted(
						np.flatnonzero(flat), bj*nx + bi)] \
						if False else None
					# area fraction of the contour resolving mode one
					pw = fldm[f'ppw_{tag}'][iw, 0][bj, bi]
					iso[f'area_res_{tag}'][iw, k] = float(
						np.nanmean((pw >= PPW_MIN).astype(float)))
				for n in range(NMODE):
					tn, _, _, _, _ = isobath_power(
						unflat(Fnx[n]), unflat(Fny[n]), lon, lat, h, mask,
						pm, pn, lev, tree, jj, ii)
					isom[f'P_mode_{tag}'][iw, k, n] = tn/1e6 \
						if np.isfinite(tn) else np.nan

			# ---- control volume budget by band
			C = d3[f'C_{tag}'].isel(window=iw).values
			for ib, (nm, lo, hi, short) in enumerate(BANDS):
				m = wet & (h >= lo) & (h < hi)
				w = np.where(m & np.isfinite(C), aw, 0.0)
				bud[f'C_{tag}'][iw, ib] = float(
					np.nansum(np.nan_to_num(C)*w)/1e6)
				bud[f'Cpos_{tag}'][iw, ib] = float(
					np.nansum(np.where(np.nan_to_num(C) > 0,
									   np.nan_to_num(C), 0.)*w)/1e6)
				if hi < np.nanmax(h[wet]):
					po, _, _, _, _ = isobath_power(Fxi2, Feta2, lon, lat, h,
												   mask, pm, pn, hi, tree,
												   jj, ii)
					po = po if np.isfinite(po) else 0.0
				else:
					po = 0.0        # the outer boundary is the domain edge,
					                # carried entirely by edge_power
				pi_, _, _, _, _ = isobath_power(Fxi2, Feta2, lon, lat, h,
												mask, pm, pn, lo, tree, jj, ii)
				pe = edge_power(Fxi2, Feta2, h, mask, pm, pn, lo, hi)
				bud[f'F_out_{tag}'][iw, ib] = po/1e6 if np.isfinite(po) \
					else np.nan
				bud[f'F_in_{tag}'][iw, ib] = pi_/1e6 if np.isfinite(pi_) \
					else np.nan
				bud[f'F_edge_{tag}'][iw, ib] = pe/1e6
				bud[f'loss_{tag}'][iw, ib] = (
					bud[f'C_{tag}'][iw, ib]
					- (bud[f'F_out_{tag}'][iw, ib]
					   - bud[f'F_in_{tag}'][iw, ib]
					   + bud[f'F_edge_{tag}'][iw, ib]))

			# ---- radiated fraction
			mg = wet & (h >= GEN_BAND[0]) & (h < GEN_BAND[1])
			wg = np.where(mg & np.isfinite(C) & (C > 0), aw, 0.0)
			cg = float(np.nansum(np.nan_to_num(C)*wg)/1e6)
			ke = ISOBATHS.index(E_ISOBATH) if E_ISOBATH in ISOBATHS else -1
			pe_ = iso[f'P_tot_{tag}'][iw, ke]
			rad[f'Cgen_{tag}'][iw] = cg
			rad[f'P_E_{tag}'][iw] = pe_
			rad[f'E_{tag}'][iw] = pe_/cg if cg else np.nan
			print(f'    generation {cg:7.0f} MW, crossing '
				  f'{E_ISOBATH:.0f} m {pe_:7.0f} MW, radiated fraction '
				  f'{pe_/cg if cg else np.nan:5.2f}')

	# -----------------------------------------------------------------
	# printed summary
	# -----------------------------------------------------------------
	line = '=' * 78
	print('\n' + line)
	print('CONTROL VOLUME BUDGET, all values in MW')
	print(line)
	print('  C = F_out - F_in + F_edge + loss, with every flux term a line')
	print('  integral and no horizontal derivative taken anywhere.')
	print(f'  {"window":>11} {"band":>7} {"run":>5} {"C":>9} {"C+":>9} '
		  f'{"F_out":>9} {"F_in":>9} {"F_edge":>9} {"loss":>9}')
	for ib, (nm, lo, hi, short) in enumerate(BANDS):
		print(f'\n  {nm}')
		for iw, wname in enumerate(wins):
			for tag in RUNS:
				print(f'  {wname:>11} {short:>7} {tag.upper():>5} '
					  f'{bud[f"C_{tag}"][iw, ib]:9.0f} '
					  f'{bud[f"Cpos_{tag}"][iw, ib]:9.0f} '
					  f'{bud[f"F_out_{tag}"][iw, ib]:9.0f} '
					  f'{bud[f"F_in_{tag}"][iw, ib]:9.0f} '
					  f'{bud[f"F_edge_{tag}"][iw, ib]:9.0f} '
					  f'{bud[f"loss_{tag}"][iw, ib]:9.0f}')
	print(line + '\n')

	print(line)
	print(f'CROSS ISOBATH POWER, MW, positive offshore')
	print(line)
	print(f'  {"window":>11} {"run":>5} ' +
		  ' '.join(f'{int(v):>7}' for v in ISOBATHS))
	for iw, wname in enumerate(wins):
		for tag in RUNS:
			v = iso[f'P_tot_{tag}'][iw]
			print(f'  {wname:>11} {tag.upper():>5} ' +
				  ' '.join(f'{x:7.0f}' if np.isfinite(x) else '    n/a'
						   for x in v))
	print(line + '\n')

	# -----------------------------------------------------------------
	# write
	# -----------------------------------------------------------------
	print('writing', OUTFILE)
	coords = dict(
		window=('window', wins),
		mode=('mode', np.arange(1, NMODE + 1)),
		band=('band', [b[3] for b in BANDS]),
		isobath=('isobath', np.array(ISOBATHS)),
		eta_rho=('eta_rho', np.arange(ny)),
		xi_rho=('xi_rho', np.arange(nx)),
		cpt=('cpt', np.arange(prof['P_prof_ref'].shape[-1])),
	)
	dv = {k: (('window', 'eta_rho', 'xi_rho'), v) for k, v in fld.items()}
	dv.update({k: (('window', 'mode', 'eta_rho', 'xi_rho'), v)
			   for k, v in fldm.items()})
	dv.update({f'iso_{k}': (('window', 'isobath'), v)
			   for k, v in iso.items()})
	dv.update({f'iso_{k}': (('window', 'isobath', 'mode'), v)
			   for k, v in isom.items()})
	dv.update({f'bud_{k}': (('window', 'band'), v) for k, v in bud.items()})
	dv.update({f'rad_{k}': ('window', v) for k, v in rad.items()})
	dv.update({f'prof_{k}': (('window', 'isobath', 'cpt'), v)
			   for k, v in prof.items()})
	dv.update(dict(
		prof_lat=(('isobath', 'cpt'), prof_lat),
		lon_rho=(('eta_rho', 'xi_rho'), lon),
		lat_rho=(('eta_rho', 'xi_rho'), lat),
		h=(('eta_rho', 'xi_rho'), h),
		mask_rho=(('eta_rho', 'xi_rho'), mask),
		angle=(('eta_rho', 'xi_rho'), ang),
	))

	ds_out = xr.Dataset(dv, coords=coords)
	ds_out.attrs.update(dict(
		harmonics=HARMFILE, conversion=FIG03,
		nmode=NMODE, nmode_flux=NMODE_FLUX, ppw_min=PPW_MIN,
		e_isobath=E_ISOBATH, break_isobath=BREAK_ISOBATH,
		gen_band_lo=GEN_BAND[0], gen_band_hi=GEN_BAND[1],
		bands=';'.join(f'{b[3]}:{b[0]}' for b in BANDS),
		windows=';'.join(f'{k}:{v[0]}..{v[1]}' for k, v in WINDOWS.items()),
		note='Every flux term is a line integral of F along an isobath or '
			 'along a domain edge, positive outward, so no horizontal '
			 'derivative of the flux is taken. The budget of a band reads '
			 'C = F_out - F_in + F_edge + loss, and loss aggregates '
			 'dissipation with the neglected advective and tendency terms '
			 'and with any spurious numerical mixing. Modal fluxes are '
			 'quoted for the first nmode_flux modes and iso_area_res gives '
			 'the fraction of each contour resolving mode one at ppw_min '
			 'points per wavelength.',
	))
	enc = {v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars}
	ds_out.to_netcdf(OUTFILE, encoding=enc)
	ds_out.close(); dh.close(); d3.close()

	import fig05_report
	fig05_report.main()


if __name__ == '__main__':
	main()