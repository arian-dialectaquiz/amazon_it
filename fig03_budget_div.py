#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig03_fluxbudget.py
=================================================================
Flux divergence and residual budget, resolved on the same three depth
bands used for the signed conversion decomposition, plus the onshore
baroclinic flux across the shelf break.

Reads only fig03_data.nc. Nothing is recomputed from the model output,
since the divergence and residual maps and the onshore flux profile were
all stored by fig03_compute.py.

	python fig03_fluxbudget.py

Why band resolved. The conversion was shown to behave as three different
systems, a clean single signed generator on the shelf, a seasonal
generator on the upper slope and a near cancelling interference field
beyond 1000 m. The flux budget should be reported on the same partition
so the two tables can be read against each other. The two region version
in the earlier draft split only at 250 m and therefore mixed the upper
slope with the deep zone.

Caveat carried into the text. The alongshore boundaries of the
configuration cut the shelf, so part of what is counted here as export
leaves through the domain edge rather than crossing the shelf break. The
onshore flux across the 200 m isobath, integrated along the break, is the
quantity that is free of that ambiguity and is reported alongside.
=================================================================
"""

import numpy as np
import xarray as xr

DATA = 'fig03_data.nc'

ds = xr.open_dataset(DATA)
lon = ds.lon_rho.values
lat = ds.lat_rho.values
h = ds.h.values
mask = ds.mask_rho.values
wins = [str(w) for w in ds.window.values]
SHELF_MAX = float(ds.attrs.get('shelf_max', 250.))
SLOPE_MAX = float(ds.attrs.get('slope_max', 3500.))
BREAK = float(ds.attrs.get('break_isobath', 200.))


def cell_area(lon, lat):
	coslat = np.cos(np.deg2rad(lat))
	dxi = np.gradient(lon, axis=1)*111.2e3*coslat
	dyi = np.gradient(lat, axis=1)*111.2e3
	dxe = np.gradient(lon, axis=0)*111.2e3*coslat
	dye = np.gradient(lat, axis=0)*111.2e3
	return np.hypot(dxi, dyi)*np.hypot(dxe, dye)


A = cell_area(lon, lat)

BANDS = [('shelf, h < 250 m',           0.,   SHELF_MAX),
		 ('upper slope, 250-1000 m',    SHELF_MAX, 1000.),
		 ('deep, 1000-3500 m',          1000., SLOPE_MAX),
		 ('slope all, 250-3500 m',      SHELF_MAX, SLOPE_MAX)]


def integ(field, m):
	w = np.where(m & np.isfinite(field), A, 0.0)
	return np.nansum(np.nan_to_num(field)*w)/1e6      # MW


print('=' * 84)
print('FLUX DIVERGENCE AND RESIDUAL BY DEPTH BAND, all values in MW')
print('  positive div F is net export from the band')
print('  D = C - div F is the residual, dissipation plus everything the')
print('  diagnostics cannot separate')
print('=' * 84)
print(f'{"band":>24} {"window":>11} {"run":>5} {"C":>8} {"div F":>8} '
	  f'{"D":>8} {"divF/C":>8} {"D/C":>7}')

res = {}
for bname, h0, h1 in BANDS:
	m = (h >= h0) & (h < h1) & (mask > 0)
	for k, w in enumerate(wins):
		for tg in ('ref', 'ctrl'):
			C = integ(ds[f'C_{tg}'].isel(window=k).values, m)
			F = integ(ds[f'divF_{tg}'].isel(window=k).values, m)
			D = integ(ds[f'D_{tg}'].isel(window=k).values, m)
			res[(bname, w, tg)] = (C, F, D)
			print(f'{bname:>24} {w:>11} {tg:>5} {C:8.1f} {F:8.1f} {D:8.1f} '
				  f'{100*F/C if C else np.nan:7.1f}% '
				  f'{100*D/C if C else np.nan:6.1f}%')
	print()

print('=' * 84)
print('RESIDUAL, REF against CTRL')
print('=' * 84)
for bname, _, _ in BANDS:
	line = f'{bname:>24}  '
	for w in wins:
		Dr = res[(bname, w, 'ref')][2]; Dc = res[(bname, w, 'ctrl')][2]
		line += f'{w[:4]} x{Dr/Dc if Dc else np.nan:5.2f}   '
	print(line)

print()
print('=' * 84)
print(f'ONSHORE BAROCLINIC FLUX ACROSS THE {BREAK:.0f} m ISOBATH, MW')
print('  positive is directed onto the shelf, integrated along the break')
print('=' * 84)
if 'onshore_ref' in ds:
	print(f'{"window":>11} {"REF":>10} {"CTRL":>10} {"REF-CTRL":>10}')
	for k, w in enumerate(wins):
		a = float(ds.onshore_ref.isel(window=k).values)
		b = float(ds.onshore_ctrl.isel(window=k).values)
		print(f'{w:>11} {a:10.1f} {b:10.1f} {a-b:10.1f}')
	print('\n  This is the quantity free of the domain edge ambiguity, since')
	print('  it is integrated across the shelf break itself rather than')
	print('  inferred from a divergence over a region whose lateral')
	print('  boundaries coincide with the open boundaries of the grid.')
else:
	print('  onshore_ref not present in this file')

print()
print('=' * 84)
print('CAVEAT TO CARRY INTO THE TEXT')
print('=' * 84)
print('  The alongshore boundaries of the configuration cut the shelf, so')
print('  part of the positive div F over the shelf leaves through the')
print('  domain edge rather than across the shelf break. The shelf export')
print('  fractions below are therefore an upper bound on the true offshore')
print('  transfer, and the onshore flux across the 200 m isobath is the')
print('  cleaner statistic for the shelf loading argument.')