#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig02_report.py
=================================================================
Figure 2  -  every number the manuscript quotes, recast from the
			 saved data file
Amazon shelf internal tide manuscript, version 3

	python fig02_report.py

Reads fig02_data.nc and writes

	fig02_report.txt      the block below, as printed
	fig02_numbers.json    a flat dictionary of every value
	fig02_numbers.tex     LaTeX macros for the Results paragraph

Called automatically at the end of fig02_compute.py, and safe to run
again on its own at any time.

The four sections follow the chain the figure draws. The lens sets the
phase speed, the phase speed and the lens set the vertical structure,
the vertical structure sets the bottom amplitude through which the
topography forces mode one, and the near bottom stratification is
reported last because it is the part of the column a surface trapped
lens is not supposed to reach.
=================================================================
"""

import numpy as np
import xarray as xr

from report import Report, cell_area, area_fraction, robust


DATA = 'fig02_data.nc'
STEM = 'fig02'
WIN_MAIN = 'peak'

BANDS = [('mid shelf, 20 to 100 m',      20.,  100., 'mid_shelf'),
		 ('outer shelf, 100 to 250 m',  100.,  250., 'outer_shelf'),
		 ('upper slope, 250 to 1000 m', 250., 1000., 'upper_slope')]


def main():
	ds = xr.open_dataset(DATA)

	lon = ds.lon_rho.values
	lat = ds.lat_rho.values
	mask = ds.mask_rho.values
	h = ds.h.values
	wet = mask > 0
	area = cell_area(lon, lat)
	wins = [str(w) for w in ds.window.values]
	stations = [str(s) for s in ds.station.values]
	c1_min = float(ds.attrs.get('c1_min', 0.05))
	s_lens = float(ds.attrs.get('s_lens', 35.))
	brk = float(ds.attrs.get('break_isobath', 200.))

	rep = Report('FIGURE 2, THE MEDIUM THE DISCHARGE BUILDS')
	rep.text(f'  data file      {DATA}')
	rep.text(f'  lens isohaline {s_lens:.0f}')
	rep.text(f'  windows        {ds.attrs.get("windows", "?")}')
	rep.text(f'  quoted window  {WIN_MAIN}')
	rep.text(f'  moorings       ' +
			 ', '.join(f'{s} at {float(ds.station_h.values[k]):.0f} m'
					   for k, s in enumerate(stations)))

	# =================================================================
	# A, the phase speed
	# =================================================================
	rep.section('A   MODE ONE PHASE SPEED, the waveguide the lens builds')
	rep.row(['window', 'band', 'REF', 'CTRL', 'ratio', 'no mode CTRL'],
			[13, 30, 10, 10, 8, 14])
	rep.rule()
	for k, w in enumerate(wins):
		cr = ds.c1_ref.isel(window=k).values
		cc = ds.c1_ctrl.isel(window=k).values
		for nm, lo, hi, short in BANDS:
			sel = wet & (h >= lo) & (h < hi)
			both = sel & np.isfinite(cr) & np.isfinite(cc) \
				   & (cr >= c1_min) & (cc >= c1_min)
			mr = float(np.nanmedian(cr[both])) if both.any() else np.nan
			mc = float(np.nanmedian(cc[both])) if both.any() else np.nan
			lost = sel & (cr >= c1_min) & ((cc < c1_min) | ~np.isfinite(cc))
			fl = area_fraction(lost.astype(float), area, sel)
			rep.values[f'c1.{w}.{short}.ref'] = mr
			rep.values[f'c1.{w}.{short}.ctrl'] = mc
			rep.values[f'c1.{w}.{short}.ratio'] = mr/mc if mc else np.nan
			rep.values[f'c1.{w}.{short}.no_mode_ctrl'] = fl
			rep.row([w, nm, f'{mr:.2f}', f'{mc:.2f}',
					 f'{mr/mc:.2f}' if mc else 'n/a', f'{100*fl:.1f} %'],
					[13, 30, 10, 10, 8, 14])
		rep.rule()
	rep.note('Phase speeds in m s-1, medians over the cells where both')
	rep.note('experiments support a resolvable mode. The last column is the')
	rep.note('area over which CTRL supports none while REF does.')

	k = wins.index(WIN_MAIN)
	dc = ds.c1_ref.isel(window=k).values - ds.c1_ctrl.isel(window=k).values
	for nm, lo, hi, short in BANDS:
		sel = wet & (h >= lo) & (h < hi)
		rep.val(f'dc1.{WIN_MAIN}.{short}.median', np.nanmedian(dc[sel]),
				'{:+.3f}', 'm s-1',
				f'{WIN_MAIN}, median REF minus CTRL of c1, {nm}')
	rep.val(f'dc1.{WIN_MAIN}.max', np.nanmax(dc[wet]), '{:+.2f}', 'm s-1',
			f'{WIN_MAIN}, largest positive difference in c1')

	# =================================================================
	# B, the lens
	# =================================================================
	rep.section(f'B   THE FRESHWATER LENS, base taken at the {s_lens:.0f} '
				'isohaline')
	rep.row(['window', 'band', 'hp REF', 'hp CTRL', 'zN2max REF', 'no lens'],
			[13, 30, 10, 10, 12, 10])
	rep.rule()
	for k, w in enumerate(wins):
		for nm, lo, hi, short in BANDS:
			sel = wet & (h >= lo) & (h < hi)
			hp_r = ds.hp_ref.isel(window=k).values
			hp_c = ds.hp_ctrl.isel(window=k).values
			zn = ds.zn2max_ref.isel(window=k).values
			nol = ds.nolens_ref.isel(window=k).values
			mr = float(np.nanmedian(hp_r[sel])) if sel.any() else np.nan
			mc = float(np.nanmedian(hp_c[sel])) if sel.any() else np.nan
			mz = float(np.nanmedian(zn[sel])) if sel.any() else np.nan
			fn = area_fraction(nol.astype(float), area, sel)
			rep.values[f'hp.{w}.{short}.ref'] = mr
			rep.values[f'hp.{w}.{short}.ctrl'] = mc
			rep.values[f'zn2max.{w}.{short}.ref'] = mz
			rep.values[f'nolens.{w}.{short}.ref'] = fn
			rep.row([w, nm, f'{mr:.1f} m', f'{mc:.1f} m', f'{mz:.1f} m',
					 f'{100*fn:.1f} %'], [13, 30, 10, 10, 12, 10])
		rep.rule()
	rep.note('hp is the depth at which salinity first reaches the lens')
	rep.note('isohaline and zN2max the depth of the maximum of N2. The gap')
	rep.note('between them is the reason the two must not be used')
	rep.note('interchangeably in the text.')

	# =================================================================
	# C, trapping and the bottom amplitude
	# =================================================================
	rep.section('C   SURFACE TRAPPING AND THE BOTTOM AMPLITUDE OF MODE ONE')
	rep.row(['window', 'band', 'T1 REF', 'T1 CTRL', '|phi1(-H)| R',
			 '|phi1(-H)| C', 'ratio'], [13, 28, 9, 10, 14, 14, 8])
	rep.rule()
	for k, w in enumerate(wins):
		tr = ds.trap1_ref.isel(window=k).values
		tc = ds.trap1_ctrl.isel(window=k).values
		pr = ds.phib1_ref.isel(window=k).values
		pc = ds.phib1_ctrl.isel(window=k).values
		for nm, lo, hi, short in BANDS:
			sel = wet & (h >= lo) & (h < hi)
			a = float(np.nanmedian(tr[sel])) if sel.any() else np.nan
			b = float(np.nanmedian(tc[sel])) if sel.any() else np.nan
			p = float(np.nanmedian(pr[sel])) if sel.any() else np.nan
			q = float(np.nanmedian(pc[sel])) if sel.any() else np.nan
			rep.values[f'trap.{w}.{short}.ref'] = a
			rep.values[f'trap.{w}.{short}.ctrl'] = b
			rep.values[f'phib.{w}.{short}.ref'] = p
			rep.values[f'phib.{w}.{short}.ctrl'] = q
			rep.values[f'phib.{w}.{short}.ratio'] = p/q if q else np.nan
			rep.row([w, nm, f'{a:.2f}', f'{b:.2f}', f'{p:.3f}', f'{q:.3f}',
					 f'{p/q:.2f}' if q else 'n/a'],
					[13, 28, 9, 10, 14, 14, 8])
		rep.rule()
	rep.note('|phi1(-H)| is dimensionless under the unit depth mean square')
	rep.note('normalisation, and it is the amplitude through which the')
	rep.note('topography forces mode one. A ratio below one means the plume')
	rep.note('has weakened the forcing of the gravest mode.')

	rep.text('')
	rep.text('  regression of |phi1(-H)| on the trapping index, '
			 f'{ds.attrs.get("h_min_stats", 50):.0f} to '
			 f'{ds.attrs.get("h_max_stats", 1000):.0f} m')
	rep.rule()
	for k, w in enumerate(wins):
		for t in ('ref', 'ctrl'):
			a = float(ds[f'fit_slope_{t}'].isel(window=k).values)
			r = float(ds[f'fit_r_{t}'].isel(window=k).values)
			rep.values[f'fit.{w}.{t}.slope'] = a
			rep.values[f'fit.{w}.{t}.r'] = r
			rep.row([w, t.upper(), f'slope {a:+.3f}', f'r {r:+.2f}'],
					[13, 8, 16, 12])

	rep.text('')
	rep.text('  at the moorings, peak window')
	rep.rule()
	kk = wins.index(WIN_MAIN)
	for n, s in enumerate(stations):
		row = [s, f'{float(ds.station_h.values[n]):.0f} m']
		for t in ('ref', 'ctrl'):
			c1 = float(ds[f'prof_c1_{t}'].isel(station=n, window=kk).values)
			hp = float(ds[f'prof_hp_{t}'].isel(station=n, window=kk).values)
			t1 = float(ds[f'prof_trap1_{t}'].isel(station=n, window=kk).values)
			rep.values[f'mooring.{s}.{t}.c1'] = c1
			rep.values[f'mooring.{s}.{t}.hp'] = hp
			rep.values[f'mooring.{s}.{t}.trap1'] = t1
			row.append(f'{t.upper()} c1 {c1:.2f}, hp {hp:.0f} m, T1 {t1:.2f}')
		rep.row(row, [6, 8, 34, 34])

	# =================================================================
	# D, the near bottom stratification
	# =================================================================
	rep.section(f'D   NEAR BOTTOM STRATIFICATION ALONG THE {brk:.0f} m '
				'ISOBATH')
	rep.rule()
	for k, w in enumerate(wins):
		nr = ds.nb2_ref_break.isel(window=k).values
		nc = ds.nb2_ctrl_break.isel(window=k).values
		mr = float(np.nanmedian(nr)); mc = float(np.nanmedian(nc))
		rel, _ = robust(100*(nr - nc)/np.where(nr > 0, nr, np.nan))
		rep.values[f'nb2.{w}.break.ref'] = mr
		rep.values[f'nb2.{w}.break.ctrl'] = mc
		rep.values[f'nb2.{w}.break.ratio'] = mr/mc if mc else np.nan
		rep.values[f'nb2.{w}.break.median_abs_rel'] = rel
		rep.row([w, f'REF {mr:.3e}', f'CTRL {mc:.3e}',
				 f'ratio {mr/mc:.2f}' if mc else 'n/a',
				 f'median |d| {rel:.1f} %'], [13, 18, 19, 14, 18])
	rep.note('Values in s-2. A ratio away from one is the discharge reaching')
	rep.note('the part of the column a surface trapped lens is not supposed')
	rep.note('to touch, and it is carried through the alignment term of')
	rep.note('Equation (factorise) in Figure 3.')

	for k, w in enumerate(wins):
		for t in ('ref', 'ctrl'):
			v = ds[f'phib1_{t}_break'].isel(window=k).values
			rep.val(f'phib.{w}.break.{t}', np.nanmedian(v), '{:.3f}', '',
					f'{w}, median |phi1(-H)| along the break, {t.upper()}')

	# =================================================================
	# macros
	# =================================================================
	rep.section('VALUES FOR THE FIGURE 2 RESULTS PARAGRAPH')
	w = WIN_MAIN
	V = rep.values

	def g(key):
		v = V.get(key)
		return np.nan if v is None else v

	pairs = [
		('FigTwoConeOuterRef', g(f'c1.{w}.outer_shelf.ref'), '{:.2f}',
		 'outer shelf mode one phase speed, REF, m s-1'),
		('FigTwoConeOuterCtrl', g(f'c1.{w}.outer_shelf.ctrl'), '{:.2f}',
		 'the same in CTRL'),
		('FigTwoConeRatio', g(f'c1.{w}.outer_shelf.ratio'), '{:.2f}',
		 'ratio of the two'),
		('FigTwoHpOuter', g(f'hp.{w}.outer_shelf.ref'), '{:.0f}',
		 'outer shelf plume base, REF, m'),
		('FigTwoZnOuter', g(f'zn2max.{w}.outer_shelf.ref'), '{:.0f}',
		 'outer shelf pycnocline core, REF, m'),
		('FigTwoTrapOuterRef', g(f'trap.{w}.outer_shelf.ref'), '{:.2f}',
		 'outer shelf surface trapping index, REF'),
		('FigTwoTrapOuterCtrl', g(f'trap.{w}.outer_shelf.ctrl'), '{:.2f}',
		 'the same in CTRL'),
		('FigTwoPhibOuterRef', g(f'phib.{w}.outer_shelf.ref'), '{:.2f}',
		 'outer shelf |phi1(-H)|, REF'),
		('FigTwoPhibOuterCtrl', g(f'phib.{w}.outer_shelf.ctrl'), '{:.2f}',
		 'the same in CTRL'),
		('FigTwoPhibRatio', g(f'phib.{w}.outer_shelf.ratio'), '{:.2f}',
		 'ratio of the two'),
		('FigTwoPhibSlope', g(f'fit.{w}.ref.slope'), '{:+.2f}',
		 'slope of |phi1(-H)| on the trapping index, REF'),
		('FigTwoPhibCorr', g(f'fit.{w}.ref.r'), '{:+.2f}',
		 'correlation of the same'),
		('FigTwoNbRatio', g(f'nb2.{w}.break.ratio'), '{:.2f}',
		 'ratio of the near bottom N2 at the break'),
		('FigTwoNbRel', g(f'nb2.{w}.break.median_abs_rel'), '{:.1f}',
		 'median absolute relative difference of the same, per cent'),
	]
	for name, value, fmt, note in pairs:
		rep.macro(name, value, fmt)
		shown = fmt.format(value) if np.isfinite(value) else 'n/a'
		rep.text(f'  \\{name:<22s} {shown:>8s}    {note}')

	rep.save(STEM)
	ds.close()


if __name__ == '__main__':
	main()