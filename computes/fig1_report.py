#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig01_report.py
=================================================================
Figure 1  -  every number the manuscript quotes, recast from the
			 saved data file
Amazon shelf internal tide manuscript, version 2

	python fig01_report.py

Reads fig01_data.nc and writes

	fig01_report.txt      the block below, as printed
	fig01_numbers.json    a flat dictionary of every value
	fig01_numbers.tex     LaTeX macros for the Results paragraph

Nothing is recomputed from the model output, so this can be run as
often as wanted and a cleared console costs nothing.

The four sections follow the four premises the figure tests, and the
final section lists, in order, the values that fill the bracketed slots
of the first Results paragraph.

Fractions of area are area weighted using the cell geometry, which is
the physically meaningful statement. The unweighted cell count fraction
is printed beside it, since that is what the earlier console summary
reported and the two differ where the grid is refined.
=================================================================
"""

import numpy as np
import xarray as xr

from report import Report, cell_area, area_fraction, robust, crossing_depth


# =====================================================================
# CONFIGURATION
# =====================================================================
DATA = 'fig01_data.nc'
STEM = 'fig01'
WIN_MAIN = 'peak'          # the window the figure draws and the text quotes

# depth bands, matching Section 2.2 of the manuscript
BANDS = [('mid shelf, 20 to 100 m',       20.,  100.),
		 ('outer shelf, 100 to 250 m',   100.,  250.),
		 ('upper slope, 250 to 1000 m',  250., 1000.),
		 ('deep, beyond 1000 m',        1000., 6000.)]

# band over which the barotropic control is judged
BT_BAND = (20., 1000.)
# band over which conversion is anchored, quoted separately
BT_BAND2 = (100., 1000.)

ALPHA_LO, ALPHA_HI = 1.0, 1.5     # the interval quoted in the Results


def main():
	ds = xr.open_dataset(DATA)

	lon = ds.lon_rho.values
	lat = ds.lat_rho.values
	mask = ds.mask_rho.values
	h = ds.h.values
	wet = mask > 0
	area = cell_area(lon, lat)

	wins = [str(w) for w in ds.window.values]
	iw = wins.index(WIN_MAIN)
	site_names = [str(s) for s in ds.site.values]
	blat = ds.break_lat.values

	trap_r = float(ds.attrs.get('trap_ratio', 3.))
	ppw_min = float(ds.attrs.get('ppw_min', 8.))
	brk = float(ds.attrs.get('break_isobath', 200.))

	rep = Report('FIGURE 1, THE MARGIN AND THE CONTROLS ON THE COMPARISON')
	rep.text(f'  data file        {DATA}')
	rep.text(f'  record           {ds.attrs.get("record_start", "?")} to '
			 f'{ds.attrs.get("record_end", "?")}')
	rep.text(f'  output interval  {float(ds.attrs.get("dt_hours", np.nan)):.2f} h')
	rep.text(f'  grid             {h.shape[0]} x {h.shape[1]}, spacing '
			 f'{np.nanmin(ds.dx.values[wet])/1e3:.2f} to '
			 f'{np.nanmax(ds.dx.values[wet])/1e3:.2f} km')
	rep.text(f'  windows          {ds.attrs.get("windows", "?")}')
	rep.text(f'  quoted window    {WIN_MAIN}')

	# =================================================================
	# PREMISE 1, the two runs share the barotropic tide
	# =================================================================
	rep.section('PREMISE 1   the two experiments share the barotropic tide')
	rep.note('Conversion is the work of the barotropic flow against the')
	rep.note('baroclinic pressure at the bed, so a small difference here is')
	rep.note('what assigns the conversion anomaly to the stratification.')
	rep.text('')

	fields = [('dzeta_rel', 'M2 elevation amplitude', '%', 'dzeta'),
			  ('dubt_rel', 'M2 depth averaged current, semi major axis', '%',
			   'dubt'),
			  ('dzeta_pha', 'M2 elevation phase', 'deg', 'dpha')]

	for lo, hi, tag in ((BT_BAND[0], BT_BAND[1], 'b1'),
						(BT_BAND2[0], BT_BAND2[1], 'b2')):
		sel = wet & (h > lo) & (h < hi)
		rep.text(f'  over {lo:.0f} to {hi:.0f} m')
		for var, nm, unit, key in fields:
			if var not in ds:
				continue
			med, p95 = robust(ds[var].values[sel])
			rep.val(f'bt.{key}.median.{tag}', med, '{:.2f}', unit,
					f'  median |REF-CTRL| of the {nm}')
			rep.val(f'bt.{key}.p95.{tag}', p95, '{:.2f}', unit,
					'  95th percentile of the same')
		rep.text('')

	for var, nm, unit, key in (('dzeta_rel_break', 'M2 elevation amplitude',
								'%', 'dzeta'),
							   ('dubt_rel_break',
								'M2 depth averaged current', '%', 'dubt')):
		if var in ds:
			med, p95 = robust(ds[var].values)
			rep.val(f'bt.{key}.median.break', med, '{:.2f}', unit,
					f'along the {brk:.0f} m isobath, median |REF-CTRL| of '
					f'the {nm}')
			rep.val(f'bt.{key}.p95.break', p95, '{:.2f}', unit,
					'  95th percentile of the same')

	for tag, var in (('ref', 'zeta_amp_ref'), ('ctrl', 'zeta_amp_ctrl')):
		if var in ds:
			sel = wet & (h > BT_BAND[0]) & (h < BT_BAND[1])
			rep.val(f'bt.zeta_amp.{tag}', np.nanmedian(ds[var].values[sel]),
					'{:.2f}', 'm',
					f'median M2 elevation amplitude, {tag.upper()}')
	for tag, var in (('ref', 'ubt_maj_ref'), ('ctrl', 'ubt_maj_ctrl')):
		if var in ds and np.isfinite(ds[var].values).any():
			sel = wet & (h > BT_BAND[0]) & (h < BT_BAND[1])
			rep.val(f'bt.ubt_maj.{tag}', np.nanmedian(ds[var].values[sel]),
					'{:.3f}', 'm s-1',
					f'median M2 semi major axis of u_bt, {tag.upper()}')

	# =================================================================
	# PREMISE 2, the lens is surface trapped where alpha is evaluated
	# =================================================================
	rep.section('PREMISE 2   the lens sits above the near bottom '
				'stratification')
	rep.note(f'The criticality argument is applied only where h > {trap_r:.0f} hp.')
	rep.text('')
	rep.row(['window', 'band', 'h>3hp area', 'by count', 'median h/hp',
			 'median hp'], [13, 30, 12, 10, 13, 10])
	rep.rule()

	for k, w in enumerate(wins):
		hr = ds.hratio_ref.isel(window=k).values
		hp = ds.hp_ref.isel(window=k).values
		fl = ds.trap_ok_ref.isel(window=k).values
		for nm, lo, hi in [(b[0], b[1], b[2]) for b in BANDS[:3]]:
			sel = wet & (h >= lo) & (h < hi)
			fa = area_fraction(fl, area, sel)
			fc = float(np.nanmean(fl[sel])) if sel.any() else np.nan
			mhr = float(np.nanmedian(hr[sel])) if sel.any() else np.nan
			mhp = float(np.nanmedian(hp[sel])) if sel.any() else np.nan
			short = nm.split(',')[0].replace(' ', '_')
			rep.values[f'trap.{w}.{short}.area_frac'] = fa
			rep.values[f'trap.{w}.{short}.median_hratio'] = mhr
			rep.values[f'trap.{w}.{short}.median_hp'] = mhp
			rep.row([w, nm, f'{100*fa:.1f} %', f'{100*fc:.1f} %',
					 f'{mhr:.1f}', f'{mhp:.1f} m'],
					[13, 30, 12, 10, 13, 10])
		rep.rule()

	# =================================================================
	# PREMISE 3, criticality is common to the two runs
	# =================================================================
	rep.section(f'PREMISE 3   criticality along the {brk:.0f} m isobath')
	rep.text('')

	for k, w in enumerate(wins):
		ar = ds.alpha_ref_break.isel(window=k).values
		ac = ds.alpha_ctrl_break.isel(window=k).values
		da = ds.dalpha_break.isel(window=k).values
		dn = ds.dn2bot_rel_break.isel(window=k).values

		med_da, p95_da = robust(da)
		sd = float(np.nanstd(ar))
		inband = np.nanmean(((ar >= ALPHA_LO) & (ar <= ALPHA_HI)).astype(float))
		sub = np.nanmean((ar < 1.).astype(float))
		flip = np.nanmean(((ar > 1.) != (ac > 1.)).astype(float))
		med_dn, _ = robust(dn)

		rep.text(f'  {w}')
		rep.val(f'alpha.{w}.median_ref', np.nanmedian(ar), '{:.2f}', '',
				'  median alpha, REF')
		rep.val(f'alpha.{w}.median_ctrl', np.nanmedian(ac), '{:.2f}', '',
				'  median alpha, CTRL')
		rep.val(f'alpha.{w}.p10_ref', np.nanpercentile(ar, 10), '{:.2f}', '',
				'  10th percentile of alpha, REF')
		rep.val(f'alpha.{w}.p90_ref', np.nanpercentile(ar, 90), '{:.2f}', '',
				'  90th percentile of alpha, REF')
		rep.val(f'alpha.{w}.frac_in_band', 100*inband, '{:.1f}', '%',
				f'  fraction of the break with {ALPHA_LO} < alpha < {ALPHA_HI}')
		rep.val(f'alpha.{w}.frac_subcritical', 100*sub, '{:.1f}', '%',
				'  fraction of the break subcritical, REF')
		rep.val(f'alpha.{w}.dalpha_median', med_da, '{:.3f}', '',
				'  median |REF-CTRL| of alpha')
		rep.val(f'alpha.{w}.dalpha_p95', p95_da, '{:.3f}', '',
				'  95th percentile of the same')
		rep.val(f'alpha.{w}.sd_along_break', sd, '{:.2f}', '',
				'  along break standard deviation of alpha, REF')
		rep.val(f'alpha.{w}.ratio_diff_to_sd', med_da/sd if sd else np.nan,
				'{:.3f}', '', '  median difference over that spread')
		rep.val(f'alpha.{w}.frac_flip', 100*flip, '{:.1f}', '%',
				'  fraction of the break where the runs straddle alpha = 1')
		rep.val(f'alpha.{w}.dn2bot_median', med_dn, '{:.1f}', '%',
				'  median |REF-CTRL| of the near bottom N2')
		rep.text('')

	rep.text('  criticality at the generation sites, REF and CTRL')
	rep.rule()
	for n, name in enumerate(site_names):
		if not bool(ds.site_inside.values[n]):
			continue
		gla = float(ds.site_lat.values[n])
		j = int(np.nanargmin(np.abs(blat - gla)))
		cells = [name]
		for k, w in enumerate(wins):
			a_r = float(ds.alpha_ref_break.isel(window=k).values[j])
			a_c = float(ds.alpha_ctrl_break.isel(window=k).values[j])
			rep.values[f'alpha.site.{name}.{w}.ref'] = a_r
			rep.values[f'alpha.site.{name}.{w}.ctrl'] = a_c
			cells.append(f'{w[:4]} {a_r:.2f} / {a_c:.2f}')
		rep.row(cells, [6, 22, 22, 22])

	# =================================================================
	# PREMISE 4, the subdomain in which the runs are compared
	# =================================================================
	rep.section('PREMISE 4   the subdomain resolved in both experiments')
	rep.note(f'Quantitative cross run comparisons are restricted to the '
			 f'columns where both')
	rep.note(f'clear {ppw_min:.0f} points per mode one wavelength.')
	rep.text('')
	rep.row(['window', 'band', 'REF', 'CTRL', 'BOTH', 'BOTH by count'],
			[13, 30, 9, 9, 9, 14])
	rep.rule()

	for k, w in enumerate(wins):
		for nm, lo, hi in [(b[0], b[1], b[2]) for b in BANDS]:
			sel = wet & (h >= lo) & (h < hi)
			short = nm.split(',')[0].replace(' ', '_')
			fr = area_fraction(ds.res_ref.isel(window=k).values, area, sel)
			fc = area_fraction(ds.res_ctrl.isel(window=k).values, area, sel)
			fb = area_fraction(ds.res_both.isel(window=k).values, area, sel)
			fbc = float(np.nanmean(ds.res_both.isel(window=k).values[sel])) \
				if sel.any() else np.nan
			rep.values[f'res.{w}.{short}.ref'] = fr
			rep.values[f'res.{w}.{short}.ctrl'] = fc
			rep.values[f'res.{w}.{short}.both'] = fb
			rep.row([w, nm, f'{100*fr:.1f}%', f'{100*fc:.1f}%',
					 f'{100*fb:.1f}%', f'{100*fbc:.1f}%'],
					[13, 30, 9, 9, 9, 14])
		rep.rule()

	dbin = ds.depth_bin.values
	for k, w in enumerate(wins):
		fb = ds.frac_res_both.isel(window=k).values
		for lev in (0.5, 0.9):
			d = crossing_depth(dbin, fb, lev)
			rep.val(f'res.{w}.depth_at_{int(100*lev)}pc', d, '{:.0f}', 'm',
					f'{w}, depth at which {int(100*lev)} % of cells are '
					f'resolved in both')

	# =================================================================
	# the values that fill the Results paragraph
	# =================================================================
	rep.section('VALUES FOR THE FIRST RESULTS PARAGRAPH, in order')
	w = WIN_MAIN
	V = rep.values

	def g(k):
		v = V.get(k)
		return np.nan if v is None else v

	pairs = [
		('FigOneDZetaMed', g('bt.dzeta.median.b1'), '{:.2f}',
		 'median |d| of the M2 elevation amplitude, 20 to 1000 m, per cent'),
		('FigOneDUbtMed', g('bt.dubt.median.b1'), '{:.2f}',
		 'median |d| of the M2 depth averaged current, per cent'),
		('FigOneDZetaPct', g('bt.dzeta.p95.b1'), '{:.2f}',
		 '95th percentile of the elevation difference, per cent'),
		('FigOneDUbtPct', g('bt.dubt.p95.b1'), '{:.2f}',
		 '95th percentile of the current difference, per cent'),
		('FigOneDPhaMed', g('bt.dpha.median.b1'), '{:.2f}',
		 'median |d| of the M2 elevation phase, degrees'),
		('FigOneTrapOuter', 100*g(f'trap.{w}.outer_shelf.area_frac'), '{:.0f}',
		 'outer shelf area with h > 3 hp, per cent'),
		('FigOneTrapSlope', 100*g(f'trap.{w}.upper_slope.area_frac'), '{:.0f}',
		 'upper slope area with h > 3 hp, per cent'),
		('FigOneTrapMid', 100*g(f'trap.{w}.mid_shelf.area_frac'), '{:.0f}',
		 'mid shelf area with h > 3 hp, per cent'),
		('FigOneAlphaA', g(f'alpha.site.A.{w}.ref'), '{:.2f}',
		 'alpha at site A, REF'),
		('FigOneAlphaB', g(f'alpha.site.B.{w}.ref'), '{:.2f}',
		 'alpha at site B, REF'),
		('FigOneDAlphaMed', g(f'alpha.{w}.dalpha_median'), '{:.3f}',
		 'median |d alpha|'),
		('FigOneAlphaSd', g(f'alpha.{w}.sd_along_break'), '{:.2f}',
		 'along break standard deviation of alpha'),
		('FigOneFlip', g(f'alpha.{w}.frac_flip'), '{:.1f}',
		 'fraction of the break straddling alpha = 1, per cent'),
		('FigOneDNbot', g(f'alpha.{w}.dn2bot_median'), '{:.1f}',
		 'median |d| of the near bottom N2 at the break, per cent'),
		('FigOneResOuter', 100*g(f'res.{w}.outer_shelf.both'), '{:.0f}',
		 'outer shelf resolved in both, per cent'),
		('FigOneResMid', 100*g(f'res.{w}.mid_shelf.both'), '{:.0f}',
		 'mid shelf resolved in both, per cent'),
	]
	for name, value, fmt, note in pairs:
		rep.macro(name, value, fmt)
		shown = fmt.format(value) if np.isfinite(value) else 'n/a'
		rep.text(f'  \\{name:<18s} {shown:>8s}    {note}')

	rep.save(STEM)
	ds.close()


if __name__ == '__main__':
	main()