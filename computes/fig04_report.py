#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig04_report.py
=================================================================
Figure 4  -  every number the manuscript quotes, recast from the
			 saved data file
Amazon shelf internal tide manuscript, version 3

	python fig04_report.py

Reads fig04_data.nc and writes fig04_report.txt, fig04_numbers.json and
fig04_numbers.tex. Called at the end of fig04_compute.py and safe to run
again at any time.

Sections follow the chain. The bottom value of each structure function
comes first, since that is the quantity through which the topography
forces each mode, then the conversion into each mode and its share, then
the recovered fraction that says how much of the transfer the retained
modes describe, then the relation between the mode one share and the
surface trapping index.

Macros are written for the quoted window and again for every window,
since the Results paragraph quotes most quantities as a seasonal triple.
=================================================================
"""

import numpy as np
import xarray as xr

from report import Report


DATA = 'fig04_data.nc'
STEM = 'fig04'
WIN_MAIN = 'peak'

WSUF = {'rising': 'Ris', 'peak': 'Peak', 'decreasing': 'Dec'}


def main():
	ds = xr.open_dataset(DATA)

	wins = [str(w) for w in ds.window.values]
	bands = [str(b) for b in ds.band.values]
	blabel = dict(x.split(':', 1) for x in
				  ds.attrs.get('bands', '').split(';') if ':' in x)
	modes = [int(m) for m in ds.mode.values]
	nkeep = int(ds.attrs.get('nmode_keep', 2))
	smin = float(ds.attrs.get('sigma_min', 0.5))
	hmin = float(ds.attrs.get('h_min', 250.))
	hmax = float(ds.attrs.get('h_max', 3500.))

	mask = ds.mask_rho.values
	h = ds.h.values
	wet = mask > 0

	rep = Report('FIGURE 4, THE MODAL PARTITION OF THE GENERATED WAVE')
	rep.text(f'  data file        {DATA}')
	rep.text(f'  harmonics from   {ds.attrs.get("harmonics", "?")}')
	rep.text(f'  depth range      {hmin:.0f} to {hmax:.0f} m')
	rep.text(f'  modes solved     {modes[0]} to {modes[-1]}, '
			 f'{nkeep} retained in the partition')
	rep.text(f'  Sigma threshold  {smin:.2f}')
	rep.text(f'  quoted window    {WIN_MAIN}')

	# =================================================================
	# A, the bottom value of the structure functions
	# =================================================================
	rep.section('A   |phi_n(-H)|, THE AMPLITUDE THAT FORCES EACH MODE')
	rep.note('Dimensionless under the unit depth mean square')
	rep.note('normalisation. A ratio below one means the discharge has')
	rep.note('weakened the forcing of that mode.')
	rep.text('')
	rep.row(['window', 'band', 'mode', 'REF', 'CTRL', 'ratio'],
			[12, 10, 7, 10, 10, 9])
	rep.rule()
	for k, w in enumerate(wins):
		for ib, b in enumerate(bands):
			lo, hi = (hmin, 1000.) if b == 'slope' else (1000., hmax)
			m = wet & (h >= lo) & (h < hi)
			for n in range(len(modes)):
				vr = np.abs(ds.phib_ref.isel(window=k, mode=n).values)
				vc = np.abs(ds.phib_ctrl.isel(window=k, mode=n).values)
				a = float(np.nanmedian(vr[m])) if m.any() else np.nan
				c = float(np.nanmedian(vc[m])) if m.any() else np.nan
				rep.values[f'phib.{w}.{b}.m{modes[n]}.ref'] = a
				rep.values[f'phib.{w}.{b}.m{modes[n]}.ctrl'] = c
				rep.values[f'phib.{w}.{b}.m{modes[n]}.ratio'] = \
					a/c if c else np.nan
				rep.row([w, b, modes[n], f'{a:.3f}', f'{c:.3f}',
						 f'{a/c:.2f}' if c else 'n/a'],
						[12, 10, 7, 10, 10, 9])
		rep.rule()

	# =================================================================
	# B, the modal conversion
	# =================================================================
	rep.section('B   MODAL CONVERSION BY BAND, all values in MW')
	rep.row(['window', 'band', 'run', 'C total'] +
			[f'C{m}' for m in modes] + ['Sigma'],
			[12, 10, 7, 10] + [9]*len(modes) + [9])
	rep.rule()
	for k, w in enumerate(wins):
		for ib, b in enumerate(bands):
			for tag in ('ref', 'ctrl'):
				Ct = float(ds[f'int_C_{tag}'][k, ib])
				Sg = float(ds[f'int_Sigma_{tag}'][k, ib])
				row = [w, b, tag.upper(), f'{Ct:.0f}']
				rep.values[f'C.{w}.{b}.{tag}'] = Ct
				rep.values[f'Sigma.{w}.{b}.{tag}'] = Sg
				for n in range(len(modes)):
					v = float(ds[f'int_Cn_{tag}'][k, ib, n])
					rep.values[f'Cn.{w}.{b}.m{modes[n]}.{tag}'] = v
					row.append(f'{v:.0f}')
				row.append(f'{Sg:.2f}')
				rep.row(row, [12, 10, 7, 10] + [9]*len(modes) + [9])
		rep.rule()
	rep.note('Sigma is the sum of the retained C_n over the full C. The')
	rep.note('remainder sits in modes the grid does not carry.')

	rep.text('')
	rep.text('  modal shares and the ratio between the runs')
	rep.rule()
	rep.row(['window', 'band', 'mode', 'Pi REF', 'Pi CTRL', 'd Pi',
			 'C_n ratio', 'area used'], [12, 10, 7, 10, 10, 9, 11, 11])
	rep.rule()
	for k, w in enumerate(wins):
		for ib, b in enumerate(bands):
			ar = float(ds[f'int_area_ok_ref'][k, ib])
			for n in range(nkeep):
				pr = float(ds['int_Pin_ref'][k, ib, n])
				pc = float(ds['int_Pin_ctrl'][k, ib, n])
				cr = float(ds['int_Cn_ref'][k, ib, n])
				cc = float(ds['int_Cn_ctrl'][k, ib, n])
				rep.values[f'Pi.{w}.{b}.m{modes[n]}.ref'] = pr
				rep.values[f'Pi.{w}.{b}.m{modes[n]}.ctrl'] = pc
				rep.values[f'Pi.{w}.{b}.m{modes[n]}.diff'] = pr - pc
				rep.values[f'Cn.ratio.{w}.{b}.m{modes[n]}'] = \
					cr/cc if cc else np.nan
				rep.values[f'area_ok.{w}.{b}'] = ar
				rep.row([w, b, modes[n], f'{pr:.3f}', f'{pc:.3f}',
						 f'{pr-pc:+.3f}', f'{cr/cc:.2f}' if cc else 'n/a',
						 f'{100*ar:.0f} %'],
						[12, 10, 7, 10, 10, 9, 11, 11])
		rep.rule()
	rep.note(f'The area used is the share of the band on which Sigma')
	rep.note(f'exceeds {smin:.2f}, which is where the partition is quoted.')

	# =================================================================
	# C, how much of the transfer the retained modes describe
	# =================================================================
	rep.section('C   RECOVERED FRACTIONS')
	for k, w in enumerate(wins):
		for tag in ('ref', 'ctrl'):
			m = wet & (h >= hmin) & (h <= hmax)
			sg = ds[f'Sigma_{tag}'].isel(window=k).values
			vf = ds[f'varfrac_{tag}'].isel(window=k).values
			a = float(np.nanmedian(sg[m])); b = float(np.nanmedian(vf[m]))
			rep.values[f'Sigma.median.{w}.{tag}'] = a
			rep.values[f'varfrac.median.{w}.{tag}'] = b
			rep.row([w, tag.upper(), f'median Sigma {a:.2f}',
					 f'median pressure variance recovered {b:.2f}'],
					[12, 7, 22, 44])
	rep.note('The second column is the share of the depth mean square of')
	rep.note('the baroclinic pressure carried by the retained modes, which')
	rep.note('is a resolved fraction free of the sign changes that make')
	rep.note('Sigma noisy where the conversion is weak.')

	# =================================================================
	# D, the partition against the trapping index
	# =================================================================
	rep.section('D   THE MODE ONE SHARE AGAINST THE SURFACE TRAPPING INDEX')
	rep.rule()
	for k, w in enumerate(wins):
		for tag in ('ref', 'ctrl'):
			s = float(ds[f'fit_slope_{tag}'][k])
			r = float(ds[f'fit_rank_{tag}'][k])
			rep.values[f'fit.{w}.{tag}.slope'] = s
			rep.values[f'fit.{w}.{tag}.rank'] = r
			rep.row([w, tag.upper(), f'slope {s:+.3f}', f'rank {r:+.2f}'],
					[12, 7, 18, 14])
	rep.note('A negative slope means the trapped columns force mode one')
	rep.note('less strongly and give a larger share to mode two.')

	# =================================================================
	# macros
	# =================================================================
	rep.section('VALUES FOR THE FIGURE 4 RESULTS PARAGRAPH')
	V = rep.values

	def g(key):
		v = V.get(key)
		return np.nan if v is None else v

	def emit(name, value, fmt, note):
		rep.macro(name, value, fmt)
		shown = fmt.format(value) if np.isfinite(value) else 'n/a'
		rep.text(f'  \\{name:<28s} {shown:>8s}    {note}')

	w = WIN_MAIN
	rep.text(f'  quoted window, {w}')
	rep.rule()
	for name, value, fmt, note in [
		('FigFourCOneRef', g(f'Cn.{w}.slope.m1.ref'), '{:.0f}',
		 'conversion into mode one, upper slope, REF, MW'),
		('FigFourCOneCtrl', g(f'Cn.{w}.slope.m1.ctrl'), '{:.0f}',
		 'the same in CTRL'),
		('FigFourCTwoRef', g(f'Cn.{w}.slope.m2.ref'), '{:.0f}',
		 'conversion into mode two, upper slope, REF, MW'),
		('FigFourCTwoCtrl', g(f'Cn.{w}.slope.m2.ctrl'), '{:.0f}',
		 'the same in CTRL'),
		('FigFourPiOneRef', g(f'Pi.{w}.slope.m1.ref'), '{:.2f}',
		 'mode one share, upper slope, REF'),
		('FigFourPiOneCtrl', g(f'Pi.{w}.slope.m1.ctrl'), '{:.2f}',
		 'the same in CTRL'),
		('FigFourSigmaRef', g(f'Sigma.{w}.slope.ref'), '{:.2f}',
		 'recovered fraction of the conversion, upper slope, REF'),
		('FigFourPhibOneRatio', g(f'phib.{w}.slope.m1.ratio'), '{:.2f}',
		 'ratio of |phi_1(-H)| between the runs, upper slope'),
		('FigFourPhibTwoRatio', g(f'phib.{w}.slope.m2.ratio'), '{:.2f}',
		 'the same for mode two'),
		('FigFourFitSlope', g(f'fit.{w}.ref.slope'), '{:+.2f}',
		 'slope of the mode one share on the trapping index, REF'),
		('FigFourFitRank', g(f'fit.{w}.ref.rank'), '{:+.2f}',
		 'rank correlation of the same'),
	]:
		emit(name, value, fmt, note)

	rep.text('')
	rep.text('  one macro per window, for the quantities quoted as triples')
	rep.rule()
	triples = [
		('COneRef', 'Cn.{w}.slope.m1.ref', '{:.0f}',
		 'conversion into mode one, upper slope, REF, MW'),
		('COneCtrl', 'Cn.{w}.slope.m1.ctrl', '{:.0f}',
		 'conversion into mode one, upper slope, CTRL, MW'),
		('CTwoRef', 'Cn.{w}.slope.m2.ref', '{:.0f}',
		 'conversion into mode two, upper slope, REF, MW'),
		('CTwoCtrl', 'Cn.{w}.slope.m2.ctrl', '{:.0f}',
		 'conversion into mode two, upper slope, CTRL, MW'),
		('PiOneRef', 'Pi.{w}.slope.m1.ref', '{:.2f}',
		 'mode one share, upper slope, REF'),
		('PiOneCtrl', 'Pi.{w}.slope.m1.ctrl', '{:.2f}',
		 'mode one share, upper slope, CTRL'),
		('SigmaRef', 'Sigma.{w}.slope.ref', '{:.2f}',
		 'recovered fraction, upper slope, REF'),
		('SigmaCtrl', 'Sigma.{w}.slope.ctrl', '{:.2f}',
		 'recovered fraction, upper slope, CTRL'),
		('PhibOneRatio', 'phib.{w}.slope.m1.ratio', '{:.2f}',
		 'ratio of |phi_1(-H)| between the runs, upper slope'),
		('PhibTwoRatio', 'phib.{w}.slope.m2.ratio', '{:.2f}',
		 'the same for mode two'),
		('FitSlopeRef', 'fit.{w}.ref.slope', '{:+.2f}',
		 'slope of the mode one share on the trapping index, REF'),
		('FitRankRef', 'fit.{w}.ref.rank', '{:+.2f}',
		 'rank correlation of the same, REF'),
		('AreaOk', 'area_ok.{w}.slope', '{:.2f}',
		 'share of the upper slope on which the partition is quoted'),
	]
	for stem, key, fmt, note in triples:
		for wname in wins:
			emit(f'FigFour{stem}{WSUF.get(wname, wname.title())}',
				 g(key.format(w=wname)), fmt, f'{wname}, {note}')

	rep.save(STEM)
	ds.close()


if __name__ == '__main__':
	main()