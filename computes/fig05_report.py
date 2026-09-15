#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig05_report.py
=================================================================
Figure 5  -  every number the manuscript quotes, recast from the
			 saved data file
Amazon shelf internal tide manuscript, version 3

	python fig05_report.py

Reads fig05_data.nc and writes fig05_report.txt, fig05_numbers.json and
fig05_numbers.tex. Called at the end of fig05_compute.py.

Every flux term is a line integral along an isobath or a domain edge, so
the budget closes by the divergence theorem without a horizontal
derivative being taken anywhere. The residual is therefore a genuine
loss and not an accumulation of differencing error.
=================================================================
"""

import numpy as np
import xarray as xr

from report import Report


DATA = 'fig05_data.nc'
STEM = 'fig05'
WIN_MAIN = 'peak'
WSUF = {'rising': 'Ris', 'peak': 'Peak', 'decreasing': 'Dec'}
MWORD = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four'}


def main():
	ds = xr.open_dataset(DATA)

	wins = [str(w) for w in ds.window.values]
	bands = [str(b) for b in ds.band.values]
	blabel = dict(x.split(':', 1) for x in
				  ds.attrs.get('bands', '').split(';') if ':' in x)
	modes = [int(m) for m in ds.mode.values]
	isob = [float(v) for v in ds.isobath.values]
	nflux = int(ds.attrs.get('nmode_flux', 2))
	ppwmin = float(ds.attrs.get('ppw_min', 8.))
	eiso = float(ds.attrs.get('e_isobath', 1000.))
	gl = float(ds.attrs.get('gen_band_lo', 50.))
	gh = float(ds.attrs.get('gen_band_hi', 1000.))

	rep = Report('FIGURE 5, RADIATION AND WHERE THE ENERGY IS LOST')
	rep.text(f'  data file        {DATA}')
	rep.text(f'  harmonics from   {ds.attrs.get("harmonics", "?")}')
	rep.text(f'  conversion from  {ds.attrs.get("conversion", "?")}')
	rep.text(f'  isobaths         ' + ', '.join(f'{v:.0f}' for v in isob))
	rep.text(f'  radiated at      {eiso:.0f} m, against generation over '
			 f'{gl:.0f} to {gh:.0f} m')
	rep.text(f'  quoted window    {WIN_MAIN}')

	# =================================================================
	# A, the control volume budget
	# =================================================================
	rep.section('A   CONTROL VOLUME BUDGET BY BAND, all values in MW')
	rep.note('C = F_out - F_in + F_edge + loss. Every flux is a line')
	rep.note('integral, positive outward, so no horizontal derivative is')
	rep.note('taken. F_edge is what leaves through the domain edges, which')
	rep.note('cut the margin alongshore.')
	rep.text('')
	rep.row(['window', 'band', 'run', 'C', 'C+', 'F_out', 'F_in',
			 'F_edge', 'loss', 'loss/C'],
			[11, 8, 6, 9, 9, 9, 9, 9, 9, 9])
	rep.rule()
	for k, w in enumerate(wins):
		for ib, b in enumerate(bands):
			for tag in ('ref', 'ctrl'):
				C = float(ds[f'bud_C_{tag}'][k, ib])
				Cp = float(ds[f'bud_Cpos_{tag}'][k, ib])
				fo = float(ds[f'bud_F_out_{tag}'][k, ib])
				fi = float(ds[f'bud_F_in_{tag}'][k, ib])
				fe = float(ds[f'bud_F_edge_{tag}'][k, ib])
				ls = float(ds[f'bud_loss_{tag}'][k, ib])
				for nm, v in (('C', C), ('Cpos', Cp), ('F_out', fo),
							  ('F_in', fi), ('F_edge', fe), ('loss', ls)):
					rep.values[f'{nm}.{w}.{b}.{tag}'] = v
				rep.values[f'loss_frac.{w}.{b}.{tag}'] = ls/C if C else np.nan
				rep.row([w, b, tag.upper(), f'{C:.0f}', f'{Cp:.0f}',
						 f'{fo:.0f}', f'{fi:.0f}', f'{fe:.0f}', f'{ls:.0f}',
						 f'{ls/C:.2f}' if C else 'n/a'],
						[11, 8, 6, 9, 9, 9, 9, 9, 9, 9])
		rep.rule()

	# =================================================================
	# B, cross isobath power
	# =================================================================
	rep.section('B   CROSS ISOBATH POWER, MW, positive offshore')
	rep.row(['window', 'run'] + [f'{v:.0f}' for v in isob],
			[11, 6] + [8]*len(isob))
	rep.rule()
	for k, w in enumerate(wins):
		for tag in ('ref', 'ctrl'):
			v = ds[f'iso_P_tot_{tag}'][k].values
			for j, lev in enumerate(isob):
				rep.values[f'P.{w}.{int(lev)}.{tag}'] = float(v[j])
			rep.row([w, tag.upper()] +
					[f'{x:.0f}' if np.isfinite(x) else 'n/a' for x in v],
					[11, 6] + [8]*len(isob))
		rep.rule()

	rep.text('')
	rep.text('  ratio between the runs, and the offshore decay of REF')
	rep.rule()
	for k, w in enumerate(wins):
		vr = ds['iso_P_tot_ref'][k].values
		vc = ds['iso_P_tot_ctrl'][k].values
		with np.errstate(invalid='ignore', divide='ignore'):
			r = np.where(np.abs(vc) > 0, vr/vc, np.nan)
			d = np.where(np.abs(vr[0]) > 0, vr/vr[0], np.nan)
		for j, lev in enumerate(isob):
			rep.values[f'P.ratio.{w}.{int(lev)}'] = float(r[j])
			rep.values[f'P.decay.{w}.{int(lev)}'] = float(d[j])
		rep.row([w, 'ratio'] +
				[f'{x:.2f}' if np.isfinite(x) else 'n/a' for x in r],
				[11, 6] + [8]*len(isob))
		rep.row([w, 'decay'] +
				[f'{x:.2f}' if np.isfinite(x) else 'n/a' for x in d],
				[11, 6] + [8]*len(isob))

	rep.text('')
	rep.text('  modal composition of the cross isobath power, MW')
	rep.rule()
	for k, w in enumerate(wins):
		for tag in ('ref', 'ctrl'):
			for n in range(nflux):
				v = ds[f'iso_P_mode_{tag}'][k, :, n].values
				for j, lev in enumerate(isob):
					rep.values[f'Pn.{w}.{int(lev)}.m{modes[n]}.{tag}'] = \
						float(v[j])
				rep.row([w, tag.upper(), f'mode {modes[n]}'] +
						[f'{x:.0f}' if np.isfinite(x) else 'n/a' for x in v],
						[11, 6, 9] + [8]*len(isob))
		rep.rule()
	rep.note(f'The fraction of each contour that resolves mode one at')
	rep.note(f'{ppwmin:.0f} points per wavelength is given below. A modal')
	rep.note('flux quoted where that fraction is small is a bound.')
	rep.text('')
	for k, w in enumerate(wins):
		for tag in ('ref', 'ctrl'):
			v = ds[f'iso_area_res_{tag}'][k].values
			for j, lev in enumerate(isob):
				rep.values[f'res.{w}.{int(lev)}.{tag}'] = float(v[j])
			rep.row([w, tag.upper(), 'resolved'] +
					[f'{100*x:.0f}%' if np.isfinite(x) else 'n/a' for x in v],
					[11, 6, 9] + [8]*len(isob))

	# =================================================================
	# C, the radiated fraction
	# =================================================================
	rep.section(f'C   RADIATED FRACTION ACROSS THE {eiso:.0f} m ISOBATH')
	rep.row(['window', 'run', 'generation', 'crossing', 'E', 'lost'],
			[11, 6, 13, 12, 9, 9])
	rep.rule()
	for k, w in enumerate(wins):
		for tag in ('ref', 'ctrl'):
			cg = float(ds[f'rad_Cgen_{tag}'][k])
			pe = float(ds[f'rad_P_E_{tag}'][k])
			E = float(ds[f'rad_E_{tag}'][k])
			rep.values[f'Cgen.{w}.{tag}'] = cg
			rep.values[f'P_E.{w}.{tag}'] = pe
			rep.values[f'E.{w}.{tag}'] = E
			rep.row([w, tag.upper(), f'{cg:.0f} MW', f'{pe:.0f} MW',
					 f'{E:.2f}', f'{1-E:.2f}'], [11, 6, 13, 12, 9, 9])
		rep.rule()
	rep.note('E is the share of what the margin generates between the given')
	rep.note('depths that crosses the isobath. The complement is lost')
	rep.note('within the margin or leaves through the domain edges.')

	# =================================================================
	# macros
	# =================================================================
	rep.section('VALUES FOR THE FIGURE 5 RESULTS PARAGRAPH')
	V = rep.values

	def g(key):
		v = V.get(key)
		return np.nan if v is None else v

	def emit(name, value, fmt, note):
		rep.macro(name, value, fmt)
		shown = fmt.format(value) if np.isfinite(value) else 'n/a'
		rep.text(f'  \\{name:<30s} {shown:>9s}    {note}')

	triples = [
		('ERef', 'E.{w}.ref', '{:.2f}', 'radiated fraction, REF'),
		('ECtrl', 'E.{w}.ctrl', '{:.2f}', 'radiated fraction, CTRL'),
		('CgenRef', 'Cgen.{w}.ref', '{:.0f}', 'generation, REF, MW'),
		('CgenCtrl', 'Cgen.{w}.ctrl', '{:.0f}', 'generation, CTRL, MW'),
		('PERef', 'P_E.{w}.ref', '{:.0f}',
		 'power crossing the quoted isobath, REF, MW'),
		('PECtrl', 'P_E.{w}.ctrl', '{:.0f}', 'the same in CTRL, MW'),
		('LossShelfRef', 'loss.{w}.shelf.ref', '{:.0f}',
		 'loss within the shelf band, REF, MW'),
		('LossShelfCtrl', 'loss.{w}.shelf.ctrl', '{:.0f}',
		 'the same in CTRL, MW'),
		('LossSlopeRef', 'loss.{w}.slope.ref', '{:.0f}',
		 'loss within the upper slope band, REF, MW'),
		('LossSlopeCtrl', 'loss.{w}.slope.ctrl', '{:.0f}',
		 'the same in CTRL, MW'),
		('EdgeShelfRef', 'F_edge.{w}.shelf.ref', '{:.0f}',
		 'power leaving the shelf band through the domain edges, REF, MW'),
	]
	for stem, key, fmt, note in triples:
		for wname in wins:
			emit(f'FigFive{stem}{WSUF.get(wname, wname.title())}',
				 g(key.format(w=wname)), fmt, f'{wname}, {note}')

	rep.text('')
	for n in range(nflux):
		mw = MWORD.get(modes[n], str(modes[n]))
		for wname in wins:
			emit(f'FigFivePMode{mw}Ref{WSUF.get(wname, wname.title())}',
				 g(f'Pn.{wname}.{int(eiso)}.m{modes[n]}.ref'), '{:.0f}',
				 f'{wname}, mode {modes[n]} power across {eiso:.0f} m, '
				 'REF, MW')

	rep.save(STEM)
	ds.close()


if __name__ == '__main__':
	main()