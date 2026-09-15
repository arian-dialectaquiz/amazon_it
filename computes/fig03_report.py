#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig03_report.py
=================================================================
Figure 3  -  every number the manuscript quotes, recast from the
			 saved data file
Amazon shelf internal tide manuscript, version 3

	python fig03_report.py

Reads fig03_data.nc and writes fig03_report.txt, fig03_numbers.json and
fig03_numbers.tex. Called at the end of fig03_compute.py and safe to run
again at any time.

Sections follow the argument. The conversion is reported first, then it
is split between the two runs into an amplitude term and an alignment
term, then the same split is applied between windows of one run, then
the two terms are related to the near bottom stratification and to the
criticality along the break, and the signed decomposition and the
detection threshold close the block.

Multiplicative factors are quoted as exp of the weighted log terms, so
an amplitude factor of 1.30 and an alignment factor of 0.95 multiply to
the total factor of 1.24 and the reader can check the arithmetic.
=================================================================
"""

import numpy as np
import xarray as xr

from report import Report, cell_area, robust


DATA = 'fig03_data.nc'
STEM = 'fig03'
WIN_MAIN = 'peak'


def main():
	ds = xr.open_dataset(DATA)

	wins = [str(w) for w in ds.window.values]
	bands = [str(b) for b in ds.band.values]
	blabel = dict(x.split(':', 1) for x in
				  ds.attrs.get('bands', '').split(';') if ':' in x)
	brk = float(ds.attrs.get('break_isobath', 200.))
	gl = float(ds.attrs.get('gen_band_lo', 100.))
	gh = float(ds.attrs.get('gen_band_hi', 1000.))

	rep = Report('FIGURE 3, CONVERSION, ITS AMPLITUDE AND ITS ALIGNMENT')
	rep.text(f'  data file        {DATA}')
	rep.text(f'  constituents     {ds.attrs.get("constituents", "?")}')
	rep.text(f'  block length     {ds.attrs.get("block_hours", "?")} h')
	rep.text(f'  generation band  {gl:.0f} to {gh:.0f} m')
	rep.text(f'  quoted window    {WIN_MAIN}')

	# =================================================================
	# A, the conversion itself
	# =================================================================
	rep.section('A   INTEGRATED CONVERSION BY DEPTH BAND, all values in MW')
	rep.row(['window', 'band', 'C+ REF', 'C- REF', 'net REF', 'k REF',
			 'C+ CTRL', 'net CTRL', 'k CTRL', 'C+ ratio'],
			[12, 10, 9, 9, 9, 7, 9, 10, 8, 9])
	rep.rule()
	for k, w in enumerate(wins):
		for ib, b in enumerate(bands):
			pr = float(ds.Cpos_ref[k, ib]); nr = float(ds.Cneg_ref[k, ib])
			tr = float(ds.Cnet_ref[k, ib]); kr = float(ds.kappa_ref[k, ib])
			pc = float(ds.Cpos_ctrl[k, ib]); tc = float(ds.Cnet_ctrl[k, ib])
			kc = float(ds.kappa_ctrl[k, ib])
			for nm, v in (('Cpos.ref', pr), ('Cneg.ref', nr),
						  ('Cnet.ref', tr), ('kappa.ref', kr),
						  ('Cpos.ctrl', pc), ('Cnet.ctrl', tc),
						  ('kappa.ctrl', kc)):
				rep.values[f'{nm}.{w}.{b}'] = v
			rep.values[f'Cpos.ratio.{w}.{b}'] = pr/pc if pc else np.nan
			rep.row([w, b, f'{pr:.0f}', f'{nr:.0f}', f'{tr:.0f}',
					 f'{kr:.2f}', f'{pc:.0f}', f'{tc:.0f}', f'{kc:.2f}',
					 f'{pr/pc:.2f}' if pc else 'n/a'],
					[12, 10, 9, 9, 9, 7, 9, 10, 8, 9])
		rep.rule()
	rep.note('k is the cancellation index. A net conversion measures')
	rep.note('generation only where it is small.')

	# =================================================================
	# B, the factorisation between the runs
	# =================================================================
	rep.section('B   THE FACTORISATION BETWEEN THE RUNS, '
				'C = 0.5 P W cos(dphi)')
	rep.note('Terms are the log ratios of Equation (factorise), averaged')
	rep.note('over each band with C of the realistic run as the weight, and')
	rep.note('shown as multiplicative factors. Amplitude times alignment')
	rep.note('equals total exactly. The last column is the fraction of the')
	rep.note('total carried by the amplitude.')
	rep.text('')
	rep.row(['window', 'band', 'total', 'amplitude', 'alignment',
			 'C+ ratio', 'amp share', 'area used'],
			[12, 10, 10, 12, 12, 10, 11, 11])
	rep.rule()
	for k, w in enumerate(wins):
		for ib, b in enumerate(bands):
			lt = float(ds.fac_ln_tot[k, ib])
			la = float(ds.fac_ln_amp[k, ib])
			ll = float(ds.fac_ln_ali[k, ib])
			sh = float(ds.fac_amp_share[k, ib])
			af = float(ds.fac_area_frac[k, ib])
			ri = float(ds.fac_ratio_int[k, ib])
			for nm, v in (('fac.total', np.exp(lt)),
						  ('fac.amplitude', np.exp(la)),
						  ('fac.alignment', np.exp(ll)),
						  ('fac.amp_share', sh),
						  ('fac.area_frac', af)):
				rep.values[f'{nm}.{w}.{b}'] = v
			rep.row([w, b, f'{np.exp(lt):.3f}', f'{np.exp(la):.3f}',
					 f'{np.exp(ll):.3f}', f'{ri:.2f}',
					 f'{100*sh:.0f} %', f'{100*af:.0f} %'],
					[12, 10, 10, 12, 12, 10, 11, 11])
		rep.rule()
	rep.note('The area used is the share of the band on which both runs')
	rep.note('convert and remain aligned, which is where the log ratio is')
	rep.note('defined. Where it is small the weighted factors describe')
	rep.note('only part of the band.')

	# =================================================================
	# C, the same split through the season
	# =================================================================
	rep.section('C   THE SAME SPLIT BETWEEN WINDOWS, relative to rising')
	rep.row(['run', 'window', 'band', 'total', 'amplitude', 'alignment'],
			[8, 12, 10, 10, 12, 12])
	rep.rule()
	for tag in ('ref', 'ctrl'):
		for k, w in enumerate(wins):
			if k == 0:
				continue
			for ib, b in enumerate(bands):
				lt = float(ds[f'seas_ln_tot_{tag}'][k, ib])
				la = float(ds[f'seas_ln_amp_{tag}'][k, ib])
				ll = float(ds[f'seas_ln_ali_{tag}'][k, ib])
				for nm, v in (('seas.total', np.exp(lt)),
							  ('seas.amplitude', np.exp(la)),
							  ('seas.alignment', np.exp(ll))):
					rep.values[f'{nm}.{tag}.{w}.{b}'] = v
				rep.row([tag.upper(), w, b, f'{np.exp(lt):.3f}',
						 f'{np.exp(la):.3f}', f'{np.exp(ll):.3f}'],
						[8, 12, 10, 10, 12, 12])
		rep.rule()
	rep.note('This is the seasonal cycle of the conversion split into the')
	rep.note('same two channels within a single experiment, so it separates')
	rep.note('the ambient signal from the discharge anomaly of section B.')

	# =================================================================
	# D, along the shelf break
	# =================================================================
	rep.section(f'D   ALONG THE {brk:.0f} m ISOBATH')
	rep.row(['window', 'run', 'median P', 'median cos', 'median dphi',
			 'median C', 'frac cos<0'], [12, 7, 12, 12, 13, 12, 12])
	rep.rule()
	for k, w in enumerate(wins):
		for tag in ('ref', 'ctrl'):
			P = ds[f'P_{tag}_break'][k].values
			X = ds[f'cosdphi_{tag}_break'][k].values
			D = ds[f'dphi_{tag}_break'][k].values
			C = ds[f'C_{tag}_break'][k].values
			fneg = float(np.nanmean((X < 0).astype(float)))
			for nm, v in (('break.P', np.nanmedian(P)),
						  ('break.cos', np.nanmedian(X)),
						  ('break.dphi', np.nanmedian(D)),
						  ('break.C', np.nanmedian(C)),
						  ('break.frac_neg', fneg)):
				rep.values[f'{nm}.{w}.{tag}'] = float(v)
			rep.row([w, tag.upper(), f'{np.nanmedian(P):.1f} Pa',
					 f'{np.nanmedian(X):+.3f}', f'{np.nanmedian(D):+.1f} deg',
					 f'{np.nanmedian(C):.2e}', f'{100*fneg:.1f} %'],
					[12, 7, 12, 12, 13, 12, 12])
		rep.rule()

	rep.text('')
	rep.text('  relations binned along the break')
	rep.rule()
	for k, w in enumerate(wins):
		for tag in ('ref', 'ctrl'):
			a = ds[f'alpha_{tag}_break'][k].values
			X = ds[f'cosdphi_{tag}_break'][k].values
			P = ds[f'P_{tag}_break'][k].values
			N = ds[f'nb2_{tag}_break'][k].values
			ok = np.isfinite(a) & np.isfinite(X)
			r1 = float(np.corrcoef(a[ok], X[ok])[0, 1]) if ok.sum() > 20 \
				else np.nan
			ok2 = np.isfinite(N) & np.isfinite(P) & (N > 0) & (P > 0)
			if ok2.sum() > 20:
				s2, _ = np.polyfit(np.log(N[ok2]), np.log(P[ok2]), 1)
				r2 = float(np.corrcoef(np.log(N[ok2]), np.log(P[ok2]))[0, 1])
			else:
				s2, r2 = np.nan, np.nan
			rep.values[f'break.corr_cos_alpha.{w}.{tag}'] = r1
			rep.values[f'break.slope_lnP_lnNb2.{w}.{tag}'] = float(s2)
			rep.values[f'break.corr_lnP_lnNb2.{w}.{tag}'] = r2
			rep.row([w, tag.upper(),
					 f'corr(cos, alpha) {r1:+.2f}',
					 f'd lnP / d lnNb2 {s2:+.2f}',
					 f'r {r2:+.2f}'], [12, 7, 26, 26, 12])

	# =================================================================
	# E, the detection threshold
	# =================================================================
	rep.section('E   DETECTION THRESHOLD FROM THE CONTROL RUN')
	for k, w in enumerate(wins):
		v = float(ds.ctrl_block_iqr[k].values)
		rep.val(f'noise.{w}.iqr', v, '{:.3e}', 'W m-2',
				f'{w}, interquartile range of the block conversion in CTRL')
	rep.note('Anomalies smaller than twice this are not reported as effects')
	rep.note('of the discharge.')

	# =================================================================
	# macros
	# =================================================================
	rep.section('VALUES FOR THE FIGURE 3 RESULTS PARAGRAPH')
	w = WIN_MAIN
	V = rep.values

	def g(key):
		v = V.get(key)
		return np.nan if v is None else v

	pairs = [
		('FigThreeCposShelfRef', g(f'Cpos.ref.{w}.shelf'), '{:.0f}',
		 'positive shelf conversion, REF, MW'),
		('FigThreeCposShelfCtrl', g(f'Cpos.ctrl.{w}.shelf'), '{:.0f}',
		 'the same in CTRL'),
		('FigThreeCposSlopeRef', g(f'Cpos.ref.{w}.slope'), '{:.0f}',
		 'positive upper slope conversion, REF, MW'),
		('FigThreeCposSlopeCtrl', g(f'Cpos.ctrl.{w}.slope'), '{:.0f}',
		 'the same in CTRL'),
		('FigThreeFacTotal', g(f'fac.total.{w}.slope'), '{:.2f}',
		 'total factor between the runs, upper slope'),
		('FigThreeFacAmp', g(f'fac.amplitude.{w}.slope'), '{:.2f}',
		 'its amplitude term'),
		('FigThreeFacAli', g(f'fac.alignment.{w}.slope'), '{:.2f}',
		 'its alignment term'),
		('FigThreeAmpShare', 100*g(f'fac.amp_share.{w}.slope'), '{:.0f}',
		 'share of the total carried by the amplitude, per cent'),
		('FigThreeSeasTotal', g(f'seas.total.ref.{w}.slope'), '{:.2f}',
		 'seasonal factor at peak against rising, REF, upper slope'),
		('FigThreeSeasAmp', g(f'seas.amplitude.ref.{w}.slope'), '{:.2f}',
		 'its amplitude term'),
		('FigThreeSeasAli', g(f'seas.alignment.ref.{w}.slope'), '{:.2f}',
		 'its alignment term'),
		('FigThreeCosRef', g(f'break.cos.{w}.ref'), '{:+.2f}',
		 'median alignment along the break, REF'),
		('FigThreeCosCtrl', g(f'break.cos.{w}.ctrl'), '{:+.2f}',
		 'the same in CTRL'),
		('FigThreeCorrCosAlpha', g(f'break.corr_cos_alpha.{w}.ref'), '{:+.2f}',
		 'correlation of the alignment with the criticality, REF'),
		('FigThreeSlopePNb', g(f'break.slope_lnP_lnNb2.{w}.ref'), '{:+.2f}',
		 'slope of ln P on ln N_b squared, REF'),
		('FigThreeKappaSlope', g(f'kappa.ref.{w}.slope'), '{:.2f}',
		 'cancellation index of the upper slope, REF'),
	]
	for name, value, fmt, note in pairs:
		rep.macro(name, value, fmt)
		shown = fmt.format(value) if np.isfinite(value) else 'n/a'
		rep.text(f'  \\{name:<24s} {shown:>8s}    {note}')

	rep.save(STEM)
	ds.close()


if __name__ == '__main__':
	main()