#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig04_compute.py
=================================================================
Figure 4  -  The modal partition of the generated wave
Amazon shelf internal tide manuscript, version 3

What the convergence test settled
---------------------------------
The expansion of the conversion onto the vertical modes is sound. Taken
pointwise over the cells that actually convert, the cumulative recovered
fraction reaches 0.90 to 1.00 by the fourth mode in every window, band
and run, and 0.99 by the second mode in the deep ocean.

What failed in the first attempt was the normalisation. The recovered
fraction had been formed as a ratio of BAND INTEGRALS of C, and Figure 3
already shows that those integrals are small residuals of large opposing
contributions. Dividing by a number that nearly vanishes produced
recovered fractions of -0.96 and 3.42 and modal shares of 1.87 and
-0.87. The modes were never the problem.

So this version keeps FOUR modes, integrates over the GENERATION CELLS
only, those where C > 0, reports the partition pointwise as well, and
extends onto the shelf, since the projection needs vertical resolution
alone.

The per mode factorisation
--------------------------
The mode one share turned out not to follow phi_n(-H), the rank
correlation coming back at -0.31, -0.10 and -0.09, while the conversion
into mode two rose by factors of 1.9 to 3.1 even though phi_2(-H) FELL
to 0.60 of its control value. Something other than the structure
function at the bed is moving the transfer between modes.

Writing the modal pressure and the barotropic vertical velocity as
harmonics turns the modal conversion into

	C_n = 0.5 phi_n(-H) P_n W cos(dphi_n)

with P_n the amplitude of the modal pressure, W the amplitude of
w_bt(-H), which the two experiments share, and cos(dphi_n) the alignment
between them. The ratio between the runs then factorises exactly,

	ln(C_n^REF / C_n^CTRL) = ln(phi ratio) + ln(P_n ratio)
	                         + ln(cos ratio)

with no residual, so the shift between modes is attributed to the
structure function at the bed, to the amplitude of the modal pressure or
to its alignment with the forcing. The identity is checked cell by cell
and the check is printed.

Outputs  fig04_data.nc, and the report written by fig04_report.py

Method
------
	p_n     = (1/H) int_{-H}^{0} p_bc phi_n dz
	C_n     = phi_n(-H) * 0.5 * (a_pn a_w + b_pn b_w)
	Pi_n    = int_gen C_n dA / int_gen C dA
	Sigma_N = sum_{n<=N} Pi_n

The harmonic coefficients come from fig03_harmonics.nc. The vertical
modes need the window mean stratification, cached in MEANCACHE.
=================================================================
"""

import os
import functools
import warnings
import numpy as np
import xarray as xr

from fig01_compute import (
	REALISTIC, CONTROL, WINDOWS, BREAK_ISOBATH,
	open_run, compute_N2, vertical_modes,
)
from fig02_compute import (
	phi_from_Phi, trapping_index, plume_base, binned, linfit, S_LENS,
)

print = functools.partial(print, flush=True)


# =====================================================================
# CONFIGURATION
# =====================================================================
HARMFILE  = 'fig03_harmonics.nc'
FIG03     = 'fig03_data.nc'
OUTFILE   = 'fig04_data.nc'
MEANCACHE = 'mean_state.nc'

NMODE = 8                 # modes solved, so the approach to unity is shown
NMODE_KEEP = 4            # modes quoted, the number the test justifies
MIN_LEV = 4               # levels per half wave required to retain a mode

H_MIN = 50.               # m, vertical resolvability sets this, not the
H_MAX = 3500.             # horizontal criterion of Figure 1

SIGMA_MIN = 0.60          # pointwise partition quoted only where the
SIGMA_MAX = 1.40          # cumulative recovered fraction sits in this band
STRONG_PCTL = 75.         # percentile of |C| defining the strong cells

ZW_TOL = 0.05             # m, tolerance on the window mean z_w cross check
ID_TOL = 1e-8             # tolerance on the modal identity check

BANDS = [('shelf, 50 to 250 m',         50.,  250., 'shelf'),
		 ('upper slope, 250 to 1000 m', 250., 1000., 'slope'),
		 ('deep, 1000 to 3500 m',      1000., 3500., 'deep')]

DEPTH_BINS = np.logspace(np.log10(H_MIN), np.log10(H_MAX), 21)
T1_BINS = np.linspace(0.05, 0.95, 19)
N_SUB = 5000
SEED = 20260826

RUNS = ('ref', 'ctrl')


# =====================================================================
# HELPERS
# =====================================================================
def grad_h_components(h, pm, pn):
	return np.gradient(h, axis=1)*pm, np.gradient(h, axis=0)*pn


def area_weights(pm, pn, mask):
	return np.where(mask > 0, 1.0/(pm*pn), 0.0)


def project(a_p, phi, dz, H):
	"""Modal amplitude, p_n = (1/H) int p_bc phi_n dz."""
	return np.nansum(a_p*phi*dz, axis=0)/np.maximum(H, 1e-9)


def resolvable(phi, min_lev=MIN_LEV):
	"""
	Whether the vertical grid carries a mode, judged by the mean number
	of levels between sign changes of its structure function.
	"""
	nz = phi.shape[0]
	nflip = np.sum(np.diff(np.sign(phi), axis=0) != 0, axis=0)
	return nz/np.maximum(nflip + 1, 1) >= min_lev


def mean_state(ds, tag, wname, t0, t1):
	"""Window mean T, S, z_rho and z_w, cached on disk after the first run."""
	key = f'{tag}_{wname}'
	stamp = f'{t0}..{t1}'
	if MEANCACHE and os.path.exists(MEANCACHE):
		with xr.open_dataset(MEANCACHE) as dc:
			if f'temp_{key}' in dc and dc.attrs.get(f'win_{key}') == stamp:
				print(f'    mean state read from {MEANCACHE}')
				return (dc[f'temp_{key}'].values, dc[f'salt_{key}'].values,
						dc[f'zr_{key}'].values, dc[f'zw_{key}'].values)
			if f'temp_{key}' in dc:
				print(f'    {MEANCACHE} holds {key} for other window bounds, '
					  'recomputing')

	sel = ds.sel(ocean_time=slice(t0, t1))
	print(f'    mean state from {sel.sizes["ocean_time"]} records')
	temp = sel.temp.mean('ocean_time').values
	salt = sel.salt.mean('ocean_time').values
	zr = sel.z_rho.mean('ocean_time').values
	zw = sel.z_w.mean('ocean_time').values

	if MEANCACHE:
		d3 = ('s_rho', 'eta_rho', 'xi_rho')
		new = xr.Dataset({
			f'temp_{key}': (d3, temp.astype('float32')),
			f'salt_{key}': (d3, salt.astype('float32')),
			f'zr_{key}': (d3, zr.astype('float32')),
			f'zw_{key}': (('s_w', 'eta_rho', 'xi_rho'),
						  zw.astype('float32'))})
		new.attrs[f'win_{key}'] = stamp
		if os.path.exists(MEANCACHE):
			with xr.open_dataset(MEANCACHE) as dc:
				old = dc.load()
			attrs = dict(old.attrs); attrs.update(new.attrs)
			new = xr.merge([old.drop_vars(
				[v for v in new.data_vars if v in old], errors='ignore'), new])
			new.attrs = attrs
		new.to_netcdf(MEANCACHE + '.tmp')
		os.replace(MEANCACHE + '.tmp', MEANCACHE)
		print(f'    mean state cached in {MEANCACHE}')
	return temp, salt, zr, zw


def rank_corr(x, y):
	"""Spearman rank correlation, numpy only."""
	x = np.asarray(x, float); y = np.asarray(y, float)
	ok = np.isfinite(x) & np.isfinite(y)
	if ok.sum() < 20:
		return np.nan
	def ranks(v):
		order = np.argsort(v, kind='mergesort')
		r = np.empty(v.size, float)
		r[order] = np.arange(v.size, dtype=float)
		return r
	return float(np.corrcoef(ranks(x[ok]), ranks(y[ok]))[0, 1])


# =====================================================================
# MAIN
# =====================================================================
def main():
	if not os.path.exists(HARMFILE):
		raise SystemExit(f'{HARMFILE} not found. Run fig03_compute.py with '
						 'SAVE_HARMONICS set to True first.')

	print('opening the harmonic coefficients')
	dh = xr.open_dataset(HARMFILE)
	lon = dh.lon_rho.values; lat = dh.lat_rho.values
	h = dh.h.values; mask = dh.mask_rho.values
	pm = dh.pm.values; pn = dh.pn.values
	ny, nx = h.shape
	wet = mask > 0
	flat = wet.ravel()
	nwet = int(flat.sum())
	nz = dh.sizes['s_rho']
	wins = [str(w) for w in dh.window.values]
	nwin = len(wins)
	nband = len(BANDS)
	aw = area_weights(pm, pn, mask)
	awf = aw.ravel()[flat]
	dHdx, dHdy = grad_h_components(h, pm, pn)
	dhx = dHdx.ravel()[flat]; dhy = dHdy.ravel()[flat]
	hb = h.ravel()[flat]
	print(f'  grid {ny} x {nx}, {nwet} wet columns, {nz} levels')
	print(f'  solving {NMODE} modes, quoting {NMODE_KEEP}')

	band_ok = wet & (h >= H_MIN) & (h <= H_MAX)
	print(f'  analysis over {H_MIN:.0f} to {H_MAX:.0f} m, '
		  f'{100*np.sum(np.where(band_ok, aw, 0.))/np.sum(np.where(wet, aw, 0.)):.1f}'
		  ' % of the wet area')

	dsr = open_run(REALISTIC)
	dsc = open_run(CONTROL)

	def unflat(v):
		a = np.full(ny*nx, np.nan); a[flat] = v
		return a.reshape(ny, nx)

	# -----------------------------------------------------------------
	# allocation
	# -----------------------------------------------------------------
	out = {f'{k}_{t}': np.full((nwin, ny, nx), np.nan)
		   for k in ('C', 'Sigma', 'trap1') for t in RUNS}
	outm = {f'{k}_{t}': np.full((nwin, NMODE, ny, nx), np.nan)
			for k in ('Cn', 'Pin', 'phib', 'Pn', 'cosn') for t in RUNS}
	prof = {f'phib_{t}': np.full((nwin, NMODE, DEPTH_BINS.size - 1, 3),
								 np.nan) for t in RUNS}

	integ = {f'{k}_{t}': np.full((nwin, nband), np.nan)
			 for k in ('Cgen', 'area_gen', 'area_ok') for t in RUNS}
	integm = {f'{k}_{t}': np.full((nwin, nband, NMODE), np.nan)
			  for k in ('Cn', 'Pin', 'Scum', 'Sstrong', 'res_frac')
			  for t in RUNS}

	# per mode factorisation of the ratio between the runs
	fac = {k: np.full((nwin, nband, NMODE), np.nan)
		   for k in ('ln_tot', 'ln_phi', 'ln_P', 'ln_cos', 'area')}

	rng = np.random.default_rng(SEED)
	sub = {f'{k}_{t}': np.full((nwin, N_SUB), np.nan)
		   for k in ('trap1', 'Pi1', 'phib1', 'h') for t in RUNS}
	binf = {f'{k}_{t}': np.full((nwin, T1_BINS.size - 1), np.nan)
			for k in ('med', 'q25', 'q75') for t in RUNS}
	fitp = {f'{k}_{t}': np.full(nwin, np.nan)
			for k in ('slope_T1', 'rank_T1', 'rank_phib') for t in RUNS}

	# -----------------------------------------------------------------
	# window loop
	# -----------------------------------------------------------------
	for iw, wname in enumerate(wins):
		t0, t1 = WINDOWS[wname]
		for tag, ds in (('ref', dsr), ('ctrl', dsc)):
			print(f'  {wname}/{tag}')
			temp, salt, zr, zw = mean_state(ds, tag, wname, t0, t1)

			zwh = dh[f'zw_mean_{tag}'].isel(window=iw).values
			d = float(np.nanmax(np.abs(zw - zwh)))
			if d > ZW_TOL:
				warnings.warn(f'{wname}/{tag}: window mean z_w differs from '
							  f'the harmonics file by {d:.3g} m')

			N2 = compute_N2(temp, salt, zr, zw, lon, lat)
			Nw = zw.shape[0]
			N2f = N2.reshape(Nw, -1)[:, flat]
			zwf = zw.reshape(Nw, -1)[:, flat]
			zrf = zr.reshape(nz, -1)[:, flat]
			saltf = salt.reshape(nz, -1)[:, flat]

			c, Phi = vertical_modes(N2f, zwf, nmodes=NMODE + 1,
									free_surface=True)
			dzf = np.diff(zwf, axis=0)
			Hf = zwf[-1] - zwf[0]

			a_p = dh[f'a_pbc_{tag}'].isel(window=iw).values
			b_p = dh[f'b_pbc_{tag}'].isel(window=iw).values
			a_p = a_p.reshape(nz, -1)[:, flat].astype(float)
			b_p = b_p.reshape(nz, -1)[:, flat].astype(float)
			a_ub = dh[f'a_ubt_{tag}'].isel(window=iw).values.ravel()[flat]
			b_ub = dh[f'b_ubt_{tag}'].isel(window=iw).values.ravel()[flat]
			a_vb = dh[f'a_vbt_{tag}'].isel(window=iw).values.ravel()[flat]
			b_vb = dh[f'b_vbt_{tag}'].isel(window=iw).values.ravel()[flat]

			a_w = -(a_ub*dhx + a_vb*dhy)
			b_w = -(b_ub*dhx + b_vb*dhy)
			Cf = 0.5*(a_p[0]*a_w + b_p[0]*b_w)

			# amplitude and phase of the barotropic vertical velocity,
			# needed by the per mode factorisation
			Wamp = np.hypot(a_w, b_w)
			ph_w = np.arctan2(b_w, a_w)

			Cn = np.full((NMODE, nwet), np.nan)
			phib = np.full((NMODE, nwet), np.nan)
			Pn = np.full((NMODE, nwet), np.nan)
			cosn = np.full((NMODE, nwet), np.nan)
			res = np.zeros((NMODE, nwet), dtype=bool)
			T1f = np.full(nwet, np.nan)
			for n in range(1, NMODE + 1):
				phin, _ = phi_from_Phi(Phi[:, n, :], zwf, c[n])
				a_pn = project(a_p, phin, dzf, Hf)
				b_pn = project(b_p, phin, dzf, Hf)
				Cn[n - 1] = phin[0]*0.5*(a_pn*a_w + b_pn*b_w)
				phib[n - 1] = phin[0]
				Pn[n - 1] = np.hypot(a_pn, b_pn)
				cosn[n - 1] = np.cos(np.arctan2(b_pn, a_pn) - ph_w)
				res[n - 1] = resolvable(phin)
				if n == 1:
					hpf, _, _ = plume_base(saltf, zrf, thresh=S_LENS)
					T1f = trapping_index(phin, zwf,
										 np.nan_to_num(hpf, nan=0.0))
					T1f = np.where(np.isfinite(hpf), T1f, np.nan)

			# identity check, C_n against 0.5 phi_n(-H) P_n W cos(dphi_n)
			chk = 0.5*phib*Pn*Wamp[None, :]*cosn
			den = np.nanmax(np.abs(Cn))
			err = np.nanmax(np.abs(Cn - chk))/max(den, 1e-30)
			print(f'    modal identity check, max relative error {err:.2e}')
			if err > ID_TOL:
				warnings.warn(f'{wname}/{tag}: the modal factorisation does '
							  'not close, check the projection')

			Ccum = np.cumsum(np.nan_to_num(Cn), axis=0)
			with np.errstate(invalid='ignore', divide='ignore'):
				Pin = np.where(np.abs(Cf) > 0, Cn/Cf[None, :], np.nan)
				Sig = np.where(np.abs(Cf) > 0,
							   Ccum[NMODE_KEEP - 1]/Cf, np.nan)

			out[f'C_{tag}'][iw] = unflat(Cf)
			out[f'Sigma_{tag}'][iw] = unflat(Sig)
			out[f'trap1_{tag}'][iw] = unflat(T1f)
			for n in range(NMODE):
				outm[f'Cn_{tag}'][iw, n] = unflat(Cn[n])
				outm[f'Pin_{tag}'][iw, n] = unflat(Pin[n])
				outm[f'phib_{tag}'][iw, n] = unflat(phib[n])
				outm[f'Pn_{tag}'][iw, n] = unflat(Pn[n])
				outm[f'cosn_{tag}'][iw, n] = unflat(cosn[n])

			for n in range(NMODE):
				for kb in range(DEPTH_BINS.size - 1):
					s = (hb >= DEPTH_BINS[kb]) & (hb < DEPTH_BINS[kb + 1]) \
						& np.isfinite(phib[n])
					if s.sum() > 20:
						v = np.abs(phib[n][s])
						prof[f'phib_{tag}'][iw, n, kb] = [
							np.nanmedian(v), np.nanpercentile(v, 25),
							np.nanpercentile(v, 75)]

			# ---- band integrals over the GENERATION CELLS
			for ib, (nm, lo, hi, short) in enumerate(BANDS):
				mf = (hb >= lo) & (hb < hi)
				gen = mf & np.isfinite(Cf) & (Cf > 0)
				wg = np.where(gen, awf, 0.0)
				tot = float(np.nansum(np.nan_to_num(Cf)*wg)/1e6)
				integ[f'Cgen_{tag}'][iw, ib] = tot
				integ[f'area_gen_{tag}'][iw, ib] = float(
					np.nansum(wg)/max(np.nansum(np.where(mf, awf, 0.)), 1e-9))

				absC = np.abs(Cf)[mf & np.isfinite(Cf)]
				thr = np.percentile(absC, STRONG_PCTL) if absC.size else np.inf
				strong = mf & np.isfinite(Cf) & (np.abs(Cf) >= thr)

				run = 0.0
				for n in range(NMODE):
					v = float(np.nansum(np.nan_to_num(Cn[n])*wg)/1e6)
					integm[f'Cn_{tag}'][iw, ib, n] = v
					integm[f'Pin_{tag}'][iw, ib, n] = v/tot if tot else np.nan
					run += v
					integm[f'Scum_{tag}'][iw, ib, n] = run/tot if tot else np.nan
					with np.errstate(invalid='ignore', divide='ignore'):
						ss = np.where(np.abs(Cf) > 0, Ccum[n]/Cf, np.nan)
					integm[f'Sstrong_{tag}'][iw, ib, n] = float(
						np.nanmedian(ss[strong])) if strong.any() else np.nan
					integm[f'res_frac_{tag}'][iw, ib, n] = float(
						np.nansum(np.where(mf & res[n], awf, 0.))
						/ max(np.nansum(np.where(mf, awf, 0.)), 1e-9))

				okm = gen & np.isfinite(Sig) & (Sig > SIGMA_MIN) \
					  & (Sig < SIGMA_MAX)
				integ[f'area_ok_{tag}'][iw, ib] = float(
					np.nansum(np.where(okm, awf, 0.))/max(np.nansum(wg), 1e-9))

			# ---- the mode one share against the trapping index
			keep = (hb >= H_MIN) & (hb <= H_MAX) & (Cf > 0) \
				   & np.isfinite(T1f) & np.isfinite(Pin[0]) \
				   & np.isfinite(Sig) & (Sig > SIGMA_MIN) & (Sig < SIGMA_MAX)
			x, y = T1f[keep], Pin[0][keep]
			_, med, q25, q75 = binned(x, y, T1_BINS)
			binf[f'med_{tag}'][iw] = med
			binf[f'q25_{tag}'][iw] = q25
			binf[f'q75_{tag}'][iw] = q75
			a_, b_, _ = linfit(x, y)
			fitp[f'slope_T1_{tag}'][iw] = a_
			fitp[f'rank_T1_{tag}'][iw] = rank_corr(x, y)
			fitp[f'rank_phib_{tag}'][iw] = rank_corr(np.abs(phib[0][keep]), y)
			if x.size:
				pick = rng.choice(x.size, size=min(N_SUB, x.size),
								  replace=False)
				sub[f'trap1_{tag}'][iw, :pick.size] = x[pick]
				sub[f'Pi1_{tag}'][iw, :pick.size] = y[pick]
				sub[f'phib1_{tag}'][iw, :pick.size] = np.abs(phib[0][keep])[pick]
				sub[f'h_{tag}'][iw, :pick.size] = hb[keep][pick]
			print(f'    Pi_1 on T1, slope {a_:+.3f}, rank '
				  f'{fitp[f"rank_T1_{tag}"][iw]:+.2f}, on |phi_1(-H)| rank '
				  f'{fitp[f"rank_phib_{tag}"][iw]:+.2f}, {x.size} cells')

	# -----------------------------------------------------------------
	# per mode factorisation of the ratio between the runs
	#
	#   ln(C_n^REF / C_n^CTRL) = ln(phi ratio) + ln(P_n ratio)
	#                            + ln(cos ratio)
	#
	# exact cell by cell wherever both runs generate into that mode with
	# the modal response and the forcing still aligned. Formed before the
	# maps are masked, so the shelf band is complete.
	# -----------------------------------------------------------------
	print('forming the per mode factorisation')
	for iw in range(nwin):
		for ib, (nm, lo, hi, short) in enumerate(BANDS):
			m = (hb >= lo) & (hb < hi)
			for n in range(NMODE):
				cr = outm['Cn_ref'][iw, n].ravel()[flat]
				cc = outm['Cn_ctrl'][iw, n].ravel()[flat]
				pr = outm['Pn_ref'][iw, n].ravel()[flat]
				pc = outm['Pn_ctrl'][iw, n].ravel()[flat]
				fr = outm['phib_ref'][iw, n].ravel()[flat]      # signed
				fc = outm['phib_ctrl'][iw, n].ravel()[flat]
				xr_ = outm['cosn_ref'][iw, n].ravel()[flat]      # signed
				xc_ = outm['cosn_ctrl'][iw, n].ravel()[flat]
				# each factor need only keep its sign between the runs, so
				# the ratio is positive. phi_n(-H) alternates sign with mode
				# number, and cos(dphi_n) follows it wherever C_n > 0.
				ok = m & (cr > 0) & (cc > 0) & (pr > 0) & (pc > 0) \
					 & (fr*fc > 0) & (xr_*xc_ > 0)
				if ok.sum() < 20:
					continue
				wt = cr[ok]*awf[ok]
				tw = wt.sum()
				if tw <= 0:
					continue
				fac['ln_tot'][iw, ib, n] = float(
					np.sum(wt*np.log(cr[ok]/cc[ok]))/tw)
				fac['ln_phi'][iw, ib, n] = float(
					np.sum(wt*np.log(np.abs(fr[ok])/np.abs(fc[ok])))/tw)
				fac['ln_P'][iw, ib, n] = float(
					np.sum(wt*np.log(pr[ok]/pc[ok]))/tw)
				fac['ln_cos'][iw, ib, n] = float(
					np.sum(wt*np.log(np.abs(xr_[ok])/np.abs(xc_[ok])))/tw)
				fac['area'][iw, ib, n] = float(
					np.sum(np.where(ok, awf, 0.))
					/ max(np.sum(np.where(m, awf, 0.)), 1e-9))

	# -----------------------------------------------------------------
	# cross check against Figure 3
	# -----------------------------------------------------------------
	if os.path.exists(FIG03):
		d3 = xr.open_dataset(FIG03)
		for tag in RUNS:
			a = out[f'C_{tag}']; b = d3[f'C_{tag}'].values
			err = np.nanmax(np.abs(a - b))/max(np.nanmax(np.abs(b)), 1e-30)
			print(f'  cross check {tag}: C against {FIG03}, max relative '
				  f'difference {err:.2e}')
		d3.close()

	# -----------------------------------------------------------------
	# convergence over the generation cells
	# -----------------------------------------------------------------
	line = '=' * 78
	print('\n' + line)
	print('CUMULATIVE RECOVERED FRACTION OVER THE GENERATION CELLS')
	print(line)
	hdr = f'{"window":>11} {"band":>7} {"run":>5} ' + \
		  ' '.join(f'{n+1:>6}' for n in range(NMODE))
	for ib, (nm, lo, hi, short) in enumerate(BANDS):
		print(f'\n  {nm}')
		print('  ' + hdr)
		for iw, wname in enumerate(wins):
			for tag in RUNS:
				v = integm[f'Scum_{tag}'][iw, ib]
				row = f'{wname:>11} {short:>7} {tag.upper():>5} '
				row += ' '.join(f'{x:6.2f}' if np.isfinite(x) else '   n/a'
								for x in v)
				print('  ' + row)
		print('  ' + '.' * 74)
	print(line + '\n')

	# -----------------------------------------------------------------
	# the per mode factorisation, printed
	# -----------------------------------------------------------------
	print(line)
	print('PER MODE FACTORISATION OF THE RATIO BETWEEN THE RUNS')
	print(line)
	print('  total = phi x P_n x alignment, shown as multiplicative')
	print('  factors. The three multiply to the total exactly. The area')
	print('  is the share of the band on which both runs generate into')
	print('  that mode with the response still aligned.')
	for ib, (nm, lo, hi, short) in enumerate(BANDS):
		print(f'\n  {nm}')
		print(f'  {"window":>11} {"mode":>5} {"total":>8} {"phi":>8} '
			  f'{"P_n":>8} {"align":>8} {"area":>8}')
		print('  ' + '.' * 62)
		for iw, wname in enumerate(wins):
			for n in range(NMODE_KEEP):
				t_ = np.exp(fac['ln_tot'][iw, ib, n])
				f_ = np.exp(fac['ln_phi'][iw, ib, n])
				p_ = np.exp(fac['ln_P'][iw, ib, n])
				x_ = np.exp(fac['ln_cos'][iw, ib, n])
				a_ = fac['area'][iw, ib, n]
				def fm(v, w=8, d=2):
					return f'{v:{w}.{d}f}' if np.isfinite(v) else f'{"n/a":>{w}}'
				print(f'  {wname:>11} {n+1:>5} {fm(t_)} {fm(f_)} {fm(p_)} '
					  f'{fm(x_)} {fm(100*a_, 7, 0)}%')
	print(line + '\n')

	# -----------------------------------------------------------------
	# mask the maps and write
	# -----------------------------------------------------------------
	keep_map = band_ok[None, :, :]
	for k in list(out):
		out[k] = np.where(keep_map, out[k], np.nan)
	for k in list(outm):
		outm[k] = np.where(keep_map[:, None, :, :], outm[k], np.nan)

	print('writing', OUTFILE)
	dctr = np.sqrt(DEPTH_BINS[1:]*DEPTH_BINS[:-1])
	coords = dict(
		window=('window', wins),
		mode=('mode', np.arange(1, NMODE + 1)),
		band=('band', [b[3] for b in BANDS]),
		eta_rho=('eta_rho', np.arange(ny)),
		xi_rho=('xi_rho', np.arange(nx)),
		depth_bin=('depth_bin', dctr),
		t1_bin=('t1_bin', 0.5*(T1_BINS[1:] + T1_BINS[:-1])),
		sub=('sub', np.arange(N_SUB)),
		stat=('stat', ['median', 'q25', 'q75']),
	)
	dv = {k: (('window', 'eta_rho', 'xi_rho'), v) for k, v in out.items()}
	dv.update({k: (('window', 'mode', 'eta_rho', 'xi_rho'), v)
			   for k, v in outm.items()})
	dv.update({f'prof_{k}': (('window', 'mode', 'depth_bin', 'stat'), v)
			   for k, v in prof.items()})
	dv.update({f'int_{k}': (('window', 'band'), v) for k, v in integ.items()})
	dv.update({f'int_{k}': (('window', 'band', 'mode'), v)
			   for k, v in integm.items()})
	dv.update({f'fac_{k}': (('window', 'band', 'mode'), v)
			   for k, v in fac.items()})
	dv.update({f'bin_{k}': (('window', 't1_bin'), v) for k, v in binf.items()})
	dv.update({f'fit_{k}': ('window', v) for k, v in fitp.items()})
	dv.update({f'sub_{k}': (('window', 'sub'), v) for k, v in sub.items()})
	dv.update(dict(
		lon_rho=(('eta_rho', 'xi_rho'), lon),
		lat_rho=(('eta_rho', 'xi_rho'), lat),
		h=(('eta_rho', 'xi_rho'), h),
		mask_rho=(('eta_rho', 'xi_rho'), mask),
	))

	ds_out = xr.Dataset(dv, coords=coords)
	ds_out.attrs.update(dict(
		harmonics=HARMFILE, realistic=REALISTIC, control=CONTROL,
		h_min=H_MIN, h_max=H_MAX, nmode=NMODE, nmode_keep=NMODE_KEEP,
		min_levels=MIN_LEV, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,
		strong_pctl=STRONG_PCTL, break_isobath=BREAK_ISOBATH,
		bands=';'.join(f'{b[3]}:{b[0]}' for b in BANDS),
		windows=';'.join(f'{k}:{v[0]}..{v[1]}' for k, v in WINDOWS.items()),
		note='Pi_n and the cumulative recovered fraction are integrals over '
			 'the generation cells, those where C > 0 within a band, so the '
			 'denominator is single signed. C_n = 0.5 phi_n(-H) P_n W '
			 'cos(dphi_n), so the fac_ fields split the ratio between the '
			 'runs into ln_phi, ln_P and ln_cos, which sum to ln_tot with no '
			 'residual, averaged over each band with C_n of the realistic '
			 'run as weight. The partition needs no horizontal resolution, '
			 'so it extends onto the shelf, but whether a mode radiates is '
			 'a separate question answered in Figure 5.',
	))
	enc = {v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars}
	ds_out.to_netcdf(OUTFILE, encoding=enc)
	ds_out.close(); dh.close()

	import fig04_report
	fig04_report.main()


if __name__ == '__main__':
	main()