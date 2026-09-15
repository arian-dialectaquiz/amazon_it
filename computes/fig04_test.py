#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig04_compute.py
=================================================================
Figure 4  -  The modal partition of the generated wave
Amazon shelf internal tide manuscript, version 3

CONVERGENCE TEST MODE
---------------------
The first attempt kept two modes and produced a recovered fraction that
was 0.59 in one window and negative in another, with modal shares of
1.87 and -0.87. Two things were wrong.

  1. The share was normalised by the truncated sum of the retained
	 C_n, which is a small residual of large opposing terms and can pass
	 through zero. Pi_n is now normalised by the FULL conversion C, so
	 it is a share of something that does not vanish by construction.

  2. The convergence being checked was the wrong one. The modal
	 expansion converges in depth integrated mean square, which is what
	 the old varfrac measured and why it read 0.98. The conversion does
	 not depend on the depth integral, it depends on the single value of
	 the pressure at the bed, and there phi_n(-H) stays order one for
	 every n while p_n decays slowly. Ninety eight per cent of the
	 variance in two modes still leaves a tail that adds coherently at
	 the endpoint and over a steep slope can flip the sign.

So this version solves NMODE modes and reports the CUMULATIVE recovered
fraction against mode number, both as a ratio of band integrals and as a
pointwise median over the strongly converting cells. That decides
whether a modal partition of the conversion is measurable here at all.

Read the convergence block the run prints. If the cumulative fraction
reaches about 0.9 within four or five modes, the partition is
recoverable and Figure 4 stands as designed with the remainder reported
as the high mode tail. If it is still wandering at mode ten, the modal
partition of the generation is not measurable in this configuration and
Figure 4 should move to the modal partition of the radiated FLUX, which
is depth integrated and converges the way the variance does.

Outputs  fig04_data.nc, and the convergence table printed to stdout

Method
------
	p_n = (1/H) int_{-H}^{0} p_bc phi_n dz
	C_n = phi_n(-H) * 0.5 * (a_pn a_w + b_pn b_w)
	Sigma_N = sum_{n<=N} C_n / C          cumulative recovered fraction
	Pi_n    = C_n / C                     share of the full conversion

The harmonic coefficients come from fig03_harmonics.nc. The vertical
modes need the window mean stratification, which is cached in MEANCACHE
after the first run.
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

# progress lines appear as they happen, so a long run never looks stuck
print = functools.partial(print, flush=True)


# =====================================================================
# CONFIGURATION
# =====================================================================
HARMFILE  = 'fig03_harmonics.nc'
FIG03     = 'fig03_data.nc'        # only for a cross check on C
OUTFILE   = 'fig04_data.nc'
MEANCACHE = 'mean_state.nc'        # set to None to disable the cache

NMODE = 10                # modes solved, for the convergence test
NMODE_KEEP = 2            # modes quoted in the partition
CONVERGENCE_ONLY = True   # True prints the table and skips fig04_report,
                          # since the report is written for a partition
                          # that may not survive this test

H_MIN = 250.              # m, the resolved subdomain of Figure 1
H_MAX = 3500.             # m, outer limit

SIGMA_MIN = 0.50          # the partition is quoted only where the
SIGMA_MAX = 1.50          # cumulative recovered fraction sits in this band
STRONG_PCTL = 75.         # percentile of |C| within a band defining the
                          # strongly converting cells used for the
                          # pointwise convergence statistic

ZW_TOL = 0.05             # m, tolerance on the window mean z_w cross check

BANDS = [('upper slope, 250 to 1000 m', 250., 1000., 'slope'),
		 ('deep, 1000 to 3500 m',      1000., 3500., 'deep')]

DEPTH_BINS = np.logspace(np.log10(H_MIN), np.log10(H_MAX), 19)
T1_BINS = np.linspace(0.02, 0.90, 23)
N_SUB = 4000
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
	"""
	Modal amplitude of a field given at rho points,

		p_n = (1/H) int p_bc phi_n dz ,

	which inverts the expansion p_bc = sum_n p_n phi_n under the unit
	depth mean square normalisation of Equation (phi).
	"""
	return np.nansum(a_p*phi*dz, axis=0)/np.maximum(H, 1e-9)


def mean_state(ds, tag, wname, t0, t1):
	"""
	Window mean T, S, z_rho and z_w, cached on disk after the first run.

	The cache stores the window bounds each entry was built with, so a
	change to WINDOWS invalidates only the entries it affects.
	"""
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
	aw = area_weights(pm, pn, mask)
	awf = aw.ravel()[flat]
	dHdx, dHdy = grad_h_components(h, pm, pn)
	dhx = dHdx.ravel()[flat]; dhy = dHdy.ravel()[flat]
	hb = h.ravel()[flat]
	print(f'  grid {ny} x {nx}, {nwet} wet columns, {nz} levels')
	print(f'  solving {NMODE} baroclinic modes, {NMODE_KEEP} quoted')
	if NMODE > nz - 2:
		warnings.warn(f'{NMODE} modes on {nz} levels is beyond what the '
					  'vertical grid can carry, the high modes are grid noise')

	band_ok = wet & (h >= H_MIN) & (h <= H_MAX)
	print(f'  analysis restricted to {H_MIN:.0f} to {H_MAX:.0f} m, '
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
		   for k in ('C', 'Sigma', 'SigmaN', 'trap1', 'varfrac')
		   for t in RUNS}
	outm = {f'{k}_{t}': np.full((nwin, NMODE, ny, nx), np.nan)
			for k in ('Cn', 'Pin', 'phib') for t in RUNS}
	cn_prof = {f'phib_{t}': np.full((nwin, NMODE, DEPTH_BINS.size - 1, 3),
									np.nan) for t in RUNS}

	nband = len(BANDS)
	integ = {f'{k}_{t}': np.full((nwin, nband), np.nan)
			 for k in ('C', 'Sigma', 'SigmaN', 'area_ok') for t in RUNS}
	integm = {f'{k}_{t}': np.full((nwin, nband, NMODE), np.nan)
			  for k in ('Cn', 'Pin', 'Ccum', 'Vcum', 'Cstrong') for t in RUNS}

	rng = np.random.default_rng(SEED)
	sub = {f'{k}_{t}': np.full((nwin, N_SUB), np.nan)
		   for k in ('trap1', 'Pi1', 'h', 'phib1') for t in RUNS}
	binf = {f'{k}_{t}': np.full((nwin, T1_BINS.size - 1), np.nan)
			for k in ('med', 'q25', 'q75') for t in RUNS}
	fitp = {f'{k}_{t}': np.full(nwin, np.nan)
			for k in ('slope', 'rank') for t in RUNS}

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
							  f'the harmonics file by {d:.3g} m, which is '
							  'beyond rounding. Check that the windows match.')

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

			# ---- harmonic coefficients from the streaming pass
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

			# ---- total conversion, from the bottom rho level
			Cf = 0.5*(a_p[0]*a_w + b_p[0]*b_w)

			# ---- modal projection over NMODE modes
			Cn = np.full((NMODE, nwet), np.nan)
			phib = np.full((NMODE, nwet), np.nan)
			pvar = np.zeros((NMODE, nwet))
			T1f = np.full(nwet, np.nan)
			for n in range(1, NMODE + 1):
				phin, _ = phi_from_Phi(Phi[:, n, :], zwf, c[n])
				a_pn = project(a_p, phin, dzf, Hf)
				b_pn = project(b_p, phin, dzf, Hf)
				pb = phin[0]
				Cn[n - 1] = pb*0.5*(a_pn*a_w + b_pn*b_w)
				phib[n - 1] = pb
				pvar[n - 1] = 0.5*(a_pn**2 + b_pn**2)
				if n == 1:
					hpf, _, _ = plume_base(saltf, zrf, thresh=S_LENS)
					T1f = trapping_index(phin, zwf,
										 np.nan_to_num(hpf, nan=0.0))
					T1f = np.where(np.isfinite(hpf), T1f, np.nan)

			Ccum = np.cumsum(np.nan_to_num(Cn), axis=0)          # (NMODE, M)
			Vcum = np.cumsum(np.nan_to_num(pvar), axis=0)

			ptot = 0.5*(np.nansum(a_p**2*dzf, axis=0)
						+ np.nansum(b_p**2*dzf, axis=0))/np.maximum(Hf, 1e-9)

			with np.errstate(invalid='ignore', divide='ignore'):
				# share of the FULL conversion, so the denominator does not
				# pass through zero by construction
				Pin = np.where(np.abs(Cf) > 0, Cn/Cf[None, :], np.nan)
				Sig = np.where(np.abs(Cf) > 0,
							   Ccum[NMODE_KEEP - 1]/Cf, np.nan)
				SigN = np.where(np.abs(Cf) > 0, Ccum[-1]/Cf, np.nan)
				varfrac = np.where(ptot > 0, Vcum[NMODE_KEEP - 1]/ptot, np.nan)

			out[f'C_{tag}'][iw] = unflat(Cf)
			out[f'Sigma_{tag}'][iw] = unflat(Sig)
			out[f'SigmaN_{tag}'][iw] = unflat(SigN)
			out[f'trap1_{tag}'][iw] = unflat(T1f)
			out[f'varfrac_{tag}'][iw] = unflat(varfrac)
			for n in range(NMODE):
				outm[f'Cn_{tag}'][iw, n] = unflat(Cn[n])
				outm[f'Pin_{tag}'][iw, n] = unflat(Pin[n])
				outm[f'phib_{tag}'][iw, n] = unflat(phib[n])

			# ---- depth binned bottom amplitudes
			for n in range(NMODE):
				for kb in range(DEPTH_BINS.size - 1):
					s = (hb >= DEPTH_BINS[kb]) & (hb < DEPTH_BINS[kb + 1]) \
						& np.isfinite(phib[n])
					if s.sum() > 20:
						v = np.abs(phib[n][s])
						cn_prof[f'phib_{tag}'][iw, n, kb] = [
							np.nanmedian(v), np.nanpercentile(v, 25),
							np.nanpercentile(v, 75)]

			# ---- band integrals and the cumulative recovered fraction
			for ib, (nm, lo, hi, short) in enumerate(BANDS):
				mf = (hb >= lo) & (hb < hi)
				w = np.where(mf & np.isfinite(Cf), awf, 0.0)
				tot = float(np.nansum(np.nan_to_num(Cf)*w)/1e6)
				integ[f'C_{tag}'][iw, ib] = tot

				# strongly converting cells, for the pointwise statistic
				absC = np.abs(Cf)[mf & np.isfinite(Cf)]
				thr = np.percentile(absC, STRONG_PCTL) if absC.size else np.inf
				strong = mf & np.isfinite(Cf) & (np.abs(Cf) >= thr)

				for n in range(NMODE):
					integm[f'Cn_{tag}'][iw, ib, n] = float(
						np.nansum(np.nan_to_num(Cn[n])*w)/1e6)
					cc = float(np.nansum(np.nan_to_num(Ccum[n])*w)/1e6)
					integm[f'Ccum_{tag}'][iw, ib, n] = cc/tot if tot else np.nan
					integm[f'Pin_{tag}'][iw, ib, n] = (
						integm[f'Cn_{tag}'][iw, ib, n]/tot if tot else np.nan)
					with np.errstate(invalid='ignore', divide='ignore'):
						vv = np.where(ptot > 0, Vcum[n]/ptot, np.nan)
						ss = np.where(np.abs(Cf) > 0, Ccum[n]/Cf, np.nan)
					integm[f'Vcum_{tag}'][iw, ib, n] = float(
						np.nanmedian(vv[mf])) if mf.any() else np.nan
					integm[f'Cstrong_{tag}'][iw, ib, n] = float(
						np.nanmedian(ss[strong])) if strong.any() else np.nan

				integ[f'Sigma_{tag}'][iw, ib] = \
					integm[f'Ccum_{tag}'][iw, ib, NMODE_KEEP - 1]
				integ[f'SigmaN_{tag}'][iw, ib] = \
					integm[f'Ccum_{tag}'][iw, ib, -1]
				okm = mf & np.isfinite(Sig) & (Sig > SIGMA_MIN) \
					  & (Sig < SIGMA_MAX)
				integ[f'area_ok_{tag}'][iw, ib] = float(
					np.nansum(np.where(okm, awf, 0.))
					/ max(np.nansum(np.where(mf, awf, 0.)), 1e-9))

			# ---- the partition against the trapping index
			keep = (hb >= H_MIN) & (hb <= H_MAX) & np.isfinite(T1f) \
				   & np.isfinite(Pin[0]) & np.isfinite(Sig) \
				   & (Sig > SIGMA_MIN) & (Sig < SIGMA_MAX)
			x, y = T1f[keep], Pin[0][keep]
			_, med, q25, q75 = binned(x, y, T1_BINS)
			binf[f'med_{tag}'][iw] = med
			binf[f'q25_{tag}'][iw] = q25
			binf[f'q75_{tag}'][iw] = q75
			a_, b_, _ = linfit(x, y)
			fitp[f'slope_{tag}'][iw] = a_
			fitp[f'rank_{tag}'][iw] = rank_corr(x, y)
			if x.size:
				pick = rng.choice(x.size, size=min(N_SUB, x.size),
								  replace=False)
				sub[f'trap1_{tag}'][iw, :pick.size] = x[pick]
				sub[f'Pi1_{tag}'][iw, :pick.size] = y[pick]
				sub[f'h_{tag}'][iw, :pick.size] = hb[keep][pick]
				sub[f'phib1_{tag}'][iw, :pick.size] = phib[0][keep][pick]
			print(f'    Pi_1 on T1, slope {a_:+.3f}, rank '
				  f'{fitp[f"rank_{tag}"][iw]:+.2f}, {x.size} columns')

	# -----------------------------------------------------------------
	# cross check the total conversion against Figure 3
	# -----------------------------------------------------------------
	if os.path.exists(FIG03):
		d3 = xr.open_dataset(FIG03)
		for tag in RUNS:
			a = out[f'C_{tag}']
			b = d3[f'C_{tag}'].values
			den = np.nanmax(np.abs(b))
			err = np.nanmax(np.abs(a - b))/max(den, 1e-30)
			print(f'  cross check {tag}: C against {FIG03}, max relative '
				  f'difference {err:.2e}')
			if err > 1e-6:
				warnings.warn(f'C disagrees with {FIG03} in {tag}')
		d3.close()

	# =================================================================
	# THE CONVERGENCE TABLE, the point of this run
	# =================================================================
	line = '=' * 78
	print('\n' + line)
	print('MODAL CONVERGENCE OF THE CONVERSION')
	print(line)
	print('  Cumulative recovered fraction of C against mode number.')
	print('  Three statistics per row.')
	print('    band     ratio of band integrals, sum_{n<=N} int C_n / int C')
	print(f'    strong   pointwise median over cells above the '
		  f'{STRONG_PCTL:.0f}th percentile of |C| in the band')
	print('    var      median cumulative share of the depth mean square')
	print('             of the baroclinic pressure, for comparison')
	print('  A partition of the conversion is usable if the band row')
	print('  settles near one within a few modes. If it wanders while the')
	print('  var row sits near one, the expansion converges in the depth')
	print('  integral and not at the bed, and Figure 4 must move to the')
	print('  modal partition of the radiated flux.')
	print(line)

	hdr = f'{"window":>11} {"band":>7} {"run":>5} {"stat":>7} ' + \
		  ' '.join(f'{n+1:>7}' for n in range(NMODE))
	for ib, (nm, lo, hi, short) in enumerate(BANDS):
		print(f'\n  {nm}')
		print('  ' + hdr)
		for iw, wname in enumerate(wins):
			for tag in RUNS:
				for stat, key in (('band', 'Ccum'), ('strong', 'Cstrong'),
								  ('var', 'Vcum')):
					v = integm[f'{key}_{tag}'][iw, ib]
					row = (f'{wname:>11} {short:>7} {tag.upper():>5} '
						   f'{stat:>7} ')
					row += ' '.join(f'{x:7.2f}' if np.isfinite(x) else
									'    n/a' for x in v)
					print('  ' + row)
			print('  ' + '.' * 74)
	print(line + '\n')

	# -----------------------------------------------------------------
	# mask the stored maps to the resolved band
	# -----------------------------------------------------------------
	keep_map = band_ok[None, :, :]
	for k in list(out):
		out[k] = np.where(keep_map, out[k], np.nan)
	for k in list(outm):
		outm[k] = np.where(keep_map[:, None, :, :], outm[k], np.nan)

	# -----------------------------------------------------------------
	# write
	# -----------------------------------------------------------------
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
			   for k, v in cn_prof.items()})
	dv.update({f'int_{k}': (('window', 'band'), v) for k, v in integ.items()})
	dv.update({f'int_{k}': (('window', 'band', 'mode'), v)
			   for k, v in integm.items()})
	dv.update({f'bin_{k}': (('window', 't1_bin'), v)
			   for k, v in binf.items()})
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
		sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,
		strong_pctl=STRONG_PCTL, break_isobath=BREAK_ISOBATH,
		bands=';'.join(f'{b[3]}:{b[0]}' for b in BANDS),
		windows=';'.join(f'{k}:{v[0]}..{v[1]}' for k, v in WINDOWS.items()),
		note='C_n = phi_n(-H) <p_n w_bt(-H)> with p_n the projection of '
			 'p_bc onto phi_n. Pi_n is C_n over the FULL conversion C, not '
			 'over the truncated sum. int_Ccum is the cumulative recovered '
			 'fraction as a ratio of band integrals, int_Cstrong the same '
			 'pointwise over the strongly converting cells, and int_Vcum '
			 'the cumulative share of the depth mean square of the '
			 'baroclinic pressure. Sigma is int_Ccum at nmode_keep and '
			 'SigmaN at nmode.',
	))
	enc = {v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars}
	ds_out.to_netcdf(OUTFILE, encoding=enc)
	ds_out.close()
	dh.close()

	if CONVERGENCE_ONLY:
		print('CONVERGENCE_ONLY is set, fig04_report.py was not called. '
			  'Read the table above before deciding what Figure 4 becomes.')
	else:
		import fig04_report
		fig04_report.main()


if __name__ == '__main__':
	main()