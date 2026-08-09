"""
fig07_compute.py
=================================================================
Figure 7  -  Synthesis, a river driven source to sink chain
Amazon shelf internal tide manuscript

Computes and stores everything the panels need:

  (a) regime diagram, shelf mode one phase speed against the positive
	  shelf conversion, one point per window, per spring neap phase and
	  per run, coloured by the total domain dissipation
  (b) the numbers annotated on the schematic, so that the drawing and
	  the tables cannot drift apart

Output:  fig07_data.nc   (read by fig07_plot.py)

Method
------
Nothing is recomputed from the model here. Every quantity has already
been formed by fig02 and fig03 and the point of this script is to put
them on one axis with a single provenance. The block resolved shelf
conversion of fig03 is partitioned by the spring neap phase, which is
identified from the Hilbert envelope of the bandpassed regional sea
surface height exactly as in the methods, and averaged within each
hydrological window. The mode one phase speed is the outer shelf median
of fig02, which is a window mean quantity and therefore common to the
two phases of a window.

The names of the variables carried by the earlier files are declared in
the adapter block below. When one of them is missing the script prints
what the file does contain and stops, rather than guessing.
=================================================================
"""

import os
import warnings
import numpy as np
import xarray as xr

from fig01_compute import WINDOWS


# =====================================================================
# CONFIGURATION
# =====================================================================
OUTFILE = 'fig07_data.nc'
FIG02FILE = 'fig02_data.nc'
FIG03FILE = 'fig03_data.nc'

RUNS = ('ref', 'ctrl')
PHASES = ('spring', 'neap')

# ---- adapter, the names these quantities carry in the earlier files.
# Edit the right hand sides to match, the script checks them and stops
# with the file contents listed when one is absent.
V02 = dict(
	c1_outer='c1_outer_{tag}',        # (window,) outer shelf median c1
	trapping='trapping_{tag}',        # (window,) surface trapping index
	Reff='Reff_{tag}',                # (window,) effective radius, km
)
V03 = dict(
	Cpos_blk='Cpos_shelf_blk_{tag}',  # (block,) positive shelf conversion
	env='env_{tag}',                  # (block,) semidiurnal SSH envelope
	block_time='block_time',          # (block,)
	Cpos_win='Cpos_shelf_{tag}',      # (window,) integrated, MW
	Ddom_win='D_domain_{tag}',        # (window,) domain dissipation, MW
	Fbreak_win='F_break_{tag}',       # (window,) flux across 200 m, MW
	Ddeep_win='D_deep_{tag}',         # (window,) deep dissipation, MW
)

# Values already reported in the manuscript, used only to annotate the
# schematic and to check the files against the tables. Order is rising,
# peak, decreasing.
TAB_CPOS_RATIO = np.array([1.19, 1.32, 1.52])     # REF over CTRL, shelf
TAB_ONSHORE_MAX = 0.14                            # onshore share, modes 1-3
TAB_DDOM_RATIO = np.array([1.22, 1.42, 1.80])     # REF over CTRL, domain
TAB_FBREAK_RATIO = np.array([1.15, 1.31, 1.61])   # REF over CTRL, 200 m
TAB_DEEP_IMPORT = np.array([1.085, 0.544, 0.987])  # GW, REF
CHECK_TOL = 0.06                                  # relative, tables vs files


# =====================================================================
# HELPERS
# =====================================================================
def need(ds, name, path):
	"""Fetch a variable or stop with the file contents listed."""
	if name not in ds:
		print(f'\n{path} carries:')
		for v in sorted(ds.data_vars):
			print(f'   {v}  {tuple(ds[v].dims)}')
		raise SystemExit(f'\n{path} has no variable {name!r}. Edit the '
						 'adapter block at the top of fig07_compute.py '
						 'so the names match.')
	return ds[name].values


def phase_mask(env, tblk, wblk, k):
	"""
	Spring and neap blocks within window k, split at the median of the
	semidiurnal envelope inside that window so that the partition is
	local to the window rather than to the record.
	"""
	m = (wblk == k) & np.isfinite(env)
	if m.sum() < 4:
		return m & False, m & False
	cut = np.nanmedian(env[m])
	return m & (env >= cut), m & (env < cut)


def window_index_of(tblk):
	"""Window index per block, from the WINDOWS bounds of fig01."""
	wins = list(WINDOWS.keys())
	w = np.full(tblk.size, -1, dtype=int)
	for k, name in enumerate(wins):
		t0, t1 = WINDOWS[name]
		w[(tblk >= np.datetime64(t0)) & (tblk <= np.datetime64(t1))] = k
	return w, wins


def check_against_tables(name, got, want, tol=CHECK_TOL):
	"""Warn when a file disagrees with the number printed in the paper."""
	got = np.asarray(got, dtype=float)
	want = np.asarray(want, dtype=float)
	if got.shape != want.shape:
		warnings.warn(f'{name}: shape {got.shape} against {want.shape}, '
					  'not checked')
		return
	with np.errstate(invalid='ignore', divide='ignore'):
		rel = np.abs(got - want)/np.maximum(np.abs(want), 1e-12)
	bad = rel > tol
	if np.any(bad):
		warnings.warn(f'{name} differs from the manuscript table by more '
					  f'than {100*tol:.0f}% in {int(bad.sum())} window(s), '
					  f'file {got[bad]}, table {want[bad]}')
	else:
		print(f'  {name} agrees with the table to within '
			  f'{100*np.nanmax(rel):.1f}%')


# =====================================================================
# ASSEMBLY
# =====================================================================
def main():
	for p in (FIG02FILE, FIG03FILE):
		if not os.path.exists(p):
			raise SystemExit(f'{p} not found, run the earlier figures first')

	d2 = xr.open_dataset(FIG02FILE)
	d3 = xr.open_dataset(FIG03FILE)

	tblk = need(d3, V03['block_time'], FIG03FILE)
	wblk, wins = window_index_of(tblk)
	nwin = len(wins)
	print(f'  {tblk.size} blocks, {nwin} windows, '
		  f'{int((wblk >= 0).sum())} blocks inside a window')

	c1 = np.full((len(RUNS), nwin), np.nan)
	trap = np.full((len(RUNS), nwin), np.nan)
	reff = np.full((len(RUNS), nwin), np.nan)
	Cpos = np.full((len(RUNS), nwin, len(PHASES)), np.nan)
	Cpos_sd = np.full((len(RUNS), nwin, len(PHASES)), np.nan)
	Cpos_win = np.full((len(RUNS), nwin), np.nan)
	Ddom = np.full((len(RUNS), nwin), np.nan)
	Fbrk = np.full((len(RUNS), nwin), np.nan)
	Ddeep = np.full((len(RUNS), nwin), np.nan)

	for it, tag in enumerate(RUNS):
		c1[it] = need(d2, V02['c1_outer'].format(tag=tag), FIG02FILE)
		trap[it] = need(d2, V02['trapping'].format(tag=tag), FIG02FILE)
		reff[it] = need(d2, V02['Reff'].format(tag=tag), FIG02FILE)

		cb = need(d3, V03['Cpos_blk'].format(tag=tag), FIG03FILE)
		env = need(d3, V03['env'].format(tag=tag), FIG03FILE)
		Cpos_win[it] = need(d3, V03['Cpos_win'].format(tag=tag), FIG03FILE)
		Ddom[it] = need(d3, V03['Ddom_win'].format(tag=tag), FIG03FILE)
		Fbrk[it] = need(d3, V03['Fbreak_win'].format(tag=tag), FIG03FILE)
		Ddeep[it] = need(d3, V03['Ddeep_win'].format(tag=tag), FIG03FILE)

		for k in range(nwin):
			ms, mn = phase_mask(env, tblk, wblk, k)
			for ip, m in enumerate((ms, mn)):
				if m.sum() == 0:
					continue
				Cpos[it, k, ip] = np.nanmean(cb[m])
				Cpos_sd[it, k, ip] = np.nanstd(cb[m])

	with np.errstate(invalid='ignore', divide='ignore'):
		ratio_C = Cpos_win[0]/Cpos_win[1]
		ratio_D = Ddom[0]/Ddom[1]
		ratio_F = np.abs(Fbrk[0])/np.abs(Fbrk[1])
	check_against_tables('shelf positive conversion ratio', ratio_C,
						 TAB_CPOS_RATIO)
	check_against_tables('domain dissipation ratio', ratio_D, TAB_DDOM_RATIO)
	check_against_tables('break crossing flux ratio', ratio_F,
						 TAB_FBREAK_RATIO)

	ds_out = xr.Dataset(
		dict(
			c1_outer=(('run', 'window'), c1),
			trapping=(('run', 'window'), trap),
			Reff=(('run', 'window'), reff),
			Cpos_phase=(('run', 'window', 'phase'), Cpos),
			Cpos_phase_sd=(('run', 'window', 'phase'), Cpos_sd),
			Cpos_window=(('run', 'window'), Cpos_win),
			D_domain=(('run', 'window'), Ddom),
			F_break=(('run', 'window'), Fbrk),
			D_deep=(('run', 'window'), Ddeep),
			ratio_Cpos=('window', ratio_C),
			ratio_Ddomain=('window', ratio_D),
			ratio_Fbreak=('window', ratio_F),
			onshore_max=((), np.float64(TAB_ONSHORE_MAX)),
			deep_import_GW=('window', TAB_DEEP_IMPORT),
		),
		coords=dict(run=('run', list(RUNS)),
					window=('window', wins),
					phase=('phase', list(PHASES))),
	)
	ds_out.attrs.update(dict(
		source_fig02=FIG02FILE, source_fig03=FIG03FILE,
		windows=';'.join(f'{k}:{v[0]}..{v[1]}' for k, v in WINDOWS.items()),
		note='Cpos_phase is the block mean positive shelf conversion in '
			 'MW within each window, split at the median semidiurnal sea '
			 'surface height envelope of that window. c1_outer is the '
			 'window mean outer shelf median mode one phase speed and is '
			 'therefore common to the two phases. Nothing here is '
			 'recomputed from the model, the file exists so that the '
			 'synthesis and the tables share one provenance.',
	))
	print('writing', OUTFILE)
	ds_out.to_netcdf(OUTFILE)
	summary(ds_out, wins)
	d2.close()
	d3.close()


def summary(ds, wins):
	print('\n--- regime diagram points ---')
	for it, tag in enumerate(RUNS):
		for k, w in enumerate(wins):
			c = float(ds.c1_outer[it, k])
			for ip, ph in enumerate(PHASES):
				y = float(ds.Cpos_phase[it, k, ip])
				print(f'{w:>11} {tag:>5} {ph:>6}  c1 {c:5.2f} m/s   '
					  f'positive shelf C {y:8.1f} MW   '
					  f'domain D {float(ds.D_domain[it, k]):8.1f} MW')
	print('\n--- ratios carried onto the schematic ---')
	for k, w in enumerate(wins):
		print(f'{w:>11}  shelf conversion {float(ds.ratio_Cpos[k]):4.2f}   '
			  f'break flux {float(ds.ratio_Fbreak[k]):4.2f}   '
			  f'domain dissipation {float(ds.ratio_Ddomain[k]):4.2f}')


if __name__ == '__main__':
	main()