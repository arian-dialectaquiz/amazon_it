#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig04_plot.py
=================================================================
Figure 4  -  The modal partition of the generated wave
Amazon shelf internal tide manuscript, version 3

Flat script. Every panel has its own axes handle.

	axa  gs[0, 0:2]   conversion into mode one, REF
	axb  gs[0, 2:4]   conversion into mode two, REF
	axc  gs[0, 4:6]   difference between the runs in the mode one share
	axd  gs[1, 0:2]   cumulative recovered fraction against mode number
	axe  gs[1, 2:4]   modal shares by band, window and run
	axf  gs[1, 4:6]   per mode factorisation of the ratio between the runs

Panel (f) has been through two earlier forms. The mode one share against
the trapping index carried no relation, rank correlations of +0.02 to
+0.37, and the bare ratio of the modal conversion said what changed
without saying why. What it shows now is the factorisation

	C_n = 0.5 phi_n(-H) P_n W cos(dphi_n)

so that the ratio between the runs splits into the structure function at
the bed, the amplitude of the modal pressure and the alignment, the
three multiplying to the total with no residual.

Conventions

	the EXPERIMENT is carried by the LINE STYLE, REF solid and CTRL
	dashed, and the WINDOW by the COLOUR, except in (d) where the colour
	carries the DEPTH BAND, in (e) where it carries the MODE, and in (f)
	where it carries the FACTOR. The caption says so.

Reads fig04_data.nc written by fig04_compute.py.
=================================================================
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
	'font.size': 11, 'axes.labelsize': 11.5, 'axes.titlesize': 11.5,
	'legend.fontsize': 9.5, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
	'axes.linewidth': 0.9,
})

# ---------------------------------------------------------------------
DATA   = 'fig04_data.nc'
OUTFIG = 'Fig4.jpeg'
DPI    = 300
WIN    = 'peak'
F_BAND = 'shelf'            # band drawn in panel (f)

COL = dict(
	ref='navy', ctrl='orangered',
	rising='steelblue', peak='darkgoldenrod', decreasing='seagreen',
	mode1='midnightblue', mode2='seagreen', mode3='darkgoldenrod',
	mode4='firebrick',
	band_shelf='midnightblue', band_slope='seagreen', band_deep='firebrick',
	f_total='black', f_phi='darkorange', f_P='teal', f_align='orchid',
	land='wheat', coast='grey', isobath='black', isobath_thin='dimgrey',
	guide='dimgrey', unity='crimson',
)

LS = dict(ref='-', ctrl='--', guide=':')
LW = dict(main=2.2, secondary=1.4, faint=0.9, guide=1.1)

CMAP_C  = 'RdBu_r'
CMAP_DP = 'PuOr_r'

CN_LIM  = None
DPI_LIM = 0.30
TICK_FS = 8.5
LET_FS = 16

ds = xr.open_dataset(DATA)

lon  = ds.lon_rho.values
lat  = ds.lat_rho.values
mask = ds.mask_rho.values
h    = np.where(mask > 0, ds.h.values, np.nan)
wins = [str(w) for w in ds.window.values]
bands = [str(b) for b in ds.band.values]
modes = [int(m) for m in ds.mode.values]
iw   = wins.index(WIN)
NKEEP = int(ds.attrs.get('nmode_keep', 4))
BREAK = float(ds.attrs.get('break_isobath', 200.))
SMIN = float(ds.attrs.get('sigma_min', 0.6))
SMAX = float(ds.attrs.get('sigma_max', 1.4))
blabel = dict(x.split(':', 1) for x in
			  ds.attrs.get('bands', '').split(';') if ':' in x)

WCOL = {w: COL.get(w, 'grey') for w in wins}
BCOL = {b: COL.get(f'band_{b}', 'grey') for b in bands}
MCOL = [COL['mode1'], COL['mode2'], COL['mode3'], COL['mode4']]
land = cfeature.NaturalEarthFeature('physical', 'land', scale='10m')


def dress(ax, labels_left=True):
	ax.set_aspect('auto', adjustable='box')
	ax.add_feature(land, facecolor=COL['land'], edgecolor=COL['coast'],
				   lw=0.4, zorder=8)
	ax.coastlines(resolution='10m', linewidths=0.4, zorder=9)
	ax.contour(lon, lat, h, levels=[BREAK], colors=COL['isobath'],
			   linewidths=1.0, zorder=6)
	ax.contour(lon, lat, h, levels=[250, 1000, 2000],
			   colors=COL['isobath_thin'], linewidths=0.5, zorder=6)
	ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()],
				  crs=ccrs.PlateCarree())
	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.22,
					  lw=0.4, color='grey', ls='--', zorder=10)
	gl.top_labels = False; gl.right_labels = False
	gl.left_labels = labels_left
	gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': TICK_FS}; gl.ylabel_style = {'size': TICK_FS}
	return gl


def panel_letter(ax, s, on_map=False):
	kw = dict(fontsize=LET_FS, fontweight='bold', va='top', ha='left')
	if on_map:
		ax.text(0.022, 0.972, s, transform=ax.transAxes, zorder=28,
				bbox=dict(fc='w', ec='0.35', boxstyle='round,pad=0.20',
						  alpha=0.95), **kw)
	else:
		ax.text(0.008, 0.965, s, transform=ax.transAxes, zorder=28, **kw)


def cbar_inside(ax, mappable, label, extend='both',
				anchor=(0.035, 0.1, 1, 1), width='60%', height='3.2%'):
	cax = inset_axes(ax, width=width, height=height, loc='lower left',
					 bbox_to_anchor=anchor, bbox_transform=ax.transAxes,
					 borderpad=0)
	cb = plt.colorbar(mappable, cax=cax, orientation='horizontal',
					  extend=extend)
	cb.set_label(label, fontsize=11, labelpad=1.5)
	cb.ax.tick_params(labelsize=11, length=2.5, pad=1.5)
	cb.outline.set_linewidth(0.7)
	cax.set_facecolor('w'); cax.patch.set_alpha(0.85); cax.set_zorder(30)
	return cb


def legend_above(ax, handles, ncol, y=1.015, x=0.0):
	return ax.legend(handles=handles, loc='lower left',
					 bbox_to_anchor=(x, y), ncol=ncol, frameon=False,
					 fontsize=9, handlelength=2.2, columnspacing=1.4,
					 borderaxespad=0.0)


# =====================================================================
# FIGURE SKELETON
# =====================================================================
fig = plt.figure(figsize=(17.4, 11.6))
gs = gridspec.GridSpec(2, 6, figure=fig,
					   height_ratios=[1.00, 0.82],
					   hspace=0.2, wspace=0.48,
					   left=0.050, right=0.975, top=0.945, bottom=0.070)

axa = fig.add_subplot(gs[0, 0:2], projection=ccrs.PlateCarree())
axb = fig.add_subplot(gs[0, 2:4], projection=ccrs.PlateCarree())
axc = fig.add_subplot(gs[0, 4:6], projection=ccrs.PlateCarree())
axd = fig.add_subplot(gs[1, 0:2])
axe = fig.add_subplot(gs[1, 2:4])
axf = fig.add_subplot(gs[1, 4:6])


# =====================================================================
# (a) (b)  CONVERSION INTO MODES ONE AND TWO
# =====================================================================
C1 = ds.Cn_ref.isel(window=iw, mode=0).values
C2 = ds.Cn_ref.isel(window=iw, mode=1).values

lim = CN_LIM
if lim is None:
	q = np.abs(C1[np.isfinite(C1)])
	lim = float(np.nanpercentile(q, 98)) if q.size else 1e-2
	lim = max(lim, 1e-4)
norm_C = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

for ax, fld, tag, nn in ((axa, C1, 'a', 1), (axb, C2, 'b', 2)):
	pcm = ax.pcolormesh(lon, lat, fld, cmap=CMAP_C, norm=norm_C,
						shading='auto', zorder=1, rasterized=True)
	dress(ax, labels_left=(tag == 'a'))
	ax.set_title(rf'$C_{nn}$, conversion into mode {nn}, REF, {WIN}',
				 fontsize=10.5, pad=5)
	panel_letter(ax, tag, on_map=True)
	if tag == 'a':
		cbar_inside(ax, pcm, r'$C_n$ (W m$^{-2}$)', extend='both')


# =====================================================================
# (c)  DIFFERENCE BETWEEN THE RUNS IN THE MODE ONE SHARE
# =====================================================================
dPi = (ds.Pin_ref.isel(window=iw, mode=0).values
	   - ds.Pin_ctrl.isel(window=iw, mode=0).values)
good = (ds.C_ref.isel(window=iw).values > 0) \
	   & (ds.C_ctrl.isel(window=iw).values > 0) \
	   & (ds.Sigma_ref.isel(window=iw).values > SMIN) \
	   & (ds.Sigma_ref.isel(window=iw).values < SMAX) \
	   & (ds.Sigma_ctrl.isel(window=iw).values > SMIN) \
	   & (ds.Sigma_ctrl.isel(window=iw).values < SMAX)
dPi = np.where(good, dPi, np.nan)

pcc = axc.pcolormesh(lon, lat, dPi, cmap=CMAP_DP,
					 norm=mcolors.TwoSlopeNorm(vmin=-DPI_LIM, vcenter=0,
											   vmax=DPI_LIM),
					 shading='auto', zorder=1, rasterized=True)
dress(axc, labels_left=False)
axc.set_title(r'$\Delta\Pi_1$, REF minus CTRL, ' + WIN, fontsize=10.5, pad=5)
cbar_inside(axc, pcc, r'$\Delta\Pi_1$', extend='both')
panel_letter(axc, 'c', on_map=True)
axc.text(0.975, 0.965,
		 f'generating cells with\n${SMIN:.1f}<\\Sigma<{SMAX:.1f}$ in both',
		 transform=axc.transAxes, fontsize=8.5, va='top', ha='right',
		 color='0.12', zorder=29, linespacing=1.35,
		 bbox=dict(fc='w', ec='0.5', boxstyle='round,pad=0.28', alpha=0.9))


# =====================================================================
# (d)  CUMULATIVE RECOVERED FRACTION AGAINST MODE NUMBER
# =====================================================================
mn = np.arange(1, len(modes) + 1)
axd.axhline(1.0, color=COL['unity'], lw=1.3, ls='--', zorder=3)
axd.axvline(NKEEP, color=COL['guide'], lw=1.1, ls=LS['guide'], zorder=3)

for ib, b in enumerate(bands):
	for tag in ('ref', 'ctrl'):
		v = ds[f'int_Scum_{tag}'].isel(window=iw, band=ib).values
		axd.plot(mn, v, color=BCOL[b], ls=LS[tag],
				 lw=LW['main'] if tag == 'ref' else LW['secondary'],
				 marker='o' if tag == 'ref' else None, ms=4, zorder=4)

axd.set_xlabel('mode number')
axd.set_ylabel(r'cumulative $\Sigma_N$')
axd.set_xlim(0.7, len(modes) + 0.3)
axd.set_ylim(-0.55, 1.25)
axd.set_xticks(mn)
axd.grid(alpha=0.25, lw=0.4)
panel_letter(axd, 'd')
axd.text(NKEEP + 0.15, 0.02, f'{NKEEP} modes quoted', fontsize=8.5,
		 color='0.35', rotation=90, va='bottom',
		 transform=axd.get_xaxis_transform())

hd = [Line2D([], [], color=BCOL[b], ls=LS['ref'], lw=LW['main'],
			 label=blabel.get(b, b).split(',')[0]) for b in bands]
hd += [Line2D([], [], color=COL['guide'], ls=LS['ref'], lw=LW['main'],
			  label='REF'),
	   Line2D([], [], color=COL['guide'], ls=LS['ctrl'], lw=LW['secondary'],
			  label='CTRL')]
legend_above(axd, hd, ncol=3)
axd.text(0.985, 0.04, f'{WIN} discharge', transform=axd.transAxes,
		 ha='right', va='bottom', fontsize=8.5, color='0.35')


# =====================================================================
# (e)  MODAL SHARES BY BAND, WINDOW AND RUN
# =====================================================================
xs, labs = [], []
x = 0
for ib, b in enumerate(bands):
	x0 = x
	for k, w in enumerate(wins):
		for tag, hatch in (('ref', ''), ('ctrl', '///')):
			base = 0.0
			for n in range(NKEEP):
				v = float(ds[f'int_Pin_{tag}'][k, ib, n].values)
				axe.bar(x, v, bottom=base, width=0.82, color=MCOL[n],
						hatch=hatch, edgecolor='k', lw=0.35)
				base += v
			xs.append(x); labs.append(f'{w[:3]}\n{"R" if tag == "ref" else "C"}')
			x += 1
		x += 0.45
	axe.text(0.5*(x0 + x - 1.45), 1.10, blabel.get(b, b).split(',')[0],
			 fontsize=9, ha='center', color='0.25', fontweight='bold')
	if ib < len(bands) - 1:
		axe.axvline(x - 0.15, color='0.5', lw=0.9, ls=':')
	x += 0.7

axe.axhline(1.0, color=COL['unity'], lw=1.1, ls='--')
axe.axhline(0.0, color='k', lw=0.9)
axe.set_xticks(xs); axe.set_xticklabels(labs, fontsize=7)
axe.set_ylabel(r'share of $\int_{\rm gen} C$')
axe.set_ylim(-0.55, 1.22)
axe.grid(alpha=0.22, lw=0.4, axis='y')
panel_letter(axe, 'e')

he = [Patch(fc=MCOL[n], ec='k', lw=0.35, label=f'mode {modes[n]}')
	  for n in range(NKEEP)]
he += [Patch(fc='w', ec='k', lw=0.35, hatch='///', label='hatched, CTRL')]
legend_above(axe, he, ncol=5)


# =====================================================================
# (f)  PER MODE FACTORISATION OF THE RATIO BETWEEN THE RUNS
# =====================================================================
ibf = bands.index(F_BAND) if F_BAND in bands else 0
mk = np.arange(1, NKEEP + 1)
axf.axhline(1.0, color=COL['unity'], lw=1.3, ls='--', zorder=3)

FACS = (('ln_tot', 'f_total', 'total'),
		('ln_phi', 'f_phi', r'$\varphi_n(-H)$'),
		('ln_P', 'f_P', r'$P_n$'),
		('ln_cos', 'f_align', r'$\cos\Delta\phi_n$'))

for key, ckey, lab in FACS:
	v = np.exp(ds[f'fac_{key}'][:, ibf, :NKEEP].values)     # (window, mode)
	med = np.nanmedian(v, axis=0)
	lo = np.nanmin(v, axis=0); hi = np.nanmax(v, axis=0)
	axf.fill_between(mk, lo, hi, color=COL[ckey], alpha=0.16, lw=0, zorder=2)
	axf.plot(mk, med, color=COL[ckey],
			 lw=LW['main'] if key == 'ln_tot' else LW['secondary'],
			 marker='o' if key == 'ln_tot' else 's', ms=5, zorder=4)

axf.set_yscale('log')
axf.set_xlabel('mode number')
axf.set_ylabel('factor, REF over CTRL')
axf.set_xlim(0.7, NKEEP + 0.3)
axf.set_xticks(mk)
axf.grid(alpha=0.25, which='both', lw=0.4)
panel_letter(axf, 'f')

hf = [Line2D([], [], color=COL[c], lw=LW['main'] if k == 'ln_tot'
			 else LW['secondary'], marker='o' if k == 'ln_tot' else 's',
			 ms=5, label=l) for k, c, l in FACS]

legend_above(axf, hf, ncol=4)
axf.text(0.985, 0.05,
		 f'{blabel.get(F_BAND, F_BAND).split(",")[0]}, median over the three\n'
		 'windows, range shaded',
		 transform=axf.transAxes, ha='right', va='bottom', fontsize=8.5,
		 color='0.35', linespacing=1.3)


# =====================================================================
fig.savefig(OUTFIG, dpi=DPI, bbox_inches='tight')
print('wrote', OUTFIG)