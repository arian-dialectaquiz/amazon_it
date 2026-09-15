#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig03_plot.py
=================================================================
Figure 3  -  Conversion, its amplitude and its alignment
Amazon shelf internal tide manuscript, version 3

Flat script. Every panel has its own axes handle and every element is
drawn by its own call.

	axa  gs[0, 0:2]   C, rising discharge, REF
	axb  gs[0, 2:4]   C, peak discharge, REF
	axc  gs[0, 4:6]   C, decreasing discharge, REF
	axd  gs[1, 0:2]   total log ratio between the runs, peak
	axe  gs[1, 2:4]   its amplitude term, same scale
	axf  gs[1, 4:6]   its alignment term, same scale
	axg  gs[2, 0:3]   alignment against criticality along the break
	axh  gs[2, 3:6]   bottom pressure amplitude against N_b squared
	axi1 axi2 axi3    signed decomposition of C by depth band, with kappa

Panels (d), (e) and (f) share one colour scale on purpose. The sum of
(e) and (f) is (d) cell by cell, so the reader sees which of the two
carries the anomaly without doing arithmetic.

Panel (h) is rebinned here from the along break arrays stored in
fig03_data.nc, over the range of N_b squared that the break actually
occupies, so no bin is spent on an empty decade. Setting REBIN_H to
False falls back to the bins formed in fig03_compute.py.

Conventions, shared with the rest of the figure set

	the EXPERIMENT is carried by the LINE STYLE, REF solid and CTRL
	dashed, and the WINDOW by the COLOUR

Every colour is a named matplotlib colour in the block below.

Reads fig03_data.nc written by fig03_compute.py.
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
DATA   = 'fig03_data.nc'
OUTFIG = 'Fig3.jpeg'
DPI    = 300
WIN    = 'peak'

# ---- colours, all named
COL = dict(
	ref='navy',
	ctrl='orangered',
	rising='steelblue',
	peak='darkgoldenrod',
	decreasing='seagreen',
	land='wheat',
	coast='grey',
	isobath='black',
	isobath_thin='dimgrey',
	positive='steelblue',
	negative='indianred',
	net='black',
	guide='dimgrey',
	critical='crimson',
	subcritical='palegreen',
)

LS = dict(ref='-', ctrl='--', guide=':')
LW = dict(main=2.2, secondary=1.4, faint=0.9, guide=1.1)

CMAP_C  = 'RdBu_r'          # signed conversion
CMAP_LN = 'PuOr_r'          # the log ratio and its two terms

C_LIM  = 2.0e-2             # W m-2, half range of the conversion scale
LN_LIM = None               # None takes the 98th percentile of the total

# ---- panel (g)
SHOW_RANK_CORR = True       # annotate the rank correlation of cos on alpha

# ---- panel (h)
NB_XLIM = (0.5e-4, 6.0e-4)  # s-2, x limits
REBIN_H = True              # rebin here over the range that has data
NB_NBIN = 18                # bins used when REBIN_H is True
NB_PCTL = (2, 98)           # percentile range of N_b^2 the bins span
H_IMPEDANCE = True         # True plots P/W, the response per unit forcing

TICK_FS = 8.5
LET_FS = 16

ds = xr.open_dataset(DATA)

lon  = ds.lon_rho.values
lat  = ds.lat_rho.values
mask = ds.mask_rho.values
hraw = ds.h.values
h    = np.where(mask > 0, hraw, np.nan)
wet  = mask > 0
wins = [str(w) for w in ds.window.values]
bands = [str(b) for b in ds.band.values]
iw   = wins.index(WIN)
BREAK = float(ds.attrs.get('break_isobath', 200.))
blabel = dict(x.split(':', 1) for x in
			  ds.attrs.get('bands', '').split(';') if ':' in x)

WCOL = {w: COL.get(w, 'grey') for w in wins}
land = cfeature.NaturalEarthFeature('physical', 'land', scale='10m')


# =====================================================================
# HELPERS
# =====================================================================
def dress(ax, labels_left=True, labels_bottom=True):
	ax.set_aspect('auto', adjustable='box')
	ax.add_feature(land, facecolor=COL['land'], edgecolor=COL['coast'],
				   lw=0.4, zorder=8)
	ax.coastlines(resolution='10m', linewidths=0.4, zorder=9)
	ax.contour(lon, lat, h, levels=[BREAK], colors=COL['isobath'],
			   linewidths=1.0, zorder=6)
	ax.contour(lon, lat, h, levels=[50, 1000, 2000],
			   colors=COL['isobath_thin'], linewidths=0.4, zorder=6)
	ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()],
				  crs=ccrs.PlateCarree())
	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.22,
					  lw=0.4, color='grey', ls='--', zorder=10)
	gl.top_labels = False; gl.right_labels = False
	gl.left_labels = labels_left; gl.bottom_labels = labels_bottom
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
					 fontsize=9.5, handlelength=2.2, columnspacing=1.6,
					 borderaxespad=0.0)


def binned_median(x, y, bins, geometric=True):
	"""Median and quartiles of y within bins of x."""
	ctr = np.sqrt(bins[1:]*bins[:-1]) if geometric \
		else 0.5*(bins[1:] + bins[:-1])
	med = np.full(ctr.size, np.nan)
	q25 = np.full(ctr.size, np.nan)
	q75 = np.full(ctr.size, np.nan)
	ok = np.isfinite(x) & np.isfinite(y)
	for k in range(ctr.size):
		s = ok & (x >= bins[k]) & (x < bins[k + 1])
		if s.sum() > 6:
			med[k] = np.nanmedian(y[s])
			q25[k] = np.nanpercentile(y[s], 25)
			q75[k] = np.nanpercentile(y[s], 75)
	return ctr, med, q25, q75


def rank_corr(x, y):
	"""
	Spearman rank correlation, computed with numpy alone.

	The Pearson coefficient is a poor statistic for the along break
	scatter of the alignment, which rises with criticality through a
	curve rather than a straight line, so the rank version is the one
	quoted on the panel.
	"""
	x = np.asarray(x, float); y = np.asarray(y, float)
	ok = np.isfinite(x) & np.isfinite(y)
	if ok.sum() < 20:
		return np.nan
	def ranks(v):
		order = np.argsort(v, kind='mergesort')
		r = np.empty(v.size, float)
		r[order] = np.arange(v.size, dtype=float)
		return r
	rx, ry = ranks(x[ok]), ranks(y[ok])
	return float(np.corrcoef(rx, ry)[0, 1])


# =====================================================================
# FIGURE SKELETON
# =====================================================================
fig = plt.figure(figsize=(17.4, 19.6))
gs = gridspec.GridSpec(4, 6, figure=fig,
					   height_ratios=[1.00, 1.00, 0.66, 0.56],
					   hspace=0.30, wspace=0.42,
					   left=0.050, right=0.975, top=0.955, bottom=0.045)

axa = fig.add_subplot(gs[0, 0:2], projection=ccrs.PlateCarree())
axb = fig.add_subplot(gs[0, 2:4], projection=ccrs.PlateCarree())
axc = fig.add_subplot(gs[0, 4:6], projection=ccrs.PlateCarree())
axd = fig.add_subplot(gs[1, 0:2], projection=ccrs.PlateCarree())
axe = fig.add_subplot(gs[1, 2:4], projection=ccrs.PlateCarree())
axf = fig.add_subplot(gs[1, 4:6], projection=ccrs.PlateCarree())
axg = fig.add_subplot(gs[2, 0:3])
axh = fig.add_subplot(gs[2, 3:6])

gs_bar = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[3, :],
										  wspace=0.30)
axi1 = fig.add_subplot(gs_bar[0])
axi2 = fig.add_subplot(gs_bar[1])
axi3 = fig.add_subplot(gs_bar[2])


# =====================================================================
# (a) (b) (c)  CONVERSION, THREE WINDOWS, REF
# =====================================================================
norm_C = mcolors.TwoSlopeNorm(vmin=-C_LIM, vcenter=0, vmax=C_LIM)
for ax, wname, tag in ((axa, 'rising', 'a'), (axb, 'peak', 'b'),
					   (axc, 'decreasing', 'c')):
	if wname not in wins:
		continue
	k = wins.index(wname)
	C = ds.C_ref.isel(window=k).values
	pcm = ax.pcolormesh(lon, lat, np.where(wet, C, np.nan), cmap=CMAP_C,
						norm=norm_C, shading='auto', zorder=1,
						rasterized=True)
	dress(ax, labels_left=(tag == 'a'))
	ax.set_title(f'{wname} discharge', fontsize=11, pad=5)
	panel_letter(ax, tag, on_map=True)
	if tag == 'a':
		cbar_inside(ax, pcm, r'$C$ (W m$^{-2}$), REF', extend='both')


# =====================================================================
# (d) (e) (f)  THE FACTORISATION, ONE COLOUR SCALE FOR THE THREE
# =====================================================================
tot = ds.ln_total.isel(window=iw).values
amp = ds.ln_amplitude.isel(window=iw).values
ali = ds.ln_alignment.isel(window=iw).values

lim = LN_LIM
if lim is None:
	q = np.abs(tot[np.isfinite(tot)])
	lim = float(np.nanpercentile(q, 98)) if q.size else 1.0
	lim = max(lim, 0.05)
norm_ln = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

for ax, fld, tag, ttl in (
		(axd, tot, 'd', r'total,  $\ln(C_{\rm REF}/C_{\rm CTRL})$'),
		(axe, amp, 'e', r'amplitude,  $\ln(P_{\rm REF}/P_{\rm CTRL})$'),
		(axf, ali, 'f', r'alignment,  $\ln(\cos\Delta\phi_{\rm REF}/'
						r'\cos\Delta\phi_{\rm CTRL})$')):
	pcl = ax.pcolormesh(lon, lat, np.where(wet, fld, np.nan), cmap=CMAP_LN,
						norm=norm_ln, shading='auto', zorder=1,
						rasterized=True)
	dress(ax, labels_left=(tag == 'd'))
	ax.set_title(ttl, fontsize=10.5, pad=5)
	panel_letter(ax, tag, on_map=True)
	if tag == 'd':
		cbar_inside(ax, pcl, f'log ratio, {WIN} discharge', extend='both')
#
#axf.text(0.975, 0.965, '(e) plus (f) equals (d)\ncell by cell',
		 #transform=axf.transAxes, fontsize=8.5, va='top', ha='right',
		 #color='0.12', zorder=29, linespacing=1.35,
		 #bbox=dict(fc='w', ec='0.5', boxstyle='round,pad=0.28', alpha=0.9))


# =====================================================================
# (g)  ALIGNMENT AGAINST CRITICALITY ALONG THE BREAK
# =====================================================================
ab = ds.alpha_bin.values
axg.axvspan(ab.min(), 1.0, color=COL['subcritical'], alpha=0.45, lw=0,
			zorder=0)
axg.axvline(1.0, color=COL['critical'], lw=1.4, ls='--', zorder=3)
axg.axhline(0.0, color=COL['guide'], lw=1.0, ls=LS['guide'], zorder=3)

for k, w in enumerate(wins):
	for tag in ('ref', 'ctrl'):
		med = ds[f'bing_med_{tag}'].isel(window=k).values
		if tag == 'ref':
			q25 = ds[f'bing_q25_{tag}'].isel(window=k).values
			q75 = ds[f'bing_q75_{tag}'].isel(window=k).values
			axg.fill_between(ab, q25, q75, color=WCOL[w], alpha=0.14, lw=0,
							 zorder=2)
		axg.plot(ab, med, color=WCOL[w], ls=LS[tag],
				 lw=LW['main'] if tag == 'ref' else LW['secondary'], zorder=4)

axg.set_xlabel(r'criticality $\alpha = |\nabla h|/s$')
axg.set_ylabel(r'alignment $\cos\Delta\phi$')
axg.set_xlim(ab.min(), 2.25)
axg.set_ylim(-1.05, 1.05)
axg.grid(alpha=0.25, lw=0.4)
panel_letter(axg, 'g')
axg.text(0.02, 0.05, 'subcritical', transform=axg.transAxes, fontsize=9,
		 color='0.3', va='bottom', ha='left')
axg.text(0.985, 0.05, f'along the {BREAK:.0f} m isobath',
		 transform=axg.transAxes, fontsize=8.5, color='0.35',
		 va='bottom', ha='right')

if SHOW_RANK_CORR:
	rr = rank_corr(ds.alpha_ref_break.isel(window=iw).values,
				   ds.cosdphi_ref_break.isel(window=iw).values)
	rc = rank_corr(ds.alpha_ctrl_break.isel(window=iw).values,
				   ds.cosdphi_ctrl_break.isel(window=iw).values)
	axg.text(0.985, 0.2,
			 f'{WIN}, rank correlation   REF {rr:+.2f},  CTRL {rc:+.2f}',
			 transform=axg.transAxes, ha='right', va='top', fontsize=9,
			 color='0.2',
			 bbox=dict(fc='w', ec='0.6', boxstyle='round,pad=0.26',
					   alpha=0.9))

hg = [Line2D([], [], color=WCOL[w], ls=LS['ref'], lw=LW['main'], label=w)
	  for w in wins]
hg += [Line2D([], [], color=COL['guide'], ls=LS['ref'], lw=LW['main'],
			  label='REF'),
	   Line2D([], [], color=COL['guide'], ls=LS['ctrl'], lw=LW['secondary'],
			  label='CTRL')]
legend_above(axg, hg, ncol=5)


# =====================================================================
# (h)  BOTTOM PRESSURE AMPLITUDE AGAINST NEAR BOTTOM STRATIFICATION
# =====================================================================
if REBIN_H:
	pool = np.concatenate([ds[f'nb2_{t}_break'].values.ravel()
						   for t in ('ref', 'ctrl')])
	pool = pool[np.isfinite(pool) & (pool > 0)]
	lo, hi = np.percentile(pool, NB_PCTL)
	NB_BINS_H = np.logspace(np.log10(lo), np.log10(hi), NB_NBIN + 1)

for k, w in enumerate(wins):
	for tag in ('ref', 'ctrl'):
		if REBIN_H:
			xx = ds[f'nb2_{tag}_break'].isel(window=k).values
			yy = ds[f'P_{tag}_break'].isel(window=k).values
			if H_IMPEDANCE:
				yy = yy/ds[f'W_{tag}_break'].isel(window=k).values
			nb, med, q25, q75 = binned_median(xx, yy, NB_BINS_H)
		else:
			nb = ds.nb2_bin.values
			med = ds[f'binh_med_{tag}'].isel(window=k).values
			q25 = ds[f'binh_q25_{tag}'].isel(window=k).values
			q75 = ds[f'binh_q75_{tag}'].isel(window=k).values
		if tag == 'ref':
			axh.fill_between(nb, q25, q75, color=WCOL[w], alpha=0.14, lw=0,
							 zorder=2)
		axh.plot(nb, med, color=WCOL[w], ls=LS[tag],
				 lw=LW['main'] if tag == 'ref' else LW['secondary'], zorder=4)

axh.set_xscale('log'); axh.set_yscale('log')
axh.set_xlabel(r'$N_b^{2}$ (s$^{-2}$)')
axh.set_ylabel(r'$P/W$ (Pa s m$^{-1}$)' if H_IMPEDANCE
			   else r'$P$, amplitude of $p_{\rm bc}(-H)$ (Pa)')
axh.set_xlim(*NB_XLIM)
axh.grid(alpha=0.25, which='both', lw=0.4)
panel_letter(axh, 'h')
axh.text(0.985, 0.05, f'along the {BREAK:.0f} m isobath',
		 transform=axh.transAxes, fontsize=8.5, color='0.35',
		 va='bottom', ha='right')
legend_above(axh, hg, ncol=5)


# =====================================================================
# (i)  SIGNED DECOMPOSITION BY DEPTH BAND, WITH KAPPA
# =====================================================================
for ax, ib in zip((axi1, axi2, axi3), range(len(bands))):
	xs, labs, kaps = [], [], []
	x = 0
	for k, w in enumerate(wins):
		for tag, hatch in (('ref', ''), ('ctrl', '///')):
			pos = float(ds[f'Cpos_{tag}'][k, ib].values)
			neg = float(ds[f'Cneg_{tag}'][k, ib].values)
			net = float(ds[f'Cnet_{tag}'][k, ib].values)
			kap = float(ds[f'kappa_{tag}'][k, ib].values)
			ax.bar(x, pos, width=0.78, color=COL['positive'], hatch=hatch,
				   edgecolor='k', lw=0.4)
			ax.bar(x, neg, width=0.78, color=COL['negative'], hatch=hatch,
				   edgecolor='k', lw=0.4)
			ax.plot(x, net, marker='D', ms=7, mfc=COL['net'], mec='w',
					mew=1.0, ls='none', zorder=6)
			ax.annotate(f'{net:.0f}', xy=(x, net), xytext=(0, 11),
						textcoords='offset points', ha='center',
						fontsize=7, fontweight='bold')
			xs.append(x); labs.append(f'{w[:4]}\n{tag.upper()}')
			kaps.append((x, kap))
			x += 1
		x += 0.5

	ymin = ax.get_ylim()[0]
	for xk, kap in kaps:
		ax.annotate(f'$\\kappa$={kap:.2f}', xy=(xk, ymin),
					xytext=(0, -16), textcoords='offset points',
					ha='center', fontsize=7, color='0.3',
					annotation_clip=False)

	ax.axhline(0, color='k', lw=0.9)
	ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=8)
	ax.set_ylabel('MW')
	ax.set_title(blabel.get(bands[ib], bands[ib]), fontsize=10.5)
	ax.grid(alpha=0.22, lw=0.4, axis='y')
	ax.margins(y=0.26)
	panel_letter(ax, 'i' if ib == 0 else '')

hi = [Patch(fc=COL['positive'], ec='k', lw=0.4, label=r'$\int C^{+}$'),
	  Patch(fc=COL['negative'], ec='k', lw=0.4, label=r'$\int C^{-}$'),
	  Line2D([], [], marker='D', ms=7, mfc=COL['net'], mec='w', ls='none',
			 label='net'),
	  Patch(fc='w', ec='k', lw=0.4, hatch='///', label='hatched, CTRL')]
axi2.legend(handles=hi, loc='lower center', bbox_to_anchor=(0.5, 1.14),
			ncol=4, frameon=False, fontsize=9.5)


# =====================================================================
fig.savefig(OUTFIG, dpi=DPI, bbox_inches='tight')
print('wrote', OUTFIG)