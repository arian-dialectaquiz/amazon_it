#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig05_plot.py
=================================================================
Figure 5  -  Radiation and where the energy is lost
Amazon shelf internal tide manuscript, version 3

Flat script. Every panel has its own axes handle.

	axa  gs[0, 0:2]   depth integrated flux, rising, REF
	axb  gs[0, 2:4]   the same at peak
	axc  gs[0, 4:6]   the same at decreasing
	axd  gs[1, 0:2]   control volume budget by band
	axe  gs[1, 2:4]   cross isobath power against isobath depth
	axf  gs[1, 4:6]   radiated fraction and what becomes of the rest

Every flux term is a line integral along an isobath or a domain edge, so
nothing here rests on a horizontal derivative of the flux.

Conventions

	the EXPERIMENT is carried by the LINE STYLE, REF solid and CTRL
	dashed, and the WINDOW by the COLOUR, except in (d) where colour
	carries the BUDGET TERM and in (e) where it carries the MODE.

Reads fig05_data.nc written by fig05_compute.py.
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
DATA   = 'fig05_data.nc'
OUTFIG = 'Fig5.jpeg'
DPI    = 300
WIN    = 'peak'

COL = dict(
	ref='navy', ctrl='orangered',
	rising='steelblue', peak='darkgoldenrod', decreasing='seagreen',
	mode1='midnightblue', mode2='seagreen', mode3='darkgoldenrod',
	mode4='firebrick',
	t_C='dimgrey', t_export='steelblue', t_edge='mediumpurple',
	t_loss='indianred', total='black',
	land='wheat', coast='grey', isobath='black', isobath_thin='dimgrey',
	guide='dimgrey', unity='crimson', quiver='black',
)

LS = dict(ref='-', ctrl='--', guide=':')
LW = dict(main=2.2, secondary=1.4, faint=0.9, guide=1.1)

CMAP_F = 'magma_r'
F_LIM  = None               # W m-1, None takes the 98th percentile
QSTEP  = 7                  # quiver decimation
QSCALE = None               # None lets matplotlib choose
TICK_FS = 8.5
LET_FS = 16

ds = xr.open_dataset(DATA)

lon  = ds.lon_rho.values
lat  = ds.lat_rho.values
mask = ds.mask_rho.values
h    = np.where(mask > 0, ds.h.values, np.nan)
ang  = ds.angle.values
wins = [str(w) for w in ds.window.values]
bands = [str(b) for b in ds.band.values]
modes = [int(m) for m in ds.mode.values]
isob = np.array([float(v) for v in ds.isobath.values])
iw   = wins.index(WIN)
NFLUX = int(ds.attrs.get('nmode_flux', 2))
EISO = float(ds.attrs.get('e_isobath', 1000.))
BREAK = float(ds.attrs.get('break_isobath', 200.))
blabel = dict(x.split(':', 1) for x in
			  ds.attrs.get('bands', '').split(';') if ':' in x)

WCOL = {w: COL.get(w, 'grey') for w in wins}
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


def cbar_inside(ax, mappable, label, extend='max',
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


def to_east_north(fx, fy, angle):
	"""Rotate grid components onto east and north for the quiver."""
	ca, sa = np.cos(angle), np.sin(angle)
	return fx*ca - fy*sa, fx*sa + fy*ca


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
# (a) (b) (c)  DEPTH INTEGRATED FLUX, THREE WINDOWS, REF
# =====================================================================
lim = F_LIM
if lim is None:
	q = ds.Fmag_ref.values
	q = q[np.isfinite(q)]
	lim = float(np.nanpercentile(q, 98)) if q.size else 1e3
norm_F = mcolors.Normalize(vmin=0, vmax=lim)

s = QSTEP
for ax, wname, tag in ((axa, 'rising', 'a'), (axb, 'peak', 'b'),
					   (axc, 'decreasing', 'c')):
	if wname not in wins:
		continue
	k = wins.index(wname)
	M = ds.Fmag_ref.isel(window=k).values
	fx = ds.Fxi_ref.isel(window=k).values
	fy = ds.Feta_ref.isel(window=k).values
	pcm = ax.pcolormesh(lon, lat, M, cmap=CMAP_F, norm=norm_F,
						shading='auto', zorder=1, rasterized=True)
	fe, fn = to_east_north(fx, fy, ang)
	ax.quiver(lon[::s, ::s], lat[::s, ::s], fe[::s, ::s], fn[::s, ::s],
			  color=COL['quiver'], scale=QSCALE, width=0.0035,
			  transform=ccrs.PlateCarree(), zorder=7)
	dress(ax, labels_left=(tag == 'a'))
	ax.set_title(f'{wname} discharge', fontsize=11, pad=5)
	panel_letter(ax, tag, on_map=True)
	if tag == 'a':
		cbar_inside(ax, pcm, r'$|\mathbf{F}_{\rm bc}|$ (W m$^{-1}$), REF')


# =====================================================================
# (d)  CONTROL VOLUME BUDGET BY BAND
# =====================================================================
TERMS = (('C', 't_C', r'$\int C$'),
		 ('net', 't_export', 'net export'),
		 ('F_edge', 't_edge', 'domain edges'),
		 ('loss', 't_loss', 'loss'))

xs, labs = [], []
x = 0
for ib, b in enumerate(bands):
	x0 = x
	for k, w in enumerate(wins):
		for tag, hatch in (('ref', ''), ('ctrl', '///')):
			vals = dict(
				C=float(ds[f'bud_C_{tag}'][k, ib].values),
				net=float(ds[f'bud_F_out_{tag}'][k, ib].values)
					- float(ds[f'bud_F_in_{tag}'][k, ib].values),
				F_edge=float(ds[f'bud_F_edge_{tag}'][k, ib].values),
				loss=float(ds[f'bud_loss_{tag}'][k, ib].values))
			wdt = 0.20
			for j, (key, ckey, lab) in enumerate(TERMS):
				axd.bar(x + (j - 1.5)*wdt, vals[key], width=wdt,
						color=COL[ckey], hatch=hatch, edgecolor='k', lw=0.3)
			xs.append(x); labs.append(f'{w[:3]}\n{"R" if tag == "ref" else "C"}')
			x += 1
		x += 0.4
	axd.text(0.5*(x0 + x - 1.4), 0.9, blabel.get(b, b).split(',')[0],
			 fontsize=9, ha='center', color='0.25', fontweight='bold',
			 transform=axd.get_xaxis_transform())
	if ib < len(bands) - 1:
		axd.axvline(x - 0.15, color='0.5', lw=0.9, ls=':')
	x += 0.6

axd.axhline(0, color='k', lw=0.9)
axd.set_xticks(xs); axd.set_xticklabels(labs, fontsize=7)
axd.set_ylabel('MW')
axd.grid(alpha=0.22, lw=0.4, axis='y')
axd.margins(y=0.22)
panel_letter(axd, 'd')

hd = [Patch(fc=COL[c], ec='k', lw=0.3, label=l) for _, c, l in TERMS]
hd += [Patch(fc='w', ec='k', lw=0.3, hatch='///', label='hatched, CTRL')]
legend_above(axd, hd, ncol=5)


# =====================================================================
# (e)  CROSS ISOBATH POWER AGAINST ISOBATH DEPTH
# =====================================================================
for tag in ('ref', 'ctrl'):
	v = ds[f'iso_P_tot_{tag}'].isel(window=iw).values
	axe.plot(isob, v, color=COL['total'], ls=LS[tag],
			 lw=LW['main'] if tag == 'ref' else LW['secondary'],
			 marker='o' if tag == 'ref' else None, ms=5, zorder=5)
	for n in range(NFLUX):
		vn = ds[f'iso_P_mode_{tag}'].isel(window=iw, mode=n).values
		axe.plot(isob, vn, color=MCOL[n], ls=LS[tag],
				 lw=LW['main'] if tag == 'ref' else LW['secondary'],
				 marker='s' if tag == 'ref' else None, ms=4, zorder=4)

axe.axvline(EISO, color=COL['unity'], lw=1.2, ls='--', zorder=3)
axe.axhline(0, color='k', lw=0.9)
axe.set_xscale('log')
axe.set_xlabel('isobath depth (m)')
axe.set_ylabel('power crossing, offshore (MW)')
axe.set_xlim(isob.min(), isob.max())
axe.set_xticks(isob)
axe.set_xticklabels([f'{v:.0f}' for v in isob], fontsize=8)
axe.grid(alpha=0.25, which='major', lw=0.4)
panel_letter(axe, 'e')

he = [Line2D([], [], color=COL['total'], ls=LS['ref'], lw=LW['main'],
			 marker='o', ms=5, label='total')]
he += [Line2D([], [], color=MCOL[n], ls=LS['ref'], lw=LW['main'],
			  marker='s', ms=4, label=f'mode {modes[n]}')
	   for n in range(NFLUX)]
he += [Line2D([], [], color=COL['guide'], ls=LS['ctrl'],
			  lw=LW['secondary'], label='CTRL')]
legend_above(axe, he, ncol=4)
axe.text(0.985, 0.06, f'{WIN} discharge', transform=axe.transAxes,
		 ha='right', va='bottom', fontsize=8.5, color='0.35')


# =====================================================================
# (f)  RADIATED FRACTION AND WHAT BECOMES OF THE REST
# =====================================================================
xs2, labs2 = [], []
x = 0
for k, w in enumerate(wins):
	for tag, hatch in (('ref', ''), ('ctrl', '///')):
		E = float(ds[f'rad_E_{tag}'][k].values)
		axf.bar(x, E, width=0.72, color=COL['t_export'], hatch=hatch,
				edgecolor='k', lw=0.35)
		axf.bar(x, max(1.0 - E, 0.0), bottom=E, width=0.72,
				color=COL['t_loss'], hatch=hatch, edgecolor='k', lw=0.35)
		axf.annotate(f'{E:.2f}', xy=(x, E), xytext=(0, 3),
					 textcoords='offset points', ha='center', fontsize=7.5,
					 fontweight='bold', color='0.15')
		xs2.append(x); labs2.append(f'{w[:3]}\n{"R" if tag == "ref" else "C"}')
		x += 1
	x += 0.45

axf.axhline(1.0, color=COL['unity'], lw=1.1, ls='--')
axf.set_xticks(xs2); axf.set_xticklabels(labs2, fontsize=8)
axf.set_ylabel('share of the generated energy')
axf.set_ylim(0, 1.18)
axf.grid(alpha=0.22, lw=0.4, axis='y')
panel_letter(axf, 'f')

hf = [Patch(fc=COL['t_export'], ec='k', lw=0.35,
			label=f'crosses {EISO:.0f} m'),
	  Patch(fc=COL['t_loss'], ec='k', lw=0.35, label='lost or leaves sideways'),
	  Patch(fc='w', ec='k', lw=0.35, hatch='///', label='hatched, CTRL')]
legend_above(axf, hf, ncol=3)


# =====================================================================
fig.savefig(OUTFIG, dpi=DPI, bbox_inches='tight')
print('wrote', OUTFIG)