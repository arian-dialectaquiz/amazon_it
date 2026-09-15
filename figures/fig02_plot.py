##############
"""
fig02_plot.py
=================================================================
Figure 2  -  The medium the discharge builds
Amazon shelf internal tide manuscript, version 3

Flat script. Every panel has its own axes handle and every element is
drawn by its own call.

	axa  gs[0, 0:2]   c1, rising discharge, REF
	axb  gs[0, 2:4]   c1, peak discharge, REF
	axc  gs[0, 4:6]   c1, decreasing discharge, REF
	axd  gs[1, 0:2]   c1 difference between the runs at peak
	axe1 axe2 axe3    vertical structure of mode one at M1, M2 and M3
	axf  gs[2, 0:3]   |phi1(-H)| against the surface trapping index
	axg  gs[2, 3:6]   near bottom N2 along the 200 m isobath

Conventions, shared with the rest of the figure set

	the EXPERIMENT is carried by the LINE STYLE, REF solid and CTRL
	dashed, so the runs separate in greyscale and for a colour blind
	reader, and the WINDOW is carried by the COLOUR

Every colour is a named matplotlib colour in the block below, so a
change there changes it everywhere in this figure.

Reads fig02_data.nc written by fig02_compute.py.
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
import cmocean.cm as cmo
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

plt.rcParams.update({
	'font.size': 11, 'axes.labelsize': 11.5, 'axes.titlesize': 11.5,
	'legend.fontsize': 9.5, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
	'axes.linewidth': 0.9,
})

# ---------------------------------------------------------------------
DATA   = 'fig02_data.nc'
OUTFIG = 'Fig2.jpeg'
DPI    = 300
WIN    = 'peak'                 # window shown in panels (d), (e) and (f)

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
	front='white',
	front_dark='black',
	lens='crimson',
	n2='silver',
	guide='dimgrey',
	site='crimson',
)

# ---- line styles, the experiment is the style
LS = dict(ref='-', ctrl='--', guide=':')
LW = dict(main=2.2, secondary=1.3, faint=0.9, guide=1.0)

# ---- the only two colour maps in this figure
CMAP_C1  = plt.cm.gnuplot
CMAP_DC1 = 'RdBu_r'

C1_MIN_PLOT = 0.05              # m s-1, bottom of the log colour scale
C1_MAX_PLOT = 3.0               # m s-1, top of the same
DC1_LIM = None                  # None takes the 98th percentile
TICK_FS = 8.5
LET_FS = 16
SHOW_PHI_BIG = True             # draw Phi_1 alongside phi_1 in panel (e)
N2_EXP = -3            # power of ten factored out of the N2 axis in panel (e)

ds = xr.open_dataset(DATA)

lon  = ds.lon_rho.values
lat  = ds.lat_rho.values
mask = ds.mask_rho.values
hraw = ds.h.values
h    = np.where(mask > 0, hraw, np.nan)
wet  = mask > 0
wins = [str(w) for w in ds.window.values]
iw   = wins.index(WIN)
stations = [str(s) for s in ds.station.values]
BREAK = float(ds.attrs.get('break_isobath', 200.))
S_LENS = float(ds.attrs.get('s_lens', 35.))

WCOL = {w: COL.get(w, 'grey') for w in wins}

land = cfeature.NaturalEarthFeature('physical', 'land', scale='10m')


def dress(ax, labels_left=True, labels_bottom=True):
	"""Coastline, isobaths, extent and graticule, common to every map."""
	ax.set_aspect('auto', adjustable='box')
	ax.add_feature(land, facecolor=COL['land'], edgecolor=COL['coast'],
				   lw=0.4, zorder=8)
	ax.coastlines(resolution='10m', linewidths=0.4, zorder=9)
	ax.contour(lon, lat, h, levels=[BREAK], colors=COL['isobath'],
			   linewidths=1.0, zorder=6)
	ax.contour(lon, lat, h, levels=[50, 100, 1000],
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
				anchor=(0.035, 0.12, 1, 1), width='70%', height='3.2%'):
	cax = inset_axes(ax, width=width, height=height, loc='lower left',
					 bbox_to_anchor=anchor, bbox_transform=ax.transAxes,
					 borderpad=0)
	cb = plt.colorbar(mappable, cax=cax, orientation='horizontal',
					  extend=extend)
	cb.set_label(label, fontsize=12, labelpad=1.5)
	cb.ax.tick_params(labelsize=12, length=2.5, pad=1.5)
	cb.outline.set_linewidth(0.7)
	cax.set_facecolor('w'); cax.patch.set_alpha(0.85); cax.set_zorder(30)
	return cb


def legend_above(ax, handles, ncol, y=1.015, x=0.0):
	return ax.legend(handles=handles, loc='lower left',
					 bbox_to_anchor=(x, y), ncol=ncol, frameon=False,
					 fontsize=9.5, handlelength=2.2, columnspacing=1.6,
					 borderaxespad=0.0)


# =====================================================================
# FIGURE SKELETON
# =====================================================================
fig = plt.figure(figsize=(17.4, 15.0))
gs = gridspec.GridSpec(3, 6, figure=fig,
					   height_ratios=[1.00, 0.92, 0.70],
					   hspace=0.30, wspace=0.42,
					   left=0.050, right=0.975, top=0.945, bottom=0.055)

axa = fig.add_subplot(gs[0, 0:2], projection=ccrs.PlateCarree())
axb = fig.add_subplot(gs[0, 2:4], projection=ccrs.PlateCarree())
axc = fig.add_subplot(gs[0, 4:6], projection=ccrs.PlateCarree())
axd = fig.add_subplot(gs[1, 0:2], projection=ccrs.PlateCarree())

gs_prof = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 2:6],
										   wspace=0.34)
axe1 = fig.add_subplot(gs_prof[0])
axe2 = fig.add_subplot(gs_prof[1])
axe3 = fig.add_subplot(gs_prof[2])

axf = fig.add_subplot(gs[2, 0:3])
axg = fig.add_subplot(gs[2, 3:6])


# =====================================================================
# (a) (b) (c)  MODE ONE PHASE SPEED, THREE WINDOWS, REF
# =====================================================================
norm_c1 = mcolors.LogNorm(vmin=C1_MIN_PLOT, vmax=C1_MAX_PLOT)
for ax, wname, tag in ((axa, 'rising', 'a'), (axb, 'peak', 'b'),
					   (axc, 'decreasing', 'c')):
	if wname not in wins:
		continue
	k = wins.index(wname)
	c1 = ds.c1_ref.isel(window=k).values
	pcm = ax.pcolormesh(lon, lat, np.where(wet, c1, np.nan), cmap=CMAP_C1,
						norm=norm_c1, shading='auto', zorder=1,
						rasterized=True)
	ss = ds.salt_surf_ref.isel(window=k).values
	ax.contour(lon, lat, np.where(wet, ss, np.nan), levels=[S_LENS],
			   colors=COL['front'], linewidths=1.6, zorder=7)
	dress(ax, labels_left=(tag == 'a'))
	ax.set_title(f'{wname} discharge', fontsize=11, pad=5)
	panel_letter(ax, tag, on_map=True)
	if tag == 'a':
		cbar_inside(ax, pcm, r'$c_1$ (m s$^{-1}$), REF', extend='both')

axa.text(0.975, 0.965, f'white, surface {S_LENS:.0f} isohaline',
		 transform=axa.transAxes, fontsize=8.5, va='top', ha='right',
		 color='0.12', zorder=29,
		 bbox=dict(fc='w', ec='0.5', boxstyle='round,pad=0.28', alpha=0.9))


# =====================================================================
# (d)  DIFFERENCE IN c1 BETWEEN THE EXPERIMENTS
# =====================================================================
dc1 = ds.c1_ref.isel(window=iw).values - ds.c1_ctrl.isel(window=iw).values
lim = DC1_LIM
if lim is None:
	q = np.abs(dc1[wet]); q = q[np.isfinite(q)]
	lim = float(np.nanpercentile(q, 98)) if q.size else 1.0
	lim = max(lim, 0.05)

pcd = axd.pcolormesh(lon, lat, np.where(wet, dc1, np.nan), cmap=CMAP_DC1,
					 norm=mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0,
											   vmax=lim),
					 shading='auto', zorder=1, rasterized=True)
for tag, ls in (('ref', LS['ref']), ('ctrl', LS['ctrl'])):
	ss = ds[f'salt_surf_{tag}'].isel(window=iw).values
	axd.contour(lon, lat, np.where(wet, ss, np.nan), levels=[S_LENS],
				colors=COL['front_dark'], linewidths=1.5, linestyles=ls,
				zorder=7)
for n, s in enumerate(stations):
	axd.plot(float(ds.station_lon.values[n]), float(ds.station_lat.values[n]),
			 marker='o', ms=7, mfc='gold', mec='k', mew=0.9, ls='none',
			 zorder=15)
	axd.text(float(ds.station_lon.values[n]) - 0.18,
			 float(ds.station_lat.values[n]), s, fontsize=9,
			 fontweight='bold', ha='right', va='center', zorder=16)
dress(axd)
axd.set_title(f'$\\Delta c_1$, REF minus CTRL, {WIN}', fontsize=11, pad=5)
cbar_inside(axd, pcd, r'$\Delta c_1$ (m s$^{-1}$)', extend='both')
panel_letter(axd, 'd', on_map=True)
axd.text(0.975, 0.965,
		 f'{S_LENS:.0f} isohaline\nsolid REF, dashed CTRL',
		 transform=axd.transAxes, fontsize=8.5, va='top', ha='right',
		 color='0.12', zorder=29, linespacing=1.35,
		 bbox=dict(fc='w', ec='0.5', boxstyle='round,pad=0.28', alpha=0.9))


# =====================================================================
# (e)  VERTICAL STRUCTURE OF MODE ONE AT THE MOORINGS
# =====================================================================
for n, (ax, s) in enumerate(zip((axe1, axe2, axe3), stations)):
	hst = float(ds.station_h.values[n])

	axn2 = ax.twiny()
	zw = ds.prof_zw_ref.isel(station=n, window=iw).values
	n2 = ds.prof_N2_ref.isel(station=n, window=iw).values*10.0**(-N2_EXP)
	axn2.fill_betweenx(zw, 0, n2, color=COL['n2'], alpha=0.55, lw=0, zorder=0)
	axn2.plot(n2, zw, color=COL['n2'], lw=1.0, zorder=1)
	axn2.set_xlim(0, max(np.nanmax(n2)*1.15, 1e-6))
	axn2.tick_params(labelsize=7.5, colors='0.35')
	axn2.set_xlabel(rf'$N^2$ ($10^{{{N2_EXP}}}$ s$^{{-2}}$), REF',
					fontsize=8.5, color='0.35', labelpad=2)
	axn2.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='lower'))
	axn2.xaxis.set_major_formatter(FormatStrFormatter('%g'))

	for tag in ('ref', 'ctrl'):
		zr = ds[f'prof_zr_{tag}'].isel(station=n, window=iw).values
		ph = ds[f'prof_phi1_{tag}'].isel(station=n, window=iw).values
		ax.plot(ph, zr, color=COL[tag], ls=LS[tag], lw=LW['main'], zorder=5)
		if SHOW_PHI_BIG:
			zwt = ds[f'prof_zw_{tag}'].isel(station=n, window=iw).values
			PH = ds[f'prof_Phi1_{tag}'].isel(station=n, window=iw).values
			sc = np.nanmax(np.abs(ph))/max(np.nanmax(np.abs(PH)), 1e-9)
			ax.plot(PH*sc, zwt, color=COL[tag], ls=LS[tag], lw=LW['faint'],
					alpha=0.55, zorder=4)
		hp = float(ds[f'prof_hp_{tag}'].isel(station=n, window=iw).values)
		if np.isfinite(hp):
			ax.axhline(-hp, color=COL['lens'], ls=LS[tag], lw=1.1, alpha=0.9,
					   zorder=6)

	ax.axvline(0, color=COL['guide'], lw=0.9, ls=LS['guide'], zorder=2)
	ax.set_ylim(-hst, 0)
	ax.set_xlabel(r'$\varphi_1$', labelpad=2)
	ax.grid(alpha=0.22, lw=0.4)
	c1r = float(ds.prof_c1_ref.isel(station=n, window=iw).values)
	c1c = float(ds.prof_c1_ctrl.isel(station=n, window=iw).values)
	pbr = float(ds.phib1_ref.isel(window=iw).values[0, 0]) * 0 + np.nan
	ax.text(0.97, 0.97, f'{s},  {hst:.0f} m', transform=ax.transAxes,
			fontsize=10.5, fontweight='bold', va='top', ha='right',
			color='0.10', zorder=27,
			bbox=dict(fc='w', ec='0.6', boxstyle='round,pad=0.24',
					  alpha=0.92))
	ax.text(0.03, 0.03, f'$c_1$  {c1r:.2f} / {c1c:.2f} m s$^{{-1}}$',
			transform=ax.transAxes, fontsize=8.5, va='bottom', ha='left',
			color='0.15',
			bbox=dict(fc='w', ec='0.6', boxstyle='round,pad=0.22',
					  alpha=0.9))
	if n == 0:
		ax.set_ylabel('depth (m)')
	panel_letter(ax, 'e' if n == 0 else '')

he = [Line2D([], [], color=COL['ref'], ls=LS['ref'], lw=LW['main'],
			 label=r'$\varphi_1$, REF'),
	  Line2D([], [], color=COL['ctrl'], ls=LS['ctrl'], lw=LW['main'],
			 label=r'$\varphi_1$, CTRL'),
	  Line2D([], [], color=COL['guide'], ls=LS['ref'], lw=LW['faint'],
			 alpha=0.6, label=r'$\Phi_1$, scaled'),
	  Line2D([], [], color=COL['lens'], ls=LS['ref'], lw=1.1,
			 label=f'plume base $h_p$')]
axe1.legend(handles=he, loc='lower center', bbox_to_anchor=(1.0, 1.1),
			ncol=4, frameon=False, fontsize=9.5, handlelength=2.2,
			columnspacing=1.5)
# =====================================================================
# (f)  BOTTOM AMPLITUDE OF MODE ONE AGAINST THE TRAPPING INDEX
# =====================================================================
tb = ds.t1_bin.values
for k, w in enumerate(wins):
	for tag in ('ref', 'ctrl'):
		x = ds[f'sub_trap1_{tag}'].isel(window=k).values
		y = ds[f'sub_phib1_{tag}'].isel(window=k).values
		if tag == 'ref':
			axf.plot(x, y, ls='none', marker='.', ms=1.6, alpha=0.10,
					 color=WCOL[w], zorder=1, rasterized=True)
		med = ds[f'bin_med_{tag}'].isel(window=k).values
		q25 = ds[f'bin_q25_{tag}'].isel(window=k).values
		q75 = ds[f'bin_q75_{tag}'].isel(window=k).values
		if tag == 'ref':
			axf.fill_between(tb, q25, q75, color=WCOL[w], alpha=0.14, lw=0,
							 zorder=2)
		axf.plot(tb, med, color=WCOL[w], ls=LS[tag], lw=LW['main'], zorder=4)

axf.set_xlabel('surface trapping index $T_1$')
axf.set_ylabel(r'$|\varphi_1(-H)|$')
axf.grid(alpha=0.25, lw=0.4)
axf.set_xlim(tb.min(), tb.max())
panel_letter(axf, 'f')

hf = [Line2D([], [], color=WCOL[w], ls=LS['ref'], lw=LW['main'], label=w)
	  for w in wins]
hf += [Line2D([], [], color=COL['guide'], ls=LS['ref'], lw=LW['main'],
			  label='REF'),
	   Line2D([], [], color=COL['guide'], ls=LS['ctrl'], lw=LW['main'],
			  label='CTRL')]
legend_above(axf, hf, ncol=5)
sl = float(ds.fit_slope_ref.isel(window=iw).values)
rr = float(ds.fit_r_ref.isel(window=iw).values)
axf.text(0.985, 0.94,
		 f'{WIN}, REF   slope {sl:+.2f},  $r$ {rr:+.2f}',
		 transform=axf.transAxes, ha='right', va='top', fontsize=9,
		 color='0.2',
		 bbox=dict(fc='w', ec='0.6', boxstyle='round,pad=0.26', alpha=0.9))
axf.text(0.5, 0.06,
		 f'columns between {float(ds.attrs.get("h_min_stats", 50)):.0f} and '
		 f'{float(ds.attrs.get("h_max_stats", 1000)):.0f} m',
		 transform=axf.transAxes, ha='right', va='bottom', fontsize=8.5,
		 color='0.35')


# =====================================================================
# (g)  NEAR BOTTOM STRATIFICATION ALONG THE 200 m ISOBATH
# =====================================================================
blat = ds.break_lat.values
for k, w in enumerate(wins):
	for tag in ('ref', 'ctrl'):
		v = ds[f'nb2_{tag}_break'].isel(window=k).values
		axg.plot(blat, v, color=WCOL[w], ls=LS[tag],
				 lw=LW['main'] if tag == 'ref' else LW['secondary'],
				 alpha=0.95 if tag == 'ref' else 0.85, zorder=4)

axg.set_yscale('log')
axg.set_xlabel(f'latitude along the {BREAK:.0f} m isobath')
axg.set_ylabel(r'$N_b^{2}$ (s$^{-2}$)')
axg.set_xlim(blat.min(), blat.max())
axg.grid(alpha=0.25, which='both', lw=0.4)
panel_letter(axg, 'g')

hg = [Line2D([], [], color=WCOL[w], ls=LS['ref'], lw=LW['main'], label=w)
	  for w in wins]
hg += [Line2D([], [], color=COL['guide'], ls=LS['ref'], lw=LW['main'],
			  label='REF'),
	   Line2D([], [], color=COL['guide'], ls=LS['ctrl'], lw=LW['secondary'],
			  label='CTRL')]
legend_above(axg, hg, ncol=5)
axg.text(0.985, 0.06, 'lowest 50 m of the column',
		 transform=axg.transAxes, ha='right', va='bottom', fontsize=8.5,
		 color='0.35')


# =====================================================================
fig.savefig(OUTFIG, dpi=DPI, bbox_inches='tight')
print('wrote', OUTFIG)