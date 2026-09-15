#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig01_plot.py
=================================================================
Figure 1  -  The margin and the controls that make the experiment
			 interpretable
Amazon shelf internal tide manuscript, version 2

Flat script. Every panel has its own axes handle and every element is
drawn by its own call, so any of them can be restyled or removed
without touching the rest.

	axa  gs[0, 0]      map, bathymetry, generation sites, moorings,
					   sections and the shelf break path
	axb  gs[0, 1]      barotropic M2 control, difference between the
					   runs in the depth averaged current, with the
					   elevation difference as an inset histogram
	axc  gs[0, 2]      surface trapping validity, h / hp with the
					   h = 3 hp contour
	axd  gs[1, :]      points per mode one wavelength against depth in
					   both runs, with the fraction resolved in BOTH
	axe  gs[2, :] top  criticality along the 200 m isobath, both runs
	axf  gs[2, :] bot  the difference in criticality between the runs,
					   sharing the latitude axis of axe

Layout notes
------------
* The three maps occupy one row at equal width. Every map axes is set
  to aspect 'auto' so the drawn map fills its box, which is what keeps
  the inset colour bars inside the panel.
* Legends sit on a strip above their panel, so no curve is covered and
  no axis has to be shortened to make room.
* The criticality difference is a panel in its own right sharing the
  latitude axis of the profile above it, in place of the inset that was
  too small to read.

Reads fig01_data.nc written by fig01_compute.py.
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

try:
	from scipy.ndimage import uniform_filter
	HAS_SCIPY = True
except ImportError:
	HAS_SCIPY = False

plt.rcParams.update({
	'font.size': 11, 'axes.labelsize': 11.5, 'axes.titlesize': 11.5,
	'legend.fontsize': 9.5, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
	'axes.linewidth': 0.9,
})

# ---------------------------------------------------------------------
DATA   = 'fig01_data.nc'
OUTFIG = 'Fig1.jpeg'
DPI    = 300
WIN    = 'peak'             # window shown in panels (b) to (f)

C_REF  = '#0b3d91'          # realistic, variable discharge
C_CTRL = '#c1440e'          # constant discharge
C_OK   = '#2a6fb5'
TICK_FS = 10

HR_MAX  = 12.0              # top of the h/hp scale in panel (c)
CONT_SMOOTH = 3             # cells, smoothing applied to the h = 3 hp
							# contour only, 0 to disable
LET_FS = 16                 # panel letter size

ds = xr.open_dataset(DATA)

lon  = ds.lon_rho.values
lat  = ds.lat_rho.values
mask = ds.mask_rho.values
hraw = ds.h.values
h    = np.where(mask > 0, hraw, np.nan)
wet  = mask > 0
wins = [str(w) for w in ds.window.values]
iw   = wins.index(WIN)

BREAK   = float(ds.attrs.get('break_isobath', 200.))
PPW_LO  = float(ds.attrs.get('ppw_lo', 8.))
PPW_HI  = float(ds.attrs.get('ppw_hi', 12.))
PPW_MIN = float(ds.attrs.get('ppw_min', 8.))
TRAP_R  = float(ds.attrs.get('trap_ratio', 3.))

moorings = {}
for item in ds.attrs.get('moorings', '').split(';'):
	if ':' in item:
		k, v = item.split(':'); a, b = v.split(',')
		moorings[k] = (float(a), float(b))

sections = {}
for item in ds.attrs.get('sections', '').split(';'):
	if ':' in item:
		k, v = item.split(':'); a, b, c, d = v.split(',')
		sections[k] = (float(a), float(b), float(c), float(d))

MAIN_SITES = [s for s in ds.attrs.get('gen_sites_main', 'A,B').split(',') if s]
site_names = [str(s) for s in ds.site.values]

land   = cfeature.NaturalEarthFeature('physical', 'land', scale='10m')
states = cfeature.NaturalEarthFeature(category='cultural',
									  name='admin_1_states_provinces_lines',
									  scale='10m')


def dress(ax, labels_left=True, labels_bottom=True):
	"""Coastline, isobaths, extent and graticule, common to every map."""
	ax.set_aspect('auto', adjustable='box')
	ax.add_feature(land, facecolor='wheat', edgecolor='grey', lw=0.4, zorder=8)
	ax.coastlines(resolution='10m', linewidths=0.4, zorder=9)
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
		ax.text(0.006, 0.965, s, transform=ax.transAxes, zorder=28, **kw)


def cbar_inside(ax, mappable, label, extend='both',
				anchor=(0.045, 0.1, 1, 1), width='100%', height='3.5%'):
	"""
	Horizontal colour bar inside the lower left of a map. Every map axes
	is set to aspect 'auto' by dress(), so the drawn map fills the box
	and this lands over the land corner rather than below the frame.
	"""
	cax = inset_axes(ax, width=width, height=height, loc='lower left',
					 bbox_to_anchor=anchor, bbox_transform=ax.transAxes,
					 borderpad=0)
	cb = plt.colorbar(mappable, cax=cax, orientation='horizontal',
					  extend=extend)
	cb.set_label(label, fontsize=12, labelpad=1.5)
	cb.ax.tick_params(labelsize=12, length=2.5, pad=1.5)
	cb.outline.set_linewidth(0.7)
	cax.set_facecolor('w')
	cax.patch.set_alpha(0.85)
	cax.set_zorder(30)
	return cb


def legend_above(ax, handles, ncol, y=1.015, x=0.0):
	"""Legend on a strip above the panel, so no curve is covered."""
	return ax.legend(handles=handles, loc='lower left',
					 bbox_to_anchor=(x, y), ncol=ncol, frameon=False,
					 fontsize=12, handlelength=1.9, columnspacing=1.6,
					 borderaxespad=0.0)


def smooth_for_contour(f, n=CONT_SMOOTH):
	"""Light smoothing so a threshold contour is a line and not speckle."""
	if not n or not HAS_SCIPY:
		return f
	ok = np.isfinite(f)
	filled = np.where(ok, f, 0.0)
	wgt = uniform_filter(ok.astype(float), n)
	out = uniform_filter(filled, n)/np.maximum(wgt, 1e-6)
	return np.where(wgt > 0.35, out, np.nan)


# =====================================================================
# FIGURE SKELETON
# =====================================================================
fig = plt.figure(figsize=(18, 15.4))
gs = gridspec.GridSpec(3, 3, figure=fig,
					   height_ratios=[1.4, 0.5, 0.9],
					   hspace=0.30, wspace=0.13,
					   left=0.052, right=0.972, top=0.945, bottom=0.052)

axa = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
axb = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
axc = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
axd = fig.add_subplot(gs[1, :])

gs_bot = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, :],
										  height_ratios=[0.74, 0.26],
										  hspace=0.10)
axe = fig.add_subplot(gs_bot[0])
axf = fig.add_subplot(gs_bot[1], sharex=axe)


# =====================================================================
# (a)  MAP OF THE MARGIN
# =====================================================================
pca = axa.pcolormesh(lon, lat, h, cmap=plt.cm.ocean,
					 norm=mcolors.LogNorm(vmin=10, vmax=4000),
					 shading='auto', zorder=1, rasterized=True)

axa.contour(lon, lat, h, levels=[1000, 2000], colors='0.55', linewidths=0.6,
			zorder=4)
csa = axa.contour(lon, lat, h, levels=[50, 100, 200], colors='k',
				  linewidths=[0.6, 0.6, 1.4], zorder=5)
axa.clabel(csa, fmt='%d', fontsize=7.5, inline=True)

for k in (0, -1):
	axa.plot(lon[k, :], lat[k, :], color='#123a7a', lw=1.5, alpha=0.85,
			 zorder=6)
	axa.plot(lon[:, k], lat[:, k], color='#123a7a', lw=1.5, alpha=0.85,
			 zorder=6)

axa.plot(ds.break_lon, ds.break_lat, color='w', lw=2.8, zorder=7)
axa.plot(ds.break_lon, ds.break_lat, color='k', lw=1.2, zorder=8)

for name, (lo0, la0, lo1, la1) in sections.items():
	axa.plot([lo0, lo1], [la0, la1], color='#8c2d04', lw=2.0, ls='--',
			 zorder=9)
	axa.text(lo1, la1, f' {name}', color='#8c2d04', fontweight='bold',
			 fontsize=9.5, va='center', zorder=16)

for n, name in enumerate(site_names):
	glo = float(ds.site_lon.values[n]); gla = float(ds.site_lat.values[n])
	inside = bool(ds.site_inside.values[n])
	main = name in MAIN_SITES
	axa.plot(glo, gla, marker='*', ms=19 if main else 13,
			 mfc='crimson' if main else 'gold', mec='k',
			 mew=1.1 if main else 0.7, alpha=1.0 if inside else 0.40,
			 ls='none', zorder=15 if main else 14)
	axa.text(glo, gla + 0.28, name + ('*' if main else ''),
			 color='k' if inside else '0.45', fontweight='bold',
			 fontsize=10.5 if main else 9.5, ha='center', va='bottom',
			 zorder=16)

for name, (mlo, mla) in moorings.items():
	axa.plot(mlo, mla, marker='o', ms=6.5, mfc='crimson', mec='k', mew=0.8,
			 ls='none', zorder=15)
	axa.text(mlo - 0.17, mla, name, color='crimson', fontweight='bold',
			 fontsize=9.5, ha='right', va='center', zorder=16)

axa.add_feature(states, facecolor='none', edgecolor='grey', lw=0.4, zorder=11)
dress(axa)
cbar_inside(axa, pca, 'depth (m)', extend='both', width='44%')
panel_letter(axa, 'a', on_map=True)

hl = [Line2D([], [], marker='*', ms=14, mfc='crimson', mec='k', ls='none',
			 label='sites A*, B*'),
	  Line2D([], [], marker='*', ms=10, mfc='gold', mec='k', ls='none',
			 label='sites C to F'),
	  Line2D([], [], marker='o', ms=6, mfc='crimson', mec='k', ls='none',
			 label='moorings'),
	  Line2D([], [], color='k', lw=1.4, label=f'{BREAK:.0f} m, panel (e)'),
	  Line2D([], [], color='#8c2d04', lw=1.8, ls='--', label='sections'),
	  Line2D([], [], color='#123a7a', lw=1.5, label='domain')]
axa.legend(handles=hl, loc='upper right', framealpha=0.93, fontsize=8.5,
		   handlelength=1.6, borderpad=0.4, labelspacing=0.35).set_zorder(29)


# =====================================================================
# (b)  BAROTROPIC M2 CONTROL
# =====================================================================
du = ds.dubt_rel.values
has_u = np.isfinite(du).any()
fld = du if has_u else ds.dzeta_rel.values
lab = (r'$\Delta$ semi major axis of $u_{bt}$ at $M_2$ (%)' if has_u
	   else r'$\Delta$ $M_2$ elevation amplitude (%)')

band = wet & (hraw > 20.) & (hraw < 1000.)
x = np.abs(fld[band]); x = x[np.isfinite(x)]
LIM = float(np.clip(np.ceil(np.nanpercentile(x, 95)), 2., 20.)) if x.size else 5.

pcb = axb.pcolormesh(lon, lat, np.where(wet, fld, np.nan), cmap='PuOr_r',
					 norm=mcolors.TwoSlopeNorm(vmin=-LIM, vcenter=0, vmax=LIM),
					 shading='auto', zorder=1, rasterized=True)
axb.contour(lon, lat, np.where(wet, ds.zeta_amp_ref.values, np.nan),
			levels=[0.5, 1.0, 1.5, 2.0], colors='w', linewidths=0.8, zorder=7)
axb.contour(lon, lat, h, levels=[BREAK], colors='k', linewidths=1.0, zorder=6)
axb.contour(lon, lat, h, levels=[50, 1000], colors='0.45', linewidths=0.4,
			zorder=6)
dress(axb, labels_left=False)
axb.set_title('barotropic $M_2$: REF minus CTRL', fontsize=11, pad=5)
cbar_inside(axb, pcb, lab, extend='both', width='46%')
panel_letter(axb, 'b', on_map=True)

axbi = inset_axes(axb, width='40%', height='26%', loc='upper right',
				  borderpad=1.0)
for v, col, nm in ((ds.dzeta_rel.values, '0.20', r'$\eta_{M_2}$'),
				   (du, 'red', r'$u_{bt}$')):
	q = v[band]; q = q[np.isfinite(q)]
	if q.size:
		axbi.hist(np.clip(q, -LIM, LIM), bins=45, histtype='step', color=col,
				  lw=1.5, density=True, label=nm)
axbi.axvline(0, color='k', lw=0.9)
axbi.set_xlim(-LIM, LIM)
axbi.set_yticks([])
axbi.set_xlabel('REF $-$ CTRL (%)', fontsize=8, labelpad=1.5)
axbi.tick_params(labelsize=7.5)
axbi.legend(fontsize=7.5, framealpha=0.9, loc='upper left', handlelength=1.1,
			borderpad=0.3)
axbi.set_title('20 to 1000 m', fontsize=8, pad=2)
axbi.patch.set_alpha(0.92)
axbi.set_zorder(29)


# =====================================================================
# (c)  SURFACE TRAPPING VALIDITY
# =====================================================================
hr = ds.hratio_ref.isel(window=iw).values
pcc = axc.pcolormesh(lon, lat, np.where(wet, np.minimum(hr, HR_MAX), np.nan),
					 cmap=plt.cm.terrain, vmin=1, vmax=HR_MAX,
					 shading='auto', zorder=1, rasterized=True)
csc = axc.contour(lon, lat, smooth_for_contour(np.where(wet, hr, np.nan)),
				  levels=[TRAP_R], colors='crimson', linewidths=2.0, zorder=7)
axc.contour(lon, lat, h, levels=[BREAK], colors='k', linewidths=1.0, zorder=6)
axc.contour(lon, lat, h, levels=[50, 1000], colors='0.45', linewidths=0.4,
			zorder=6)
dress(axc, labels_left=False)
axc.set_title(f'depth over plume thickness: {WIN} discharge', fontsize=11,
			  pad=5)
cbar_inside(axc, pcc, r'$h/h_p$, REF', extend='max', width='40%')
panel_letter(axc, 'c', on_map=True)

axc.text(0.975, 0.965, f'red, $h = {TRAP_R:.0f}\\,h_p$\nseaward limit of the\n'
		 'criticality argument',
		 transform=axc.transAxes, fontsize=10, va='top', ha='right',
		 color='0.12', zorder=29, linespacing=1.35,
		 bbox=dict(fc='w', ec='crimson', boxstyle='round,pad=0.30',
				   alpha=0.92))


# =====================================================================
# (d)  RESOLUTION AND THE COMMON RESOLVED BAND
# =====================================================================
dbin = ds.depth_bin.values
for tag, col in (('ref', C_REF), ('ctrl', C_CTRL)):
	med = ds[f'ppw_med_{tag}'].isel(window=iw).values
	q25 = ds[f'ppw_q25_{tag}'].isel(window=iw).values
	q75 = ds[f'ppw_q75_{tag}'].isel(window=iw).values
	axd.fill_between(dbin, q25, q75, color=col, alpha=0.16, lw=0)
	axd.plot(dbin, med, color=col, lw=2.4)

axd.axhspan(PPW_LO, PPW_HI, color='0.55', alpha=0.28, lw=0, zorder=0)
axd.axhline(PPW_MIN, color='0.2', lw=1.1, ls='--')
axd.axvline(BREAK, color='k', lw=1.0, ls=':')
axd.text(BREAK*1.06, 0.5, 'shelf break', transform=axd.get_xaxis_transform(),
		 fontsize=12, rotation=90, va='top', color='0.3')
axd.set_xscale('log'); axd.set_yscale('log')
axd.set_xlabel('depth (m)')
axd.set_ylabel(r'points per $\lambda_1 = c_1 T_{M_2}$')
axd.set_xlim(dbin.min(), dbin.max())
axd.grid(alpha=0.25, which='both', lw=0.4)
axd.tick_params(labelsize=10)
panel_letter(axd, 'd')

axd2 = axd.twinx()
fb = ds.frac_res_both.isel(window=iw).values
axd2.fill_between(dbin, 0, 100*fb, color='forestgreen', alpha=0.13, lw=0, zorder=0)
axd2.plot(dbin, 100*fb, color='forestgreen', lw=2.0)
axd2.set_ylim(0, 104)
axd2.set_ylabel('cells resolved in both runs (%)', color='forestgreen')
axd2.tick_params(axis='y', colors='forestgreen', labelsize=10)

hd = [Line2D([], [], color=C_REF, lw=2.4, label='REF, variable discharge'),
	  Line2D([], [], color=C_CTRL, lw=2.4, label='CTRL, constant discharge'),
	  Line2D([], [], color='forestgreen', lw=2.0,
			 label='resolved in both runs, right axis'),
	  Line2D([], [], color='0.2', lw=1.1, ls='--',
			 label=f'{PPW_MIN:.0f} points per wavelength'),
	  Line2D([], [], color='0.55', lw=7, alpha=0.5,
			 label='8 to 12 points, Hallberg (2013)')]
legend_above(axd, hd, ncol=5)
axd.text(1.0, 1.015, f'{WIN} discharge', transform=axd.transAxes,
		 ha='right', va='bottom', fontsize=10, color='0.35')


# =====================================================================
# (e)  CRITICALITY ALONG THE 200 m ISOBATH
# =====================================================================
blat = ds.break_lat.values
ar = ds.alpha_ref_break.isel(window=iw).values
ac = ds.alpha_ctrl_break.isel(window=iw).values
da = ds.dalpha_break.isel(window=iw).values

axe.axhspan(0, 1, color='tan', alpha=0.55, lw=0, zorder=0)
axe.axhline(1.0, color='k', lw=1.3, ls='--', zorder=3)
axe.text(-1, 0.75, r'$\alpha=1$', fontsize=9.5,
		 va='bottom', color='0.2')

for jw, w in enumerate(wins):
	if jw == iw:
		continue
	axe.plot(blat, ds.alpha_ref_break.isel(window=jw).values, color=C_REF,
			 lw=0.8, alpha=0.28)
axe.plot(blat, ac, color=C_CTRL, lw=2.0)
axe.plot(blat, ar, color=C_REF, lw=2.0)

for n, name in enumerate(site_names):
	if not bool(ds.site_inside.values[n]):
		continue
	gla = float(ds.site_lat.values[n])
	if not (blat.min() <= gla <= blat.max()):
		continue
	main = name in MAIN_SITES
	for ax_ in (axe, axf):
		ax_.axvline(gla, color='crimson' if main else '0.65',
					lw=1.2 if main else 0.7, ls='-' if main else ':',
					alpha=0.85, zorder=1)
	axe.text(gla, 0.025, name + ('*' if main else ''),
			 transform=axe.get_xaxis_transform(), ha='center', va='bottom',
			 fontsize=10 if main else 9,
			 fontweight='bold' if main else 'normal',
			 color='crimson' if main else '0.4')

axe.set_ylabel(r'criticality $\alpha = |\nabla h| / s$')
axe.set_xlim(blat.min(), blat.max())
axe.set_ylim(0, min(3.5, np.nanpercentile(np.r_[ar, ac], 99.5)*1.18))
axe.grid(alpha=0.25, lw=0.4)
axe.tick_params(labelbottom=False, labelsize=10)
panel_letter(axe, 'e')

he = [Line2D([], [], color=C_REF, lw=2.0, label='REF, variable discharge'),
	  Line2D([], [], color=C_CTRL, lw=2.0, label='CTRL, constant discharge'),
	  Line2D([], [], color=C_REF, lw=0.8, alpha=0.4,
			 label='REF, rising and decreasing windows'),
	  Line2D([], [], color='tan', lw=7, label='subcritical')]
legend_above(axe, he, ncol=4)
axe.text(1.0, 1.015, f'{WIN} discharge', transform=axe.transAxes,
		 ha='right', va='bottom', fontsize=10, color='0.35')


# =====================================================================
# (f)  THE DIFFERENCE IN CRITICALITY BETWEEN THE RUNS
# =====================================================================
axf.axhline(0, color='k', lw=0.9)
axf.plot(blat, da, color='0.12', lw=1.1)
axf.fill_between(blat, 0, da, color='0.5', alpha=0.45, lw=0)
axf.set_xlabel(f'latitude along the {BREAK:.0f} m isobath')
axf.set_ylabel(r'$\Delta\alpha$', labelpad=8)
axf.set_xlim(blat.min(), blat.max())
lim = np.nanmax(np.abs(da))*1.25 if np.isfinite(da).any() else 1e-3
axf.set_ylim(-max(lim, 1e-3), max(lim, 1e-3))
axf.grid(alpha=0.25, lw=0.4)
axf.tick_params(labelsize=10)
panel_letter(axf, 'f')

if np.isfinite(da).any() and np.isfinite(ar).any():
	axf.text(0.995, 0.06,
			 f'REF $-$ CTRL,  median $|\\Delta\\alpha|$ = '
			 f'{np.nanmedian(np.abs(da)):.3f},  along break sd of '
			 f'$\\alpha$ = {np.nanstd(ar):.2f}',
			 transform=axf.transAxes, fontsize=9.5, va='bottom', ha='right',
			 color='0.15',
			 bbox=dict(fc='w', ec='0.6', boxstyle='round,pad=0.28',
					   alpha=0.9))


# =====================================================================
fig.savefig(OUTFIG, dpi=DPI, bbox_inches='tight')
print('wrote', OUTFIG)