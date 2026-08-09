#################################
"""
fig05_plot.py
=================================================================
Figure 5  -  Coherence made at the source, lost in the plume

Flat script. Every panel has its own axes handle and every curve is
drawn by its own call, so any of them can be removed or restyled
without touching the rest.

	axa  gs[0,0]    map, coherent fraction of the mode 1 flux, REF
	axb  gs[0,1]    map, mode 2, REF
	axc  gs[0,2]    map, mode 3, REF
	axd  gs[0,3]    map, mode 1 coherent fraction, REF minus CTRL
	axe  gs[1,0:2]  coherent modal sea level against fit length,
					four stations along the path from site A
	axf  gs[1,2]    coherent fraction against fit length per mode,
					shelf band, both runs
	axg  gs[1,3]    coherent fraction along the path per mode, both runs
	axh  gs[2,0]    decorrelation timescale against the river forcing
	axi  gs[2,1]    decorrelation timescale against |d hp / dt|
	axj  gs[2,2:4]  coherent and incoherent M2 sea level per mode in the
					SWOT box, against Tchilibou et al. (2025)

Reads fig05_data.nc written by fig05_compute.py.

--- coherent fraction against fit length, shelf band ---
  ref mode 1   2.1d 1.00   4.2d 0.95   6.2d 0.92   8.3d 0.90  12.5d 0.87  16.7d 0.86  20.8d 0.85  25.0d 0.84  33.3d 0.84  41.7d 0.84  50.0d 0.85  66.7d 0.85
  ref mode 2   2.1d 1.00   4.2d 0.88   6.2d 0.81   8.3d 0.76  12.5d 0.70  16.7d 0.65  20.8d 0.62  25.0d 0.60  33.3d 0.57  41.7d 0.55  50.0d 0.54  66.7d 0.54
  ref mode 3   2.1d 1.00   4.2d 0.88   6.2d 0.82   8.3d 0.77  12.5d 0.71  16.7d 0.67  20.8d 0.65  25.0d 0.63  33.3d 0.60  41.7d 0.58  50.0d 0.58  66.7d 0.57
 ctrl mode 1   2.1d 1.00   4.2d 0.94   6.2d 0.91   8.3d 0.88  12.5d 0.85  16.7d 0.84  20.8d 0.83  25.0d 0.82  33.3d 0.81  41.7d 0.81  50.0d 0.82  66.7d 0.83
 ctrl mode 2   2.1d 1.00   4.2d 0.88   6.2d 0.81   8.3d 0.76  12.5d 0.69  16.7d 0.64  20.8d 0.61  25.0d 0.58  33.3d 0.54  41.7d 0.52  50.0d 0.50  66.7d 0.49
 ctrl mode 3   2.1d 1.00   4.2d 0.89   6.2d 0.84   8.3d 0.80  12.5d 0.75  16.7d 0.72  20.8d 0.70  25.0d 0.69  33.3d 0.67  41.7d 0.67  50.0d 0.67  66.7d 0.67

--- coherent fraction against fit length, deep band ---
  ref mode 1   2.1d 1.00   4.2d 0.97   6.2d 0.95   8.3d 0.93  12.5d 0.91  16.7d 0.89  20.8d 0.87  25.0d 0.86  33.3d 0.85  41.7d 0.85  50.0d 0.84  66.7d 0.83
  ref mode 2   2.1d 1.00   4.2d 0.97   6.2d 0.94   8.3d 0.93  12.5d 0.90  16.7d 0.88  20.8d 0.87  25.0d 0.86  33.3d 0.85  41.7d 0.85  50.0d 0.84  66.7d 0.83
  ref mode 3   2.1d 1.00   4.2d 0.92   6.2d 0.87   8.3d 0.83  12.5d 0.78  16.7d 0.74  20.8d 0.72  25.0d 0.70  33.3d 0.67  41.7d 0.65  50.0d 0.63  66.7d 0.61
 ctrl mode 1   2.1d 1.00   4.2d 0.97   6.2d 0.95   8.3d 0.93  12.5d 0.91  16.7d 0.89  20.8d 0.88  25.0d 0.87  33.3d 0.86  41.7d 0.85  50.0d 0.84  66.7d 0.82
 ctrl mode 2   2.1d 1.00   4.2d 0.97   6.2d 0.94   8.3d 0.92  12.5d 0.89  16.7d 0.87  20.8d 0.85  25.0d 0.84  33.3d 0.82  41.7d 0.81  50.0d 0.80  66.7d 0.78
 ctrl mode 3   2.1d 1.00   4.2d 0.94   6.2d 0.90   8.3d 0.87  12.5d 0.82  16.7d 0.79  20.8d 0.77  25.0d 0.75  33.3d 0.72  41.7d 0.71  50.0d 0.70  66.7d 0.67

--- decorrelation timescale, days, shelf band ---
  ref mode 1  median    nan   resolved in   0.0% of the centres
  ref mode 2  median    nan   resolved in   0.0% of the centres
  ref mode 3  median    nan   resolved in   0.0% of the centres
 ctrl mode 1  median    nan   resolved in   0.0% of the centres
 ctrl mode 2  median    nan   resolved in   0.0% of the centres
 ctrl mode 3  median    nan   resolved in   0.0% of the centres

--- M2 baroclinic sea level in the SWOT box, cm std ---
  ref mode 1  coherent 0.874  incoherent 0.733  total 1.141  coherent share of variance  58.7%
  ref mode 2  coherent 0.485  incoherent 0.343  total 0.594  coherent share of variance  66.6%
  ref mode 3  coherent 0.060  incoherent 0.056  total 0.082  coherent share of variance  53.5%
 ctrl mode 1  coherent 0.766  incoherent 0.687  total 1.029  coherent share of variance  55.4%
 ctrl mode 2  coherent 0.535  incoherent 0.403  total 0.670  coherent share of variance  63.8%
 ctrl mode 3  coherent 0.058  incoherent 0.063  total 0.086  coherent share of variance  45.9%
  SWOT track 20 for reference, total SLA std 1.03, 0.58 and 0.74 cm for mode 1, mode 2 and the higher modes, with 24, 16 and 4 per cent of the variance removed by the coherent atlas (Tchilibou et al. 2025, their Table 2)
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
import cmocean.cm as cmo

plt.rcParams.update({
	'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
	'legend.fontsize': 8, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
	'axes.linewidth': 0.9,
})

# ---------------------------------------------------------------------
DATA = 'fig05_data.nc'
OUTFIG = 'Fig5.jpeg'
DPI = 300

C_REF = '#0b3d91'
C_CTRL = '#c1440e'
MODE_C = ['#1b3b6f', '#4e9f3d', '#d1495b']      # modes 1, 2, 3
ST_C = ['#264653', '#2a9d8f', '#e9c46a', '#e76f51']
TICK_FS = 7

# stations drawn in panel (e), indices into the station coordinate
ST1, ST2, ST3, ST4 = 1, 3, 5, 7

# SWOT track 20 off the Amazon shelf, Tchilibou et al. (2025), their
# Table 2. Standard deviation of the SLA in each wavelength band and the
# share of the variance removed by the coherent atlas, which is the
# coherent share plotted as the reference in panel (j).
SWOT_STD = np.array([1.03, 0.58, 0.74])         # cm, mode 1, 2, higher
SWOT_COH_SHARE = np.array([0.24, 0.16, 0.04])   # variance fraction
SWOT_COH = SWOT_STD*np.sqrt(SWOT_COH_SHARE)
SWOT_INC = SWOT_STD*np.sqrt(1. - SWOT_COH_SHARE)

ds = xr.open_dataset(DATA)
lon = ds.lon_rho.values
lat = ds.lat_rho.values
h = np.where(ds.mask_rho.values > 0, ds.h.values, np.nan)
mask = ds.mask_rho.values
wins = [str(w) for w in ds.window.values]
WIN = ds.attrs.get('plot_window', 'peak')
iw = wins.index(WIN)
BREAK = 200.
GCUT = float(ds.attrs.get('gamma_cut', np.exp(-1.)))
L = ds.fitlen.values
pdist = ds.path_dist.values
tb = ds.block_time.values

land = cfeature.NaturalEarthFeature('physical', 'land', scale='10m')


def dress(ax, labels_left=True):
	ax.add_feature(land, facecolor='wheat', edgecolor='grey', lw=0.4, zorder=8)
	ax.coastlines(resolution='10m', linewidths=0.4, zorder=9)
	ax.contour(lon, lat, h, levels=[BREAK], colors='k', linewidths=1.0,
			   zorder=6)
	ax.contour(lon, lat, h, levels=[50, 1000], colors='0.45', linewidths=0.4,
			   zorder=6)
	ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()],
				  crs=ccrs.PlateCarree())
	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.2,
					  lw=0.4, color='grey', ls='--', zorder=10)
	gl.top_labels = False
	gl.right_labels = False
	gl.left_labels = labels_left
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': TICK_FS}
	gl.ylabel_style = {'size': TICK_FS}
	return gl


def panel_letter(ax, s, on_map=False):
	kw = dict(fontsize=15, fontweight='bold', va='top', ha='left')
	if on_map:
		ax.text(0.025, 0.965, s, transform=ax.transAxes, zorder=20,
				bbox=dict(fc='w', ec='0.4', boxstyle='round,pad=0.2',
						  alpha=0.9), **kw)
	else:
		ax.text(0.03, 0.965, s, transform=ax.transAxes, **kw)


# =====================================================================
# FIGURE SKELETON
# =====================================================================
fig = plt.figure(figsize=(18.0, 15.5))
gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1,1,1],
					   hspace=0.2, wspace=0.25,
					   left=0.05, right=0.975, top=0.96, bottom=0.05)

axa = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
axb = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
axc = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
axd = fig.add_subplot(gs[0, 3], projection=ccrs.PlateCarree())
axe = fig.add_subplot(gs[1, 0:2])
axf = fig.add_subplot(gs[1, 2:4])
axg = fig.add_subplot(gs[2, 0:2])
axj = fig.add_subplot(gs[2, 2:4])


# =====================================================================
# (a)  COHERENT FRACTION OF THE MODE 1 FLUX, REF
# =====================================================================
g1r = ds.gammaF_ref.isel(window=iw, mode=0).values
pa = axa.pcolormesh(lon, lat, np.where(mask > 0, g1r, np.nan),
					cmap=cmo.thermal, vmin=0, vmax=1, shading='auto',
					zorder=1, rasterized=True)
axa.plot(ds.path_lon.values, ds.path_lat.values, color='crimson', lw=1.6,
		 zorder=12, transform=ccrs.PlateCarree())
axa.plot(ds.station_lon.values, ds.station_lat.values, ls='none', marker='o',
		 ms=5, mfc='w', mec='crimson', mew=1.1, zorder=13,
		 transform=ccrs.PlateCarree())
dress(axa)
axa.set_title(f'mode 1, REF, {WIN} discharge', fontsize=10)
panel_letter(axa, 'a', on_map=True)

cax_a = inset_axes(axa, width='40%', height='2.8%', loc='lower left',
				   bbox_to_anchor=(0.05, 0.12, 1, 1),
				   bbox_transform=axa.transAxes, borderpad=0)
cb_a = fig.colorbar(pa, cax=cax_a, orientation='horizontal')
cb_a.set_label(r'coherent fraction $\gamma_n$', fontsize=7.5)
cb_a.ax.tick_params(labelsize=6.5)
cax_a.set_zorder(30)


# =====================================================================
# (b)  COHERENT FRACTION OF THE MODE 2 FLUX, REF
# =====================================================================
g2r = ds.gammaF_ref.isel(window=iw, mode=1).values
pb = axb.pcolormesh(lon, lat, np.where(mask > 0, g2r, np.nan),
					cmap=cmo.thermal, vmin=0, vmax=1, shading='auto',
					zorder=1, rasterized=True)
dress(axb, labels_left=False)
axb.set_title('mode 2, REF', fontsize=10)
panel_letter(axb, 'b', on_map=True)


# =====================================================================
# (c)  COHERENT FRACTION OF THE MODE 3 FLUX, REF
# =====================================================================
g3r = ds.gammaF_ref.isel(window=iw, mode=2).values
pc = axc.pcolormesh(lon, lat, np.where(mask > 0, g3r, np.nan),
					cmap=cmo.thermal, vmin=0, vmax=1, shading='auto',
					zorder=1, rasterized=True)
dress(axc, labels_left=False)
axc.set_title('mode 3, REF', fontsize=10)
panel_letter(axc, 'c', on_map=True)


# =====================================================================
# (d)  MODE 1 COHERENT FRACTION, REF MINUS CTRL
# =====================================================================
g1c = ds.gammaF_ctrl.isel(window=iw, mode=0).values
dg = g1r - g1c
lim = float(np.nanpercentile(np.abs(dg[mask > 0]), 97))
lim = lim if np.isfinite(lim) and lim > 0 else 0.3
pd_ = axd.pcolormesh(lon, lat, np.where(mask > 0, dg, np.nan),
					 cmap='RdBu_r',
					 norm=mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
					 shading='auto', zorder=1, rasterized=True)
axd.contour(lon, lat, ds.c1_ref.isel(window=iw).values, levels=[0.8],
			colors='k', linewidths=0.9, zorder=7)
dress(axd, labels_left=False)
axd.set_title(r'mode 1, $\gamma$ REF minus CTRL', fontsize=10)
panel_letter(axd, 'd', on_map=True)

cax_d = inset_axes(axd, width='40%', height='2.8%', loc='lower left',
				   bbox_to_anchor=(0.05, 0.12, 1, 1),
				   bbox_transform=axd.transAxes, borderpad=0)
cb_d = fig.colorbar(pd_, cax=cax_d, orientation='horizontal', extend='both')
cb_d.set_label(r'$\Delta\gamma_1$', fontsize=7.5)
cb_d.ax.tick_params(labelsize=6.5)
cax_d.set_zorder(30)


# =====================================================================
# (e)  COHERENT SEA LEVEL AGAINST FIT LENGTH, STATIONS ALONG THE PATH
# =====================================================================
sd = ds.station_dist.values

axe.plot(L, ds.Acoh_L_ref.isel(station=ST1, mode=0).values, color=ST_C[0],
		 lw=2.0, marker='o', ms=3.5, label=f'{sd[ST1]:+.0f} km, REF')
axe.plot(L, ds.Acoh_L_ctrl.isel(station=ST1, mode=0).values, color=ST_C[0],
		 lw=1.2, ls='--', label=f'{sd[ST1]:+.0f} km, CTRL')

axe.plot(L, ds.Acoh_L_ref.isel(station=ST2, mode=0).values, color=ST_C[1],
		 lw=2.0, marker='o', ms=3.5, label=f'{sd[ST2]:+.0f} km, REF')
axe.plot(L, ds.Acoh_L_ctrl.isel(station=ST2, mode=0).values, color=ST_C[1],
		 lw=1.2, ls='--', label=f'{sd[ST2]:+.0f} km, CTRL')

axe.plot(L, ds.Acoh_L_ref.isel(station=ST3, mode=0).values, color=ST_C[2],
		 lw=2.0, marker='o', ms=3.5, label=f'{sd[ST3]:+.0f} km, REF')
axe.plot(L, ds.Acoh_L_ctrl.isel(station=ST3, mode=0).values, color=ST_C[2],
		 lw=1.2, ls='--', label=f'{sd[ST3]:+.0f} km, CTRL')

axe.plot(L, ds.Acoh_L_ref.isel(station=ST4, mode=0).values, color=ST_C[3],
		 lw=2.0, marker='o', ms=3.5, label=f'{sd[ST4]:+.0f} km, REF')
axe.plot(L, ds.Acoh_L_ctrl.isel(station=ST4, mode=0).values, color=ST_C[3],
		 lw=1.2, ls='--', label=f'{sd[ST4]:+.0f} km, CTRL')

axe.set_xscale('log')
axe.set_xlabel('fit length (days)')
axe.set_ylabel(r'coherent mode 1 $M_2$ sea level (cm)')
axe.grid(alpha=0.22, lw=0.4)
axe.legend(loc=3, ncol=2, fontsize=7, framealpha=0.9)
axe.set_title('distance along the path from site A, negative shoreward',
			  fontsize=9)
panel_letter(axe, 'e')


# =====================================================================
# (f)  COHERENT FRACTION AGAINST FIT LENGTH, SHELF BAND
# =====================================================================
axf.plot(L, ds.gam_L_reg_ref.isel(region=0, mode=0).values, color=MODE_C[0],
		 lw=2.0, marker='o', ms=3.5, label='mode 1 REF')
axf.plot(L, ds.gam_L_reg_ref.isel(region=0, mode=1).values, color=MODE_C[1],
		 lw=2.0, marker='o', ms=3.5, label='mode 2 REF')
axf.plot(L, ds.gam_L_reg_ref.isel(region=0, mode=2).values, color=MODE_C[2],
		 lw=2.0, marker='o', ms=3.5, label='mode 3 REF')
axf.plot(L, ds.gam_L_reg_ctrl.isel(region=0, mode=0).values, color=MODE_C[0],
		 lw=1.2, ls='--', label='mode 1 CTRL')
axf.plot(L, ds.gam_L_reg_ctrl.isel(region=0, mode=1).values, color=MODE_C[1],
		 lw=1.2, ls='--', label='mode 2 CTRL')
axf.plot(L, ds.gam_L_reg_ctrl.isel(region=0, mode=2).values, color=MODE_C[2],
		 lw=1.2, ls='--', label='mode 3 CTRL')

axf.axhline(GCUT, color='0.35', lw=1.0, ls=':')
axf.text(L[1], GCUT + 0.02, r'$1/e$', fontsize=8, color='0.35')
axf.set_xscale('log')
axf.set_ylim(0, 1.02)
axf.set_xlabel('fit length (days)')
axf.set_ylabel(r'coherent fraction $\gamma_n$')
axf.set_title('shelf band, 30 to 300 m', fontsize=9)
axf.grid(alpha=0.22, lw=0.4)
axf.legend(loc='lower left', ncol=2, fontsize=7, framealpha=0.9)
panel_letter(axf, 'f')


# =====================================================================
# (g)  COHERENT FRACTION ALONG THE PROPAGATION PATH
# =====================================================================
axg.plot(pdist, ds.gam_path_ref.isel(window=iw, mode=0).values,
		 color=MODE_C[0], lw=2.0, label='mode 1 REF')
axg.plot(pdist, ds.gam_path_ref.isel(window=iw, mode=1).values,
		 color=MODE_C[1], lw=2.0, label='mode 2 REF')
axg.plot(pdist, ds.gam_path_ref.isel(window=iw, mode=2).values,
		 color=MODE_C[2], lw=2.0, label='mode 3 REF')
axg.plot(pdist, ds.gam_path_ctrl.isel(window=iw, mode=0).values,
		 color=MODE_C[0], lw=1.2, ls='--', label='mode 1 CTRL')
axg.plot(pdist, ds.gam_path_ctrl.isel(window=iw, mode=1).values,
		 color=MODE_C[1], lw=1.2, ls='--', label='mode 2 CTRL')
axg.plot(pdist, ds.gam_path_ctrl.isel(window=iw, mode=2).values,
		 color=MODE_C[2], lw=1.2, ls='--', label='mode 3 CTRL')

axg.axvline(0., color='k', lw=1.0, ls=':')
axg.text(2., 0.04, 'site A', fontsize=8, rotation=90, color='0.3')
axg.axhline(GCUT, color='0.35', lw=1.0, ls=':')
axg.set_xlabel('distance along the path (km)')
axg.set_ylabel(r'coherent fraction $\gamma_n$')
axg.set_ylim(0, 1.02)
axg.set_title(f'{WIN} discharge, shoreward to the left', fontsize=9)
axg.grid(alpha=0.22, lw=0.4)
axg.legend(loc='lower right', ncol=2, fontsize=7, framealpha=0.9)
panel_letter(axg, 'g')

# =====================================================================
# (j)  COHERENT AND INCOHERENT M2 SEA LEVEL IN THE SWOT BOX
# =====================================================================
x = np.array([0., 1., 2.])
w = 0.26

axj.bar(x - w, ds.ssh_coh_ref.values, width=w, color=C_REF, edgecolor='k',
		lw=0.5, label='coherent REF')
axj.bar(x - w, ds.ssh_inc_ref.values, width=w, bottom=ds.ssh_coh_ref.values,
		color=C_REF, alpha=0.35, edgecolor='k', lw=0.5,
		label='incoherent REF')
axj.bar(x, ds.ssh_coh_ctrl.values, width=w, color=C_CTRL, edgecolor='k',
		lw=0.5, label='coherent CTRL')
axj.bar(x, ds.ssh_inc_ctrl.values, width=w, bottom=ds.ssh_coh_ctrl.values,
		color=C_CTRL, alpha=0.35, edgecolor='k', lw=0.5,
		label='incoherent CTRL')
axj.bar(x + w, SWOT_COH, width=w, color='0.35', edgecolor='k', lw=0.5,
		label='coherent SWOT')
axj.bar(x + w, SWOT_INC, width=w, bottom=SWOT_COH, color='0.35', alpha=0.35,
		edgecolor='k', lw=0.5, label='incoherent SWOT')

axj.set_xticks(x)
axj.set_xticklabels(['mode 1', 'mode 2', 'mode 3 and higher'])
axj.set_ylabel(r'$M_2$ baroclinic sea level, standard deviation (cm)')
axj.set_title('SWOT track 20 box, deeper than 1000 m, '
			  'reference from Tchilibou et al. (2025)', fontsize=9)
axj.grid(alpha=0.22, lw=0.4, axis='y')
axj.legend(loc='upper right', ncol=3, fontsize=7, framealpha=0.9)
panel_letter(axj, 'h')


# =====================================================================
fig.savefig(OUTFIG, dpi=DPI, bbox_inches='tight')
print('wrote', OUTFIG)





