##############
"""
fig02_plot.py
=================================================================
Figure 2  -  The plume as a waveguide

Flat script. Every panel has its own axes handle:

	axa  gs[0, 0:2]   c1 map, rising
	axb  gs[0, 2:4]   c1 map, peak
	axc  gs[0, 4:6]   c1 map, decreasing
	axd  gs[1, 0:3]   dc1 = REF - CTRL at peak, with the 35 isohaline
	axe1 gs[1, 3]     mode structure, inner shelf
	axe2 gs[1, 4]     mode structure, mid shelf
	axe3 gs[1, 5]     mode structure, shelf break
	axf1 gs[2, 0:3]   deformation radius against shelf width
	axf2 gs[2, 3:6]   surface trapping index against plume thickness

Reads fig02_data.nc written by fig02_compute.py.
=================================================================

--- Figure 2 summary, from fig02_data.nc ---
ratios use only cells where BOTH runs support a mode one, c1 > 0.05 m/s

	 rising  outer c1 REF  0.78 CTRL  0.34 (x2.25, +0.43 m/s, 100% of cells)
			 mid   c1 REF  0.60 CTRL  0.33 (x1.80, +0.26 m/s,  97% of cells)
			 trapping index REF 0.73 CTRL 0.57   plume thickness  7.7 m   Reff REF   128 CTRL    53 km
			 CTRL supports NO mode one on    2% of the mid shelf, where REF does

	   peak  outer c1 REF  1.02 CTRL  0.59 (x1.72, +0.43 m/s,  99% of cells)
			 mid   c1 REF  0.69 CTRL  0.27 (x2.59, +0.42 m/s,  85% of cells)
			 trapping index REF 0.72 CTRL 0.61   plume thickness  8.5 m   Reff REF   154 CTRL    81 km
			 CTRL supports NO mode one on   14% of the mid shelf, where REF does

 decreasing  outer c1 REF  0.99 CTRL  0.51 (x1.93, +0.48 m/s,  92% of cells)
			 mid   c1 REF  0.86 CTRL  0.47 (x1.82, +0.39 m/s,  40% of cells)
			 trapping index REF 0.76 CTRL 0.66   plume thickness  8.5 m   Reff REF   135 CTRL    59 km
			 CTRL supports NO mode one on   57% of the mid shelf, where REF does

median shelf width 195 km, range 52 to 277 km

  Reff / shelf width,      rising   REF 0.65   CTRL 0.27
  Reff / shelf width,        peak   REF 0.79   CTRL 0.41
  Reff / shelf width,  decreasing   REF 0.69   CTRL 0.30

Below one means the shelf is WIDER than a deformation radius, so
mode one is rotationally trapped within Reff of the coast and the
cross shelf propagation is not a free non rotating problem.

two layer estimate departs from the full solution by 13% on the outer shelf

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
	'font.size': 10,
	'axes.labelsize': 10,
	'axes.titlesize': 11,
	'legend.fontsize': 8,
	'xtick.labelsize': 9,
	'ytick.labelsize': 9,
	'axes.linewidth': 0.9,
})

# ---------------------------------------------------------------------
DATA   = 'fig02_data.nc'
OUTFIG = 'Fig2.jpeg'
DPI    = 300

C_REF  = '#0b3d91'
C_CTRL = '#c1440e'

# c1 colour scale. 'log' keeps the shelf range legible without turning the
# deep ocean into a saturated blob, since c1 there exceeds 2 m s-1. Set to
# 'linear' with C1_MAX to go back to the previous look.
C1_SCALE = 'log'      # 'log' or 'linear'
C1_MIN_CB = 0.05      # m s-1, bottom of the log scale
C1_MAX = 3.0          # m s-1, top of the c1 colour scale
DC1_MAX = 0.8         # m s-1, half range of the difference colour scale

CBAR_IN_EACH = True   # a colourbar inside every top row map, over the land
TICK_FS = 7           # font size of the lon and lat tick labels
TOP_WSPACE = 0.04     # horizontal gap between the three top row maps

ds = xr.open_dataset(DATA)

lon = ds.lon_rho.values
lat = ds.lat_rho.values
h = np.where(ds.mask_rho.values > 0, ds.h.values, np.nan)
mask = ds.mask_rho.values
wins = [str(w) for w in ds.window.values]
WIN = ds.attrs.get('plot_window', 'peak')
iw = wins.index(WIN)
ISO = float(ds.attrs.get('isohaline', 35.))
SHELF_MAX = float(ds.attrs.get('shelf_max_depth', 250.))

land = cfeature.NaturalEarthFeature('physical', 'land', scale='10m')

def dress(ax, labels_left=True, labels_bottom=True):
	ax.add_feature(land, facecolor='wheat', edgecolor='grey', lw=0.4, zorder=8)
	ax.coastlines(resolution='10m', linewidths=0.4, zorder=9)
	ax.contour(lon, lat, h, levels=[200], colors='k', linewidths=0.9, zorder=6)
	ax.contour(lon, lat, h, levels=[50, 1000], colors='0.45',
			   linewidths=0.4, zorder=6)
	ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()],
				  crs=ccrs.PlateCarree())
	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.2,
					  lw=0.4, color='grey', ls='--', zorder=10)
	gl.top_labels = False; gl.right_labels = False
	gl.left_labels = labels_left; gl.bottom_labels = labels_bottom
	gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': TICK_FS}
	gl.ylabel_style = {'size': TICK_FS}
	return gl


def cbar_on_land(ax, mappable, label, loc='lower left',
				 anchor=(0.05, 0.1, 1, 1), width='42%', height='3.0%',
				 extend='max'):
	"""Horizontal colourbar placed inside the axes, over the land mask."""
	cax = inset_axes(ax, width=width, height=height, loc=loc,
					 bbox_to_anchor=anchor, bbox_transform=ax.transAxes,
					 borderpad=0)
	cb = plt.colorbar(mappable, cax=cax, orientation='horizontal',
					  extend=extend)
	cb.set_label(label, fontsize=7.5, labelpad=1)
	cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
	cb.outline.set_linewidth(0.6)
	cax.set_zorder(30)
	return cb
# =====================================================================
# FIGURE SKELETON
# =====================================================================
fig = plt.figure(figsize=(16.5, 13.5))
gs = gridspec.GridSpec(3, 6, figure=fig,
					   height_ratios=[1.0, 1.05, 0.80],
					   hspace=0.26, wspace=0.55,
					   left=0.05, right=0.98, top=0.965, bottom=0.055)

gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :],
										  wspace=TOP_WSPACE)
axa = fig.add_subplot(gs_top[0], projection=ccrs.PlateCarree())
axb = fig.add_subplot(gs_top[1], projection=ccrs.PlateCarree())
axc = fig.add_subplot(gs_top[2], projection=ccrs.PlateCarree())
axd = fig.add_subplot(gs[1, 0:3], projection=ccrs.PlateCarree())
axe1 = fig.add_subplot(gs[1, 3])
axe2 = fig.add_subplot(gs[1, 4])
axe3 = fig.add_subplot(gs[1, 5])
axf1 = fig.add_subplot(gs[2, 0:3])
axf2 = fig.add_subplot(gs[2, 3:6])


# =====================================================================
# (a) (b) (c)  MODE ONE PHASE SPEED, THREE WINDOWS, REF
# =====================================================================
if C1_SCALE == 'log':
	norm_c1 = mcolors.LogNorm(vmin=C1_MIN_CB, vmax=C1_MAX)
	cb_extend = 'both'
else:
	norm_c1 = mcolors.Normalize(vmin=0, vmax=C1_MAX)
	cb_extend = 'max'
maps = [(axa, 'rising', 'a'), (axb, 'peak', 'b'), (axc, 'decreasing', 'c')]

for ax, wname, tag in maps:
	if wname not in wins:
		continue
	k = wins.index(wname)
	c1 = ds.c1_ref.isel(window=k).values
	pcm = ax.pcolormesh(lon, lat, np.where(mask > 0, c1, np.nan),
						cmap=plt.cm.gist_stern_r, norm=norm_c1, shading='auto',
						zorder=1, rasterized=True)
	# the plume edge, for context
	sss = ds.sss_ref.isel(window=k).values
	ax.contour(lon, lat, sss, levels=[ISO], colors='w', linewidths=1.6,
			   zorder=7)
	dress(ax, labels_left=(tag == 'a'))
	ax.set_title(f'{wname}', fontsize=11, pad=4)
	ax.text(0.03, 0.96, tag, transform=ax.transAxes, fontsize=15,
			fontweight='bold', va='top', ha='left', zorder=20,
			bbox=dict(fc='w', ec='0.4', boxstyle='round,pad=0.2', alpha=0.9))
	if CBAR_IN_EACH or tag == 'a':
		cbar_on_land(ax, pcm, r'$c_1$ (m s$^{-1}$)', extend=cb_extend)


# =====================================================================
# (d) dc1 = REF - CTRL AT PEAK, WITH THE SURFACE 35 ISOHALINE
# =====================================================================
dc1 = ds.dc1.isel(window=iw).values
pcd = axd.pcolormesh(lon, lat, np.where(mask > 0, dc1, np.nan),
					 cmap=cmo.balance,
					 norm=mcolors.TwoSlopeNorm(vmin=-DC1_MAX, vcenter=0,
											   vmax=DC1_MAX),
					 shading='auto', zorder=1, rasterized=True)

sss_r = ds.sss_ref.isel(window=iw).values
sss_c = ds.sss_ctrl.isel(window=iw).values
ci = axd.contour(lon, lat, sss_r, levels=[ISO], colors='k', linewidths=2.2,
				 zorder=7)
axd.contour(lon, lat, sss_c, levels=[ISO], colors='k', linewidths=1.2,
			linestyles='--', zorder=7)

# the profile stations
axd.plot(ds.station_lon.values, ds.station_lat.values, marker='o', ms=7,
		 mfc='yellow', mec='k', mew=1.0, ls='none', zorder=15)
for n, s in enumerate(ds.station.values):
	axd.text(float(ds.station_lon.values[n]) - 0.12,
			 float(ds.station_lat.values[n]), str(s),
			 fontsize=9, fontweight='bold', ha='right', va='center', zorder=16)

dress(axd)
axd.set_title(f'{WIN} discharge', fontsize=10, color='0.3', loc='right')
axd.text(0.02, 0.97, 'd', transform=axd.transAxes, fontsize=15,
		 fontweight='bold', va='top', ha='left', zorder=20,
		 bbox=dict(fc='w', ec='0.4', boxstyle='round,pad=0.2', alpha=0.9))

hl = [plt.Line2D([], [], color='k', lw=2.2, label=f'{ISO:.0f} isohaline, REF'),
	  plt.Line2D([], [], color='k', lw=1.2, ls='--',
				 label=f'{ISO:.0f} isohaline, CTRL'),
	  plt.Line2D([], [], marker='o', ms=6, mfc='yellow', mec='k', ls='none',
				 label='profile stations')]
axd.legend(handles=hl, loc='lower left', framealpha=0.9, fontsize=8)

if CBAR_IN_EACH :
	cbar_on_land(axd, pcd, r'$\Delta c_1$ = REF $-$ CTRL (m s$^{-1}$)', extend='both')
#	
#cax_d = inset_axes(axd, width='4.5%', height='75%', loc='center left',
				   #bbox_to_anchor=(1.03, 0., 1, 1),
				   #bbox_transform=axd.transAxes, borderpad=0)
#cbd = fig.colorbar(pcd, cax=cax_d, extend='both')
#
#
#cbd.set_label(r'$\Delta c_1$ = REF $-$ CTRL (m s$^{-1}$)', fontsize=9)
#cbd.ax.tick_params(labelsize=8)


# =====================================================================
# (e) MODE ONE STRUCTURE AT THREE STATIONS
# =====================================================================
stations = [str(s) for s in ds.station.values]
want = [s for s in ['M1', 'M2', 'M3', 'V200'] if s in stations][:3]
axes_e = [axe1, axe2, axe3]
labels_e = ['e', '', '']

for m, (ax, sname) in enumerate(zip(axes_e, want)):
	n = stations.index(sname)
	for tag, col, lab in (('ref', C_REF, 'REF'), ('ctrl', C_CTRL, 'CTRL')):
		zr = ds[f'zr_{tag}'].isel(window=iw, station=n).values
		zw = ds[f'zw_{tag}'].isel(window=iw, station=n).values
		ph = ds[f'phi1_{tag}'].isel(window=iw, station=n).values
		Ph = ds[f'Phi1_{tag}'].isel(window=iw, station=n).values
		c1 = float(ds[f'c1st_{tag}'].isel(window=iw, station=n).values)
		if not np.isfinite(ph).any():
			continue
		ph = ph/np.nanmax(np.abs(ph))
		Ph = Ph/np.nanmax(np.abs(Ph))
		ax.plot(ph, zr, color=col, lw=2.0,
				label=f'{lab}  $c_1$={c1:.2f}')
		ax.plot(Ph, zw, color=col, lw=1.0, ls=':', alpha=0.8)

	# N2 in the background, to place the halocline
	n2 = ds['N2_ref'].isel(window=iw, station=n).values
	zw = ds['zw_ref'].isel(window=iw, station=n).values
	if np.isfinite(n2).any():
		axt = ax.twiny()
		axt.fill_betweenx(zw, 0, n2*1e3, color='0.6', alpha=0.28, lw=0)
		axt.set_xlim(0, np.nanmax(n2*1e3)*1.6)
		axt.tick_params(labelsize=7, colors='0.4')
		if m == 2:
			axt.set_xlabel(r'$N^2$ ($10^{-3}$ s$^{-2}$)', fontsize=8,
						   color='0.4')
		else:
			axt.set_xticklabels([])

	ax.axvline(0, color='k', lw=0.7)
	hst = float(ds.station_h.values[n])
	ax.set_ylim(-hst, 0)
	ax.set_xlim(-1.15, 1.15)
	ax.text(0.04, 0.4, f'{sname}, {hst:.0f} m', transform=ax.transAxes,
			fontsize=9, fontweight='bold', va='top', ha='left', zorder=25,
			bbox=dict(fc='w', ec='0.55', boxstyle='round,pad=0.25',
					  alpha=0.88, lw=0.6))
	ax.grid(alpha=0.22, lw=0.4)
	if m == 0:
		ax.set_ylabel('depth (m)')
		ax.text(0.05, 0.03, 'e', transform=ax.transAxes, fontsize=15,
				fontweight='bold', va='bottom', ha='left')
	ax.set_xlabel(r'$\varphi_1$ (solid), $\Phi_1$ (dotted)', fontsize=8)
	ax.legend(loc='lower right', fontsize=7, framealpha=0.9)



# =====================================================================
# (f1) DEFORMATION RADIUS AGAINST LOCAL SHELF WIDTH
# =====================================================================
W = ds.shelf_width.values
shelf = (h > 20) & (h < SHELF_MAX) & (mask > 0) & np.isfinite(W)

mk = {'rising': 'o', 'peak': 'X', 'decreasing': '^'}
mc = {'rising': '#3b82c4', 'peak': '#c0392b', 'decreasing': '#27ae60'}

for wname in wins:
	k = wins.index(wname)
	R = ds.Reff_ref.isel(window=k).values
	sel = shelf & np.isfinite(R)
	if sel.sum() < 10:
		continue
	wb = np.linspace(np.nanpercentile(W[sel], 2),
					 np.nanpercentile(W[sel], 98), 18)
	ctr, med, lo, hi = [], [], [], []
	for a, b in zip(wb[:-1], wb[1:]):
		s = sel & (W >= a) & (W < b)
		if s.sum() > 30:
			ctr.append(0.5*(a + b))
			med.append(np.nanmedian(R[s]))
			lo.append(np.nanpercentile(R[s], 25))
			hi.append(np.nanpercentile(R[s], 75))
	if not ctr:
		continue
	axf1.fill_between(ctr, lo, hi, color=mc[wname], alpha=0.16, lw=0)
	axf1.plot(ctr, med, color=mc[wname], lw=2.0, marker=mk[wname], ms=6,
			  label=f'REF, {wname}')

R = ds.Reff_ctrl.isel(window=iw).values
sel = shelf & np.isfinite(R)
wb = np.linspace(np.nanpercentile(W[sel], 2), np.nanpercentile(W[sel], 98), 18)
ctr, med = [], []
for a, b in zip(wb[:-1], wb[1:]):
	s = sel & (W >= a) & (W < b)
	if s.sum() > 30:
		ctr.append(0.5*(a + b)); med.append(np.nanmedian(R[s]))
axf1.plot(ctr, med, color='0.25', lw=2.0, ls='--', label=f'CTRL, {WIN}')

lims = [0, max(np.nanpercentile(W[shelf], 99), 300)]
axf1.plot(lims, lims, color='k', lw=1.1, ls=':')
axf1.text(lims[1]*0.62, lims[1]*0.66, r'$R_{\mathrm{eff}} = W$',
		  rotation=38, fontsize=8, color='0.3')
axf1.fill_between(lims, lims, [lims[1]*4]*2, color='#cfe8cf', alpha=0.35, lw=0)
axf1.text(lims[1]*0.06, lims[1]*1.35,
		  'shelf narrower than a deformation radius,\n'
		  'rotation does not control cross shelf propagation',
		  fontsize=8, color='0.25', va='center')

axf1.set_xlim(lims)
axf1.set_ylim(0, lims[1]*1.6)
axf1.set_xlabel('local shelf width, coast to the 200 m isobath (km)')
axf1.set_ylabel(r'$R_{\mathrm{eff}} = \min(c_1/|f|,\ \sqrt{c_1/\beta})$ (km)')
axf1.legend(loc='lower right', ncol=2, framealpha=0.9)
axf1.grid(alpha=0.22, lw=0.4)
axf1.text(0.015, 0.965, 'f', transform=axf1.transAxes, fontsize=15,
		  fontweight='bold', va='top', ha='left')


# =====================================================================
# (f2) SURFACE TRAPPING INDEX AGAINST PLUME THICKNESS
# =====================================================================
for wname in wins:
	k = wins.index(wname)
	hp = ds.hp_ref.isel(window=k).values
	T = ds.trap_ref.isel(window=k).values
	sel = shelf & np.isfinite(hp) & np.isfinite(T)
	if sel.sum() < 30:
		continue
	hb = np.linspace(0, np.nanpercentile(hp[sel], 97), 16)
	ctr, med, lo, hi = [], [], [], []
	for a, b in zip(hb[:-1], hb[1:]):
		s = sel & (hp >= a) & (hp < b)
		if s.sum() > 30:
			ctr.append(0.5*(a + b))
			med.append(np.nanmedian(T[s]))
			lo.append(np.nanpercentile(T[s], 25))
			hi.append(np.nanpercentile(T[s], 75))
	if not ctr:
		continue
	axf2.fill_between(ctr, lo, hi, color=mc[wname], alpha=0.16, lw=0)
	axf2.plot(ctr, med, color=mc[wname], lw=2.0, marker=mk[wname], ms=6,
			  label=f'REF, {wname}')

hp = ds.hp_ctrl.isel(window=iw).values
T = ds.trap_ctrl.isel(window=iw).values
sel = shelf & np.isfinite(hp) & np.isfinite(T)
hb = np.linspace(0, np.nanpercentile(hp[sel], 97), 16)
ctr, med = [], []
for a, b in zip(hb[:-1], hb[1:]):
	s = sel & (hp >= a) & (hp < b)
	if s.sum() > 30:
		ctr.append(0.5*(a + b)); med.append(np.nanmedian(T[s]))
axf2.plot(ctr, med, color='0.25', lw=2.0, ls='--', label=f'CTRL, {WIN}')

axf2.axhline(0.5, color='k', lw=0.9, ls=':')
axf2.text(0.015, 0.965, 'g', transform=axf2.transAxes, fontsize=15,
		  fontweight='bold', va='top', ha='left')
#axf2.text(axf2.get_xlim()[1]*0.02, 0.52,
		  #'half the mode one kinetic energy above the plume base',
		  #fontsize=8, color='0.3', va='bottom')
axf2.set_xlabel('plume thickness, depth of maximum $N^2$ (m)')
axf2.set_ylabel('surface trapping index')
axf2.set_ylim(0, 1)
axf2.legend(loc=1, ncol=2, framealpha=0.9)
axf2.grid(alpha=0.22, lw=0.4)


# =====================================================================
fig.savefig(OUTFIG, dpi=DPI, bbox_inches='tight')

print('wrote', OUTFIG)