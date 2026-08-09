#############################
"""
fig01_plot.py
=================================================================
Figure 1  -  Forcing and the equatorial internal tide regime

Flat script. Every panel has its own axes handle so each one can be
tuned independently:

	axa  gs[0:2, 0:2]   map, bathymetry, sites, moorings, sections
	axb  gs[0, 2]       points per mode one wavelength against depth
	axc  gs[1, 2]       discharge, windows, spring neap envelope
	axd  gs[2, 0:2]     criticality along the shelf break
	axe  gs[2, 2]       rotary spectra

Reads fig01_data.nc written by fig01_compute.py.
=================================================================
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import cartopy
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
DATA    = 'fig01_data.nc'
OUTFIG  = 'Fig1.jpeg'
DPI     = 300
WIN     = 'peak'            # window shown on the map and in panel (d)

C_REF   = '#0b3d91'         # realistic discharge
C_CTRL  = '#c1440e'         # constant discharge
C_GREY  = '0.35'

ds = xr.open_dataset(DATA)

lon  = ds.lon_rho.values
lat  = ds.lat_rho.values
h    = np.where(ds.mask_rho.values > 0, ds.h.values, np.nan)
mask = ds.mask_rho.values

wins = [str(w) for w in ds.window.values]
iw   = wins.index(WIN)

# generation sites, taken from the snapped positions in the data file
gen_sites, gen_inside = {}, {}
if 'site' in ds.dims:
	for n, s in enumerate(ds.site.values):
		k = str(s)
		gen_sites[k] = (float(ds.site_lon.values[n]),
						float(ds.site_lat.values[n]))
		gen_inside[k] = bool(ds.site_inside.values[n])
MAIN_SITES = [s for s in ds.attrs.get('gen_sites_main', 'A,B').split(',') if s]

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

windows = {}
for item in ds.attrs.get('windows', '').split(';'):
	if ':' in item:
		k, v = item.split(':'); t0, t1 = v.split('..')
		windows[k] = (np.datetime64(t0), np.datetime64(t1))

PPW_LO = float(ds.attrs.get('ppw_lo', 8.))
PPW_HI = float(ds.attrs.get('ppw_hi', 12.))
M2_CPD = float(ds.attrs.get('omega_M2_cpd', 1.9323))
M4_CPD = float(ds.attrs.get('omega_M4_cpd', 3.8646))

land   = cfeature.NaturalEarthFeature('physical', 'land', scale='10m')
states = cfeature.NaturalEarthFeature(category='cultural',
									  name='admin_1_states_provinces_lines',
									  scale='10m')

# =====================================================================
# FIGURE SKELETON
# =====================================================================
fig = plt.figure(figsize=(15.5, 11.5))
gs = gridspec.GridSpec(3, 3, figure=fig,
					   width_ratios=[1.0, 1.0, 0.95],
					   height_ratios=[1.0, 1.0, 0.85],
					   hspace=0.30, wspace=0.28,
					   left=0.055, right=0.975, top=0.955, bottom=0.065)

axa = fig.add_subplot(gs[0:2, 0:2], projection=ccrs.PlateCarree())
axb = fig.add_subplot(gs[0, 2])
axd = fig.add_subplot(gs[2, :])
axe = fig.add_subplot(gs[1,2])


# =====================================================================
# (a) MAP
# =====================================================================
axa.set_aspect('auto', adjustable='box')

pc = axa.pcolormesh(lon, lat, h, cmap=plt.cm.ocean,
					norm=mcolors.LogNorm(vmin=10, vmax=4000),
					shading='auto', zorder=1, rasterized=True)

cs_thin = axa.contour(lon, lat, h, levels=[1000, 2000],
					  colors='0.55', linewidths=0.6, linestyles='-', zorder=4)
cs_main = axa.contour(lon, lat, h, levels=[50, 100, 200],
					  colors='k', linewidths=[0.6, 0.6, 1.3],
					  linestyles=['-', '-', '-'], zorder=5)
axa.clabel(cs_main, fmt='%d', fontsize=7, inline=True)

# domain outline
axa.plot(lon[0, :],  lat[0, :],  color='darkblue', lw=1.4, alpha=0.8, zorder=6)
axa.plot(lon[-1, :], lat[-1, :], color='darkblue', lw=1.4, alpha=0.8, zorder=6)
axa.plot(lon[:, 0],  lat[:, 0],  color='darkblue', lw=1.4, alpha=0.8, zorder=6)
axa.plot(lon[:, -1], lat[:, -1], color='darkblue', lw=1.4, alpha=0.8, zorder=6)

# shelf break path used in panel (d)
axa.plot(ds.break_lon, ds.break_lat, color='w', lw=2.6, zorder=7)
axa.plot(ds.break_lon, ds.break_lat, color='k', lw=1.1, zorder=8)

# cross shelf sections
for name, (lo0, la0, lo1, la1) in sections.items():
	axa.plot([lo0, lo1], [la0, la1], color='brown', lw=2.0,
			 ls='--', zorder=9)
	axa.text(lo1, la1, f' {name}', color='brown', fontweight='bold',
			 fontsize=10, va='center', zorder=16)

# generation sites A to F, A and B are the two dominant ones
for name, (glo, gla) in gen_sites.items():
	inside = gen_inside.get(name, True)
	main = name in MAIN_SITES
	axa.plot(glo, gla, marker='*',
			 ms=20 if main else 14,
			 mfc='crimson' if main else 'gold',
			 mec='k', mew=1.1 if main else 0.7,
			 alpha=1.0 if inside else 0.40,
			 ls='none', zorder=15 if main else 14)
	axa.text(glo, gla + 0.26, name + ('*' if main else ''),
			 color='k' if inside else '0.45',
			 fontweight='bold', fontsize=11 if main else 10,
			 ha='center', va='bottom', zorder=16)
	if not inside:
		axa.text(glo, gla - 0.26, 'outside\ndomain', color='0.45',
				 fontsize=6.5, ha='center', va='top', zorder=16)

# AMASSEDS moorings
for name, (mlo, mla) in moorings.items():
	axa.plot(mlo, mla, marker='o', ms=7, mfc='crimson', mec='k', mew=0.8,
			 ls='none', zorder=15)
	axa.text(mlo - 0.15, mla, name, color='crimson', fontweight='bold',
			 fontsize=10, ha='right', va='center', zorder=16)

# virtual stations
vs = [i for i, s in enumerate(ds.station.values) if str(s).startswith('V')]
axa.plot(ds.station_lon.values[vs], ds.station_lat.values[vs],
		 marker='s', ms=6, mfc='w', mec='k', mew=1.0, ls='none', zorder=15)
for i in vs:
	axa.text(ds.station_lon.values[i], ds.station_lat.values[i] - 0.20,
			 str(ds.station.values[i]).replace('V', ''),
			 fontsize=7, ha='center', va='top', zorder=16)

axa.add_feature(land,   facecolor='wheat', edgecolor='grey', lw=0.5, zorder=10)
axa.add_feature(states, facecolor='none',  edgecolor='grey', lw=0.4, zorder=11)
axa.coastlines(resolution='10m', linewidths=0.5, zorder=12)

axa.set_extent([lon.min(), lon.max(), lat.min(), lat.max()],
			   crs=ccrs.PlateCarree())
gl = axa.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.25,
				   lw=0.5, color='grey', ls='--', zorder=13)
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

cax = inset_axes(axa, width='34%', height='2.6%', loc='lower left',
				 bbox_to_anchor=(0.04, 0.06, 1, 1),
				 bbox_transform=axa.transAxes, borderpad=0)
cb = fig.colorbar(pc, cax=cax, orientation='horizontal', extend='both')
cb.set_label('depth (m)', fontsize=8)
cb.ax.tick_params(labelsize=7)

# legend, built by hand so it stays readable over the map
hl = [plt.Line2D([], [], marker='*', ms=15, mfc='crimson', mec='k', ls='none',
				 label='main IT generation sites A*, B*'),
	  plt.Line2D([], [], marker='*', ms=11, mfc='gold', mec='k', ls='none',
				 label='secondary sites C to F'),
	  plt.Line2D([], [], marker='o', ms=6, mfc='crimson', mec='k', ls='none',
				 label='AMASSEDS moorings'),
	  plt.Line2D([], [], marker='s', ms=6, mfc='w', mec='k', ls='none',
				 label='virtual stations'),
	  plt.Line2D([], [], color='k', lw=1.3, label='200 m, shelf break path'),
	  plt.Line2D([], [], color='brown', lw=1.8, ls='--',
				 label='cross shelf sections')]
axa.legend(handles=hl, loc='upper right', framealpha=0.9, fontsize=8)

axa.text(0.015, 0.975, '(a)', transform=axa.transAxes, fontsize=17,
		 fontweight='bold', va='top', ha='left', zorder=30)


# =====================================================================
# (b) POINTS PER MODE ONE WAVELENGTH
# =====================================================================
ok = (mask > 0) & np.isfinite(h)
hh = h[ok]
bins = np.logspace(np.log10(8), np.log10(3000), 26)
ctr = np.sqrt(bins[1:]*bins[:-1])

for tag, col, lab in ((f'ppw_ref', C_REF, 'REF, variable discharge'),
					  (f'ppw_ctrl', C_CTRL, 'CTRL, constant discharge')):
	p = ds[tag].isel(window=iw).values[ok]
	med = np.full(ctr.size, np.nan)
	q25 = np.full(ctr.size, np.nan)
	q75 = np.full(ctr.size, np.nan)
	for k in range(ctr.size):
		sel = (hh >= bins[k]) & (hh < bins[k + 1]) & np.isfinite(p)
		if sel.sum() > 20:
			med[k] = np.nanmedian(p[sel])
			q25[k] = np.nanpercentile(p[sel], 25)
			q75[k] = np.nanpercentile(p[sel], 75)
	axb.fill_between(ctr, q25, q75, color=col, alpha=0.18, lw=0)
	axb.plot(ctr, med, color=col, lw=2.0, label=lab)

axb.axhspan(PPW_LO, PPW_HI, color='0.6', alpha=0.30, lw=0, zorder=0)
axb.axhline(PPW_LO, color='0.3', lw=0.9, ls='--')
axb.axvline(float(ds.attrs.get('break_isobath', 200.)),
			color='k', lw=0.9, ls=':')
axb.text(float(ds.attrs.get('break_isobath', 200.))*1.08, axb.get_ylim()[1]*0.4,
		 'break', fontsize=7, rotation=90, va='top', color='0.3')
axb.set_xscale('log'); axb.set_yscale('log')
axb.set_xlabel('bottom depth (m)')
axb.set_ylabel(r'points per $\lambda_1$')
axb.legend(loc=4, framealpha=0.9)
axb.grid(alpha=0.25, which='both', lw=0.4)
axb.text(0.02, 0.97, '(b)', transform=axb.transAxes, fontsize=15,
		 fontweight='bold', va='top', ha='left')
axb.set_title(f'{WIN} discharge', fontsize=9, color='0.3', loc='right')


# =====================================================================
# (c) CRITICALITY ALONG THE SHELF BREAK
# =====================================================================
blat = ds.break_lat.values
ar = ds['alpha_ref_break'].isel(window=iw).values
ac = ds['alpha_ctrl_break'].isel(window=iw).values

axd.axhspan(0, 1, color='#cfe8cf', alpha=0.55, lw=0, zorder=0)
axd.axhline(1.0, color='k', lw=1.2, ls='--', zorder=3)
axd.text(blat.min(), 0.7, r'$\alpha=1$, critical', fontsize=8, va='bottom')

axd.plot(blat, ac, color=C_CTRL, lw=1.8, label='CTRL, constant discharge')
axd.plot(blat, ar, color=C_REF, lw=1.8, label='REF, variable discharge')

# the other two windows, thin, to show the seasonal spread
for jw, w in enumerate(wins):
	if jw == iw:
		continue
	axd.plot(blat, ds['alpha_ref_break'].isel(window=jw).values,
			 color=C_REF, lw=0.8, alpha=0.35)

# mark the generation sites on the along break axis
for name, (glo, gla) in gen_sites.items():
	if not gen_inside.get(name, True):
		continue
	if blat.min() <= gla <= blat.max():
		main = name in MAIN_SITES
		axd.axvline(gla, color='crimson' if main else '0.65',
					lw=1.1 if main else 0.7, ls=':' if not main else '-',
					alpha=0.8 if main else 1.0, zorder=1)
		axd.text(gla, 0.2, name + ('*' if main else ''),
				 transform=axd.get_xaxis_transform(),
				 ha='center', va='top', fontsize=9 if main else 8,
				 fontweight='bold' if main else 'normal',
				 color='crimson' if main else '0.4')

axd.set_xlabel(f'latitude along the {float(ds.attrs.get("break_isobath",200)):.0f} m isobath')
axd.set_ylabel(r'criticality $\alpha = |\nabla h| / s$')
axd.set_xlim(blat.min(), blat.max())
axd.legend(loc=2, ncol=2, framealpha=0.9)
axd.grid(alpha=0.25, lw=0.4)
axd.text(0.95, 0.1, '(d)', transform=axd.transAxes, fontsize=15,
		 fontweight='bold', va='top', ha='left')

# inset: the difference, which is the actual point of the panel
axdi = inset_axes(axd, width='20%', height='30%', loc=1, borderpad=1.1)
d_alpha = ar - ac
axdi.axhline(0, color='k', lw=0.8)
axdi.plot(blat, d_alpha, color='0.15', lw=1.2)
axdi.fill_between(blat, 0, d_alpha, color='0.5', alpha=0.4, lw=0)
axdi.set_title(r'$\Delta\alpha$ = REF $-$ CTRL', fontsize=8, pad=2)
axdi.tick_params(labelsize=7)
lim = np.nanmax(np.abs(d_alpha))*1.3 if np.isfinite(d_alpha).any() else 1
axdi.set_ylim(-max(lim, 1e-3), max(lim, 1e-3))
axdi.grid(alpha=0.25, lw=0.3)


# =====================================================================
# (c) ROTARY SPECTRA
# =====================================================================
fr = ds.freq.values
stations = [str(s) for s in ds.station.values]
show = [s for s in ['M1', 'M3', 'V200', 'V1000'] if s in stations]
cols = plt.cm.terrain(np.linspace(0.05, 0.85, max(len(show), 1)))

if not np.isfinite(fr).any():
	axe.text(0.5, 0.5, 'no spectra in the data file\nrun  python '
					   'fig01_compute.py --spectra-only',
			 transform=axe.transAxes, ha='center', va='center',
			 fontsize=9, color='crimson')

# semidiurnal and quarter diurnal bands, shaded before the lines
axe.axvspan(1/13.*24, 1/11.*24, color='0.75', alpha=0.35, lw=0, zorder=0)

for k, s in enumerate(show):
	n = stations.index(s)
	ccw = ds['psd_ccw_surf_ref'].values[n]
	cw  = ds['psd_cw_surf_ref'].values[n]
	if np.isfinite(ccw).any():
		axe.loglog(fr, ccw, color=cols[k], lw=1.5, ls='-',
				   label=f'{s} CCW')
	if np.isfinite(cw).any():
		axe.loglog(fr, cw, color=cols[k], lw=1.2, ls='--', alpha=0.8,
				   label=f'{s} CW')

axe.axvline(M2_CPD, color='k', lw=1.0, ls='-', alpha=0.8)
axe.text(M2_CPD*1.04, 0.97, 'M$_2$', transform=axe.get_xaxis_transform(),
		 fontsize=9, va='top')
axe.axvline(M4_CPD, color='k', lw=0.9, ls='--', alpha=0.7)
axe.text(M4_CPD*1.04, 0.97, 'M$_4$', transform=axe.get_xaxis_transform(),
		 fontsize=9, va='top')

# the inertial band, which at these latitudes sits deep in the subtidal
fst = ds.station_f.values
fcpd = np.abs(fst)*86400/(2*np.pi)
fcpd = fcpd[np.isfinite(fcpd) & (fcpd > 0)]
if fcpd.size:
	axe.axvspan(fcpd.min(), fcpd.max(), color='#7fb3d5', alpha=0.35, lw=0)
	axe.text(np.sqrt(fcpd.min()*fcpd.max()), 0.9,
			 r'$f$', transform=axe.get_xaxis_transform(),
			 ha='center', fontsize=10, color='#1a5276')
	axe.text(np.sqrt(fcpd.min()*fcpd.max()), 0.02,
			 f'$T_f \\approx$ {1/np.mean(fcpd):.0f} d',
			 transform=axe.get_xaxis_transform(),
			 ha='center', fontsize=7, color='#1a5276')

axe.set_xlabel('frequency (cpd)')
axe.set_ylabel(r'rotary PSD (m$^2$ s$^{-2}$ cpd$^{-1}$)')
axe.legend(loc='lower left', ncol=2, framealpha=0.9, fontsize=7)
axe.grid(alpha=0.25, which='both', lw=0.4)
axe.text(0.02, 0.97, '(c)', transform=axe.transAxes, fontsize=15,
		 fontweight='bold', va='top', ha='left')

dt_h = float(ds.attrs.get('dt_hours', np.nan))
#if np.isfinite(dt_h):
	#nyq = 12./dt_h
	#ok = nyq > M4_CPD*1.5
	#axe.text(0.98, 0.02,
			 #f'output {dt_h:.0f} h, Nyquist {nyq:.1f} cpd'
			 #+ ('' if ok else '  M$_4$ ALIASED'),
			 #transform=axe.transAxes, ha='right', va='bottom',
			 #fontsize=7, color='0.35' if ok else 'crimson')
# =====================================================================
fig.savefig(OUTFIG, dpi=DPI, bbox_inches='tight')
print('wrote', OUTFIG)