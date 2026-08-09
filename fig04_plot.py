#################################
"""
fig04_plot.py
=================================================================
Figure 4  -  The shelf break gate and the frontal scatterer

Flat script. Every panel has its own axes handle:

	axa1 axa2 axa3  gs[0,0] [0,1] [0,2]   modal flux across the sections
	axc             gs[0,3]               onshore fraction against Gamma
	axd             gs[1,0:2]             scattering rate map with rays
	axb             gs[1,2:4]             break crossing modal power in time

Reads fig04_data.nc written by fig04_compute.py.

--- summary, break crossing modal power in MW ---
	 rising   ref  onshore M1   41.6 M2    9.7 M3    1.6   offshore M1  487.9 M2   14.2 M3   22.7   onshore fraction 0.092
	 rising  ctrl  onshore M1   70.2 M2    2.0 M3    0.2   offshore M1  444.7 M2   35.4 M3    8.1   onshore fraction 0.129
	   peak   ref  onshore M1    9.1 M2   10.1 M3    2.0   offshore M1  801.5 M2   30.8 M3   18.4   onshore fraction 0.024
	   peak  ctrl  onshore M1   13.3 M2    1.1 M3    0.5   offshore M1  609.8 M2   34.7 M3   13.1   onshore fraction 0.022
 decreasing   ref  onshore M1   31.2 M2   34.1 M3    3.3   offshore M1  472.6 M2   47.8 M3   12.8   onshore fraction 0.114
 decreasing  ctrl  onshore M1   55.0 M2    1.6 M3    0.4   offshore M1  317.3 M2   21.7 M3    8.2   onshore fraction 0.141

	 rising  mode 1 onshore fraction  ref 0.079   ctrl 0.136     median Gamma  ref 0.488   ctrl 0.805   
	   peak  mode 1 onshore fraction  ref 0.011   ctrl 0.021     median Gamma  ref 0.322   ctrl 0.737   
 decreasing  mode 1 onshore fraction  ref 0.062   ctrl 0.148     median Gamma  ref 0.397   ctrl 0.801   

	 rising  higher mode share of the break flux  ref   8.3%   ctrl   8.1%   
	   peak  higher mode share of the break flux  ref   7.0%   ctrl   7.3%   
 decreasing  higher mode share of the break flux  ref  16.3%   ctrl   7.9%   
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
 
try:
	from scipy.ndimage import uniform_filter
	HAS_SCIPY = True
except ImportError:
	HAS_SCIPY = False
 
plt.rcParams.update({
	'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
	'legend.fontsize': 8, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
	'axes.linewidth': 0.9,
})
 
# ---------------------------------------------------------------------
DATA   = 'fig04_data.nc'
OUTFIG = 'Fig4.jpeg'
DPI    = 300
 
C_REF  = '#0b3d91'
C_CTRL = '#c1440e'
MODE_C = ['#1b3b6f', '#4e9f3d', '#d1495b']      # modes 1, 2, 3
TICK_FS = 7
 
SHOW_TIMESERIES = True   # False gives a strict six panel figure
 
# The scattering rate is the gradient of a ratio, so it is noisy at the
# grid scale. A fixed limit saturated the whole map, so the range is taken
# from a robust percentile and the field is lightly smoothed.
SCAT_SMOOTH = 3          # grid cells, 0 to disable
SCAT_PCTL = 97.0         # percentile setting the colour range
C1_MAX = 2.0             # m s-1, top of the c1 scale in panel f
RAY_STRIDE = 2           # draw every Nth ray, to thin the fan
 
ds = xr.open_dataset(DATA)
lon = ds.lon_rho.values
lat = ds.lat_rho.values
h = np.where(ds.mask_rho.values > 0, ds.h.values, np.nan)
mask = ds.mask_rho.values
wins = [str(w) for w in ds.window.values]
WIN = ds.attrs.get('plot_window', 'peak')
iw = wins.index(WIN)
NM = int(ds.attrs.get('nmode', 3))
BREAK = float(ds.attrs.get('break_isobath', 200.))
 
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
	gl.top_labels = False; gl.right_labels = False
	gl.left_labels = labels_left
	gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': TICK_FS}; gl.ylabel_style = {'size': TICK_FS}
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
nrow = 2 #3 if SHOW_TIMESERIES else 2
hr = [1, 1]#[0.80, 1.15, 0.55] if SHOW_TIMESERIES else [0.85, 1.15]
#fig = plt.figure(figsize=(18.0, 14.5 if SHOW_TIMESERIES else 11.0))
fig = plt.figure(figsize=(18.0, 11))# if SHOW_TIMESERIES else 11.0))

gs = gridspec.GridSpec(nrow, 4, figure=fig, height_ratios=hr,
					   hspace=0.15, wspace=0.32,
					   left=0.05, right=0.975, top=0.955, bottom=0.055)
 
axa = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[0, 1])
axc = fig.add_subplot(gs[0, 2])
axd = fig.add_subplot(gs[0, 3])
axe = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
axf = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
axg = fig.add_subplot(gs[1, 2:]) #if SHOW_TIMESERIES else None
 
 
# =====================================================================
# (a) (b) (c)  MODAL FLUX ALONG THE CROSS SHELF SECTIONS
# =====================================================================
secs = [str(s) for s in ds.section.values]
for m, (ax, sname, tag) in enumerate(zip((axa, axb, axc), secs, 'abc')):
	d = ds.sec_dist.isel(section=m).values
	hh = ds.sec_h.isel(section=m).values
	for n in range(NM):
		ax.plot(d, ds.secF_ref.isel(section=m, mode=n).values/1e3,
				color=MODE_C[n], lw=2.0,
				label=f'mode {n+1}' if m == 0 else None)
		ax.plot(d, ds.secF_ctrl.isel(section=m, mode=n).values/1e3,
				color=MODE_C[n], lw=1.2, ls='--', alpha=0.85)
 
	cross = np.where(np.diff(np.sign(hh - BREAK)) != 0)[0]
	for x in d[cross]:
		ax.axvline(x, color='k', lw=0.9, ls=':')
	if len(cross):
		ax.text(d[cross[0]], 0.97, f' {BREAK:.0f} m',
				transform=ax.get_xaxis_transform(), fontsize=7,
				rotation=90, va='top', color='0.3')
 
	axt = ax.twinx()
	axt.plot(d, hh, color='0.55', lw=0.9, alpha=0.7)
	axt.set_yscale('log'); axt.invert_yaxis()
	axt.tick_params(labelsize=6.5, colors='0.5')
	if m == 2:
		axt.set_ylabel('depth (m)', fontsize=8, color='0.5')
	else:
		axt.set_yticklabels([])
 
	ax.set_xlabel('distance from the coast (km)')
	ax.set_title(sname, fontsize=10)
	ax.grid(alpha=0.22, lw=0.4)
	panel_letter(ax, tag)
	if m == 0:
		ax.set_ylabel(r'$|\mathbf{F}_n|$ (kW m$^{-1}$)')
		ax.plot([], [], color='0.3', lw=2.0, label='REF')
		ax.plot([], [], color='0.3', lw=1.2, ls='--', label='CTRL')
		ax.legend(loc='upper right', fontsize=7.5, framealpha=0.9, ncol=2)
 
axb.set_title(f'{secs[1]}   ({WIN} discharge)', fontsize=10)
 
 
# =====================================================================
# (d)  ONSHORE FRACTION AGAINST THE IMPEDANCE CONTRAST
# =====================================================================
hp = ds.hp_break.isel(window=iw).values
norm_hp = mcolors.Normalize(vmin=0, vmax=np.nanpercentile(hp, 95)
							if np.isfinite(hp).any() else 20)
sc = None
for tag, mk, lab in (('ctrl', 's', 'CTRL'), ('ref', 'o', 'REF')):
	G = ds[f'gamma_{tag}'].isel(window=iw).values
	P = ds[f'phi_on_{tag}'].isel(window=iw).values
	ok = np.isfinite(G) & np.isfinite(P)
	if tag == 'ref' and np.isfinite(hp).any():
		sc = axd.scatter(G[ok], P[ok], c=hp[ok], cmap=plt.cm.gnuplot_r,
						 norm=norm_hp, s=14, marker=mk, lw=0.2,
						 edgecolor='k', label=lab, zorder=4)
	else:
		axd.scatter(G[ok], P[ok], c='0.6', s=11, marker=mk, lw=0.2,
					edgecolor='k', alpha=0.7, label=lab, zorder=3)
	if ok.sum() > 30:
		bins = np.linspace(np.nanpercentile(G[ok], 2),
						   np.nanpercentile(G[ok], 98), 12)
		ctr, med = [], []
		for a_, b_ in zip(bins[:-1], bins[1:]):
			s_ = ok & (G >= a_) & (G < b_)
			if s_.sum() > 8:
				ctr.append(0.5*(a_ + b_)); med.append(np.nanmedian(P[s_]))
		axd.plot(ctr, med, color=C_REF if tag == 'ref' else C_CTRL,
				 lw=2.2, zorder=6,
				 label=f'{lab} median' if False else None)
 
gg = np.linspace(0, 0.95, 100)
axd.plot(gg, 1 - gg**2, color='k', lw=1.4, ls='--', zorder=5)
axd.text(0.60, 1 - 0.60**2 + 0.04, r'$1-\Gamma^{2}$', fontsize=8,
		 rotation=-40, color='0.2')
axd.set_xlabel(r'$\Gamma = (c_1^{\,\mathrm{deep}}-c_1^{\,\mathrm{shelf}})/'
			   r'(c_1^{\,\mathrm{deep}}+c_1^{\,\mathrm{shelf}})$')
axd.set_ylabel('onshore fraction of the mode 1 flux')
axd.set_ylim(-0.03, 1.05)
axd.grid(alpha=0.22, lw=0.4)
axd.legend(loc='center left', fontsize=7.5, framealpha=0.9)
panel_letter(axd, 'd')
if sc is not None:
	cax = inset_axes(axd, width='35%', height='3.2%', loc=3,
					 bbox_to_anchor=(0.051, 0.15, 1, 1),
					 bbox_transform=axd.transAxes, borderpad=0)

	cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
	cb.set_label('plume thickness (m)', fontsize=7)
	cb.ax.tick_params(labelsize=6.5)
 
 
# =====================================================================
# (e)  MODE ONE TO HIGHER MODE SCATTERING RATE
# =====================================================================
scat = ds.scat_ref.isel(window=iw).values.copy()
if SCAT_SMOOTH and HAS_SCIPY:
	filled = np.where(np.isfinite(scat), scat, 0.0)
	wgt = uniform_filter(np.isfinite(scat).astype(float), SCAT_SMOOTH)
	scat = np.where(wgt > 0.2,
					uniform_filter(filled, SCAT_SMOOTH)/np.maximum(wgt, 1e-6),
					np.nan)
lim = np.nanpercentile(np.abs(scat[mask > 0]), SCAT_PCTL)
lim = float(lim) if np.isfinite(lim) and lim > 0 else 1e-3
 
pcm = axe.pcolormesh(lon, lat, np.where(mask > 0, scat, np.nan),
					 cmap='PuOr_r',
					 norm=mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
					 shading='auto', zorder=1, rasterized=True)
axe.contour(lon, lat, ds.c_ref.isel(window=iw, mode=0).values,
			levels=[0.5, 0.8], colors='w', linewidths=1.0, zorder=7)
dress(axe)
axe.set_title(f'mode 1 to higher mode scattering, {WIN} discharge',
			  fontsize=10)
panel_letter(axe, 'e', on_map=True)
 
cax_e = inset_axes(axe, width='40%', height='2.8%', loc='lower left',
				   bbox_to_anchor=(0.05, 0.12, 1, 1),
				   bbox_transform=axe.transAxes, borderpad=0)
cb_e = fig.colorbar(pcm, cax=cax_e, orientation='horizontal', extend='both')
cb_e.set_label(r'$\partial_\ell$(higher mode share)  (km$^{-1}$)',
			   fontsize=7.5)
cb_e.ax.tick_params(labelsize=6.5)
cax_e.set_zorder(30)

 
# =====================================================================
# (f)  MODE ONE RAYS THROUGH THE c1 FIELD
# =====================================================================
c1 = ds.c_ref.isel(window=iw, mode=0).values
pcf = axf.pcolormesh(lon, lat, np.where(mask > 0, c1, np.nan),
					 cmap=plt.cm.terrain_r, vmin=0, vmax=C1_MAX,
					 shading='auto', zorder=1, rasterized=True)
axf.contour(lon, lat, c1, levels=[0.5, 0.8], colors='w', linewidths=1.0,
			zorder=7)
 
sites = [str(s) for s in ds.site.values]
nturn = ntot = 0
for r, sname in enumerate(sites):
	P = ds.rays.isel(site=r).values
	tn = ds.ray_turned.isel(site=r).values
	for q in range(0, P.shape[0], RAY_STRIDE):
		col = 'crimson' if tn[q] else '0.12'
		axf.plot(P[q, :, 0], P[q, :, 1], color=col, lw=0.8, alpha=0.8,
				 zorder=11)
		if tn[q]:
			fin = np.isfinite(P[q, :, 0])
			if fin.any():
				k = np.where(fin)[0][-1]
				axf.plot(P[q, k, 0], P[q, k, 1], marker='x', ms=4,
						 color='crimson', zorder=12)
	nturn += int(tn.sum()); ntot += tn.size
	axf.plot(P[0, 0, 0], P[0, 0, 1], marker='*', ms=15, mfc='gold',
			 mec='k', mew=0.9, ls='none', zorder=13)
	axf.text(P[0, 0, 0], P[0, 0, 1] + 0.25, sname, fontsize=9,
			 fontweight='bold', ha='center', va='bottom', zorder=14)
 
dress(axf, labels_left=False)
axf.set_title('mode 1 rays through the $c_1$ field', fontsize=10)
panel_letter(axf, 'f', on_map=True)
 
hl = [plt.Line2D([], [], color='0.12', lw=1.2, label='ray, no turning'),
	  plt.Line2D([], [], color='crimson', lw=1.2,
				 label=f'ray that turns ({nturn}/{ntot})'),
	  plt.Line2D([], [], color='w', lw=1.2,
				 label=r'$c_1$ = 0.5, 0.8 m s$^{-1}$'),
	  plt.Line2D([], [], marker='*', ms=11, mfc='gold', mec='k', ls='none',
				 label='generation site')]
axf.legend(handles=hl, loc='upper right', framealpha=0.9, fontsize=7.5)
 
cax_f = inset_axes(axf, width='40%', height='2.8%', loc='lower left',
				   bbox_to_anchor=(0.05, 0.12, 1, 1),
				   bbox_transform=axf.transAxes, borderpad=0)
cb_f = fig.colorbar(pcf, cax=cax_f, orientation='horizontal', extend='max')
cb_f.set_label(r'$c_1$ (m s$^{-1}$)', fontsize=7.5)
cb_f.ax.tick_params(labelsize=6.5)
cax_f.set_zorder(30)
 
 
# =====================================================================
# (g)  BREAK CROSSING MODAL POWER IN TIME
# =====================================================================
if SHOW_TIMESERIES:
	tb = ds.block_time.values
	for n in range(NM):
		for tag, ls, lw in (('ref', '-', 2.0), ('ctrl', '--', 1.2)):
			axg.plot(tb, ds[f'Pon_block_{tag}'].isel(mode=n).values,
					 color=MODE_C[n], ls=ls, lw=lw,
					 label=f'mode {n+1} {tag.upper()}')
 
	wcol = {'rising': '#8ecae6', 'peak': '#ffb703', 'decreasing': '#90be6d'}
	for item in ds.attrs.get('windows', '').split(';'):
		if ':' not in item:
			continue
		nm_, v = item.split(':'); t0, t1 = v.split('..')
		axg.axvspan(np.datetime64(t0), np.datetime64(t1),
					color=wcol.get(nm_, '0.7'), alpha=0.16, lw=0, zorder=0)
		axg.text(np.datetime64(t0)
				 + (np.datetime64(t1) - np.datetime64(t0))/2, 0.8, nm_,
				 transform=axg.get_xaxis_transform(), ha='center', va='top',
				 fontsize=8, color='0.25', fontweight='bold')
 
	axg.set_ylabel('onshore power across the break (MW)')
	for l in axg.get_xticklabels():
		l.set_rotation(20); l.set_ha('right')
	axg.legend(loc=1, ncol=3, fontsize=7, framealpha=0.9)
	axg.grid(alpha=0.22, lw=0.4)
	panel_letter(axg, 'g')
 
	axg2 = axg.twinx()
	for tag, ls in (('ref', '-'), ('ctrl', '--')):
		on = ds[f'Pon_block_{tag}'].values
		off = ds[f'Poff_block_{tag}'].values
		tot = np.nansum(on, axis=1) + np.nansum(off, axis=1)
		hi = np.nansum(on[:, 1:], axis=1) + np.nansum(off[:, 1:], axis=1)
		with np.errstate(invalid='ignore', divide='ignore'):
			axg2.plot(tb, np.where(tot > 0, hi/tot, np.nan), color='k',
					  ls=ls, lw=1.4, alpha=0.75,
					  label=f'higher mode share {tag.upper()}')
	axg2.set_ylabel('modes 2 and 3 share of the break flux', color='0.25')
	axg2.tick_params(axis='y', colors='0.25')
	axg2.set_ylim(0, 1)
	axg2.legend(loc=4, fontsize=7, framealpha=0.9)
 
 
# =====================================================================
fig.savefig(OUTFIG, dpi=DPI, bbox_inches='tight')
print('wrote', OUTFIG)
