"""
fig06_plot.py
=================================================================
Figure 6  -  Synthesis, a river driven source to sink chain

Flat script. Every panel has its own axes handle and every element is
drawn by its own call, so any of them can be removed or restyled
without touching the rest.

	axa  gs[0]   regime diagram, outer shelf mode one speed against the
				 positive shelf conversion, one point per window, phase
				 and run, coloured by the total domain dissipation
	axb  gs[1]   schematic of the chain, drawn parametrically so that
				 the annotated numbers come from fig07_data.nc

Reads fig07_data.nc written by fig07_compute.py.

The schematic is a first pass. Every element is a named call below, so
it can be tuned here, or the figure can be saved as a vector file and
finished in a drawing program without losing the numbers.
=================================================================
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({
	'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
	'legend.fontsize': 8, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
	'axes.linewidth': 0.9,
})

# ---------------------------------------------------------------------
DATA = 'fig06_data.nc'
OUTFIG = 'Fig6.jpeg'
DPI = 300
SAVE_VECTOR = True                  # also write Fig7.pdf for hand editing

C_REF = '#0b3d91'
C_CTRL = '#c1440e'
C_FRESH = '#7fc7c1'
C_OCEAN = '#cfd8e3'
C_BED = '#b9a88f'
MK = {'spring': 'o', 'neap': 's'}

ds = xr.open_dataset(DATA)
wins = [str(w) for w in ds.window.values]
runs = [str(r) for r in ds.run.values]
nwin = len(wins)

WIN_MK = ['o', 's', 'D']            # rising, peak, decreasing


def panel_letter(ax, s):
	ax.text(0.015, 0.985, s, transform=ax.transAxes, fontsize=15,
			fontweight='bold', va='top', ha='left', zorder=30,
			bbox=dict(fc='w', ec='0.4', boxstyle='round,pad=0.18',
					  alpha=0.9))


# =====================================================================
# FIGURE SKELETON
# =====================================================================
fig = plt.figure(figsize=(15.0, 6.4))
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.25],
					   wspace=0.18, left=0.055, right=0.985,
					   top=0.93, bottom=0.10)

axa = fig.add_subplot(gs[0])
axb = fig.add_subplot(gs[1])


# =====================================================================
# (a)  REGIME DIAGRAM, ONE POINT PER WINDOW AND RUN
# =====================================================================
D = ds.D_domain.values
C = ds.Cpos_window.values
X = ds.c1_outer.values
vmin, vmax = np.nanmin(D), np.nanmax(D)

# arrow from the control to the realistic state, drawn first so the
# markers sit on top of it
for k in range(nwin):
	if not np.all(np.isfinite([X[1, k], X[0, k], C[1, k], C[0, k]])):
		continue
	axa.add_patch(FancyArrowPatch((X[1, k], C[1, k]), (X[0, k], C[0, k]),
								  arrowstyle='-|>', mutation_scale=14,
								  color='0.5', lw=1.2, shrinkA=14,
								  shrinkB=14, zorder=1, alpha=0.85))

for it, tag in enumerate(runs):
	for k in range(nwin):
		if not np.isfinite(X[it, k]) or not np.isfinite(C[it, k]):
			continue
		axa.scatter(X[it, k], C[it, k], s=230, marker=WIN_MK[k % 3],
					c=[float(D[it, k])], cmap='inferno_r',
					vmin=vmin, vmax=vmax,
					edgecolors=(C_REF if tag == 'ref' else C_CTRL),
					linewidths=2.2, zorder=3)
		axa.annotate(f'{wins[k]}, {tag.upper()}', (X[it, k], C[it, k]),
					 textcoords='offset points', xytext=(0, 15),
					 ha='center', fontsize=7.5, color='0.3', zorder=4)

sm = plt.cm.ScalarMappable(cmap='inferno_r',
						   norm=plt.Normalize(vmin=vmin, vmax=vmax))

cb = fig.colorbar(sm, ax=axa, fraction=0.045, pad=0.02)
cb.set_label('total domain dissipation (MW)', fontsize=9)
cb.ax.tick_params(labelsize=8)

axa.set_xlabel(r'outer shelf mode one phase speed $c_1$ (m s$^{-1}$)')
axa.set_ylabel('positive shelf conversion (MW)')
axa.grid(alpha=0.22, lw=0.4)

axa.margins(x=0.14, y=0.16)

handles = [Line2D([], [], ls='none', marker=WIN_MK[k % 3], ms=9, mfc='w',
				  mec='0.3', label=wins[k]) for k in range(nwin)]
handles += [Line2D([], [], ls='none', marker='o', ms=9, mfc='w',
					mec=C_REF, mew=2, label='REF'),
			Line2D([], [], ls='none', marker='o', ms=9, mfc='w',
				   mec=C_CTRL, mew=2, label='CTRL')]
axa.legend(handles=handles, loc='lower right', ncol=2, fontsize=8,
		   framealpha=0.9)
panel_letter(axa, 'a')


# =====================================================================
# (b)  SCHEMATIC OF THE CHAIN
# =====================================================================
axb.set_xlim(0, 10)
axb.set_ylim(-3.2, 2.4)
axb.axis('off')

# ---- bed and water column
bed = Polygon([(0, -0.15), (2.6, -0.35), (4.3, -0.75), (5.0, -2.2),
			   (5.6, -2.9), (10, -3.1), (10, -3.2), (0, -3.2)],
			  closed=True, facecolor=C_BED, edgecolor='0.4', lw=0.8,
			  zorder=2)
axb.add_patch(bed)
axb.add_patch(Rectangle((0, -3.2), 10, 5.0, facecolor=C_OCEAN,
						edgecolor='none', zorder=1))

# ---- plume lens, thick at the coast and thinning seaward
lens = Polygon([(0, 1.85), (2.2, 1.76), (3.6, 1.64), (4.4, 1.55),
				(4.75, 1.50), (4.75, 1.85)], closed=True, facecolor=C_FRESH,
			   edgecolor='0.35', lw=0.9, alpha=0.85, zorder=3)
axb.add_patch(lens)
axb.text(1.1, 1.62, 'river built waveguide', fontsize=9, style='italic',
		 color='0.15', zorder=6)
axb.text(1.1, 1.30, r'$c_1$ doubled, mode one surface trapped',
		 fontsize=8, color='0.25', zorder=6)

# ---- river inflow
axb.add_patch(FancyArrowPatch((-0.05, 1.62), (1.0, 1.62),
							  arrowstyle='-|>', mutation_scale=18,
							  color='#2166ac', lw=3.0, zorder=5))
axb.text(0.62, 2.06, 'Amazon discharge', fontsize=9, color='#2166ac',
		 fontweight='bold', zorder=6)

# ---- generation on the shelf, arrow width scaled by the REF over CTRL
# positive conversion ratio of the decreasing window
rC = float(ds.ratio_Cpos[-1])
for x in (2.4, 3.2, 4.0):
	axb.add_patch(FancyArrowPatch((x, -0.45), (x, 1.15),
								  arrowstyle='-|>', mutation_scale=13,
								  color='#b2182b', lw=1.0 + 1.4*rC,
								  zorder=5))
axb.text(0.1, -0.95, 'generation sustained on the shelf', fontsize=9,
		 color='#b2182b', zorder=6)

# ---- seaward radiation, width scaled by the break crossing flux ratio
rF = float(ds.ratio_Fbreak[-1])
axb.add_patch(FancyArrowPatch((4.6, 0.75), (8.4, 0.05),
							  arrowstyle='-|>', mutation_scale=22,
							  color='#1b3b6f', lw=1.5 + 3.2*rF, zorder=5))
axb.text(5.6, 0.95, 'seaward radiation as mode one', fontsize=9,
		 color='#1b3b6f', zorder=6)


# ---- onshore return at the front, thin and turning
axb.add_patch(FancyArrowPatch((4.6, 0.45), (3.1, 0.55),
							  arrowstyle='-|>', mutation_scale=11,
							  connectionstyle='arc3,rad=0.35',
							  color='#4e9f3d', lw=1.4, zorder=5))

axb.plot([4.55, 4.55], [1.45, -0.85], color='0.35', lw=1.0, ls='--',
		 zorder=4)
axb.text(4.62, -2.35, 'plume front', fontsize=7.5, rotation=90,
		 color='0.35', zorder=6)

# ---- deep absorption
axb.add_patch(FancyArrowPatch((8.6, -0.15), (8.6, -2.35),
							  arrowstyle='-|>', mutation_scale=16,
							  color='#5c3d99', lw=2.6, zorder=5))
axb.text(5.5, -2.62, f'deep zone absorbs '
		 f'{float(ds.deep_import_GW.min()):.1f} to '
		 f'{float(ds.deep_import_GW.max()):.1f} GW and dissipates it',
		 fontsize=8.5, color='#5c3d99', zorder=6)

# ---- the total the chain carries


# ---- isobath ticks for orientation
for x, lab in ((2.6, '50 m'), (4.3, '200 m'), (5.6, '1000 m')):
	axb.plot([x, x], [-3.05, -2.85], color='0.35', lw=1.0, zorder=6)
	axb.text(x, -3.1, lab, fontsize=7, ha='center', va='top', color='0.35',
			 zorder=6)

panel_letter(axb, 'b')


# =====================================================================
fig.savefig(OUTFIG, dpi=DPI, bbox_inches='tight')
print('wrote', OUTFIG)
