#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig03_coherence.py
=================================================================
Two checks on the peak window conversion collapse, both from the
EXISTING fig03_data.nc. Nothing is recomputed from the model output.

    python fig03_coherence.py

CHECK A, amplitude or phase
---------------------------
    C = -<u_bt p_bc(-H)> . grad H

and the file already stores the M2 amplitude of p_bc(-H). If that
amplitude holds up during the peak window while C collapses, the loss
cannot be an amplitude loss. Either the barotropic tide vanished, which
it cannot, since both runs share the same tidal forcing, or the phase
between bottom baroclinic pressure and barotropic velocity moved towards
quadrature. The second is the incoherence claim.

The conversion efficiency

    E = C / (0.5 |grad H| A_pbc)          [m s-1]

is proportional to A_ubt cos(dphi). A collapse in E with A_pbc steady
isolates cos(dphi) as the cause.

CHECK B, coherent against block conversion
------------------------------------------
The maps hold the conversion from a harmonic fit over the whole window.
The block series holds the same quantity from independent fits in 50 hour
blocks, area averaged over the 100 to 1000 m band. Their ratio

    gamma = <C_coherent> / <C_block>

is the fraction of the mean conversion that stays phase locked to the
astronomical forcing across the window. A low gamma at peak, with the
block series healthy, is the direct measurement.

Cell areas are reconstructed from lon and lat, since pm and pn were not
stored. That is accurate to well under a per cent on this grid and it
cancels in every ratio reported below.
=================================================================
"""

import numpy as np
import xarray as xr

DATA = 'fig03_data.nc'
PLOT = True
OUTFIG = 'Fig3_coherence_check.jpeg'

ds = xr.open_dataset(DATA)
lon = ds.lon_rho.values
lat = ds.lat_rho.values
h = ds.h.values
mask = ds.mask_rho.values
wins = [str(w) for w in ds.window.values]
BAND = (float(ds.attrs.get('break_band_lo', 100.)),
        float(ds.attrs.get('break_band_hi', 1000.)))
SHELF_MAX = float(ds.attrs.get('shelf_max', 250.))
SLOPE_MAX = float(ds.attrs.get('slope_max', 3500.))


def cell_area(lon, lat):
    """Cell area from the grid geometry, a stand in for 1/(pm pn)."""
    coslat = np.cos(np.deg2rad(lat))
    dxi = np.gradient(lon, axis=1)*111.2e3*coslat
    dyi = np.gradient(lat, axis=1)*111.2e3
    dxe = np.gradient(lon, axis=0)*111.2e3*coslat
    dye = np.gradient(lat, axis=0)*111.2e3
    return np.hypot(dxi, dyi)*np.hypot(dxe, dye)


def grad_h_mag(h, lon, lat):
    coslat = np.cos(np.deg2rad(lat))
    dx = np.gradient(lon, axis=1)*111.2e3*coslat
    dy = np.gradient(lat, axis=0)*111.2e3
    with np.errstate(divide='ignore', invalid='ignore'):
        gx = np.gradient(h, axis=1)/np.where(np.abs(dx) > 1, dx, np.nan)
        gy = np.gradient(h, axis=0)/np.where(np.abs(dy) > 1, dy, np.nan)
    return np.hypot(np.nan_to_num(gx), np.nan_to_num(gy))


A = cell_area(lon, lat)
gmag = grad_h_mag(h, lon, lat)

band = (h >= BAND[0]) & (h <= BAND[1]) & (mask > 0)
shelf = (h < SHELF_MAX) & (mask > 0)
slope = (h >= SHELF_MAX) & (h <= SLOPE_MAX) & (mask > 0)


def wmean(field, m):
    w = np.where(m & np.isfinite(field), A, 0.0)
    tot = w.sum()
    return np.nansum(np.nan_to_num(field)*w)/tot if tot > 0 else np.nan


# =====================================================================
print('=' * 72)
print('CHECK A   is the peak collapse an amplitude loss or a phase loss?')
print('=' * 72)
print(f'{"window":>11} {"run":>5} {"<C> band":>12} {"<A_pbc> band":>14} '
      f'{"E = C/(0.5|gradH|A)":>21}')
print(f'{"":>11} {"":>5} {"W m-2":>12} {"Pa":>14} {"m s-1":>21}')

effA = {}
for tag in ('ref', 'ctrl'):
    for k, w in enumerate(wins):
        C = ds[f'C_{tag}'].isel(window=k).values
        Ap = ds[f'pbc_bot_{tag}'].isel(window=k).values
        with np.errstate(divide='ignore', invalid='ignore'):
            E = C/(0.5*gmag*np.where(Ap > 1e-6, Ap, np.nan))
        cb, ab, eb = wmean(C, band), wmean(Ap, band), wmean(E, band)
        effA[(tag, w)] = (cb, ab, eb)
        print(f'{w:>11} {tag:>5} {cb:12.3e} {ab:14.2f} {eb:21.3e}')

print()
for tag in ('ref', 'ctrl'):
    c0, a0, e0 = effA[(tag, wins[0])]
    c1, a1, e1 = effA[(tag, 'peak')] if 'peak' in wins else (np.nan,)*3
    print(f'  {tag}: peak against rising  '
          f'C x{c1/c0:6.3f}   A_pbc x{a1/a0:6.3f}   E x{e1/e0:6.3f}')
print()
print('  If A_pbc holds near 1 while C and E fall by an order of magnitude,')
print('  the bottom baroclinic pressure is still there and only its phase')
print('  relative to the barotropic tide has moved. That is the incoherence')
print('  reading, and it is not compatible with a real loss of generation.')

# =====================================================================
print()
print('=' * 72)
print('CHECK B   coherent conversion against the block resolved conversion')
print('=' * 72)

tb = ds.block_time.values
wspec = {}
for item in ds.attrs.get('windows', '').split(';'):
    if ':' in item:
        nm, v = item.split(':'); t0, t1 = v.split('..')
        wspec[nm] = (np.datetime64(t0), np.datetime64(t1))

print(f'{"window":>11} {"run":>5} {"coherent":>12} {"block mean":>12} '
      f'{"gamma":>8} {"n blocks":>9}')
print(f'{"":>11} {"":>5} {"W m-2":>12} {"W m-2":>12}')

gam = {}
for tag in ('ref', 'ctrl'):
    Cb = ds[f'C_block_{tag}'].values
    for k, w in enumerate(wins):
        if w not in wspec:
            continue
        t0, t1 = wspec[w]
        sel = (tb >= t0) & (tb <= t1) & np.isfinite(Cb)
        if sel.sum() < 3:
            continue
        coh = wmean(ds[f'C_{tag}'].isel(window=k).values, band)
        blk = float(np.nanmean(Cb[sel]))
        g = coh/blk if blk else np.nan
        gam[(tag, w)] = g
        print(f'{w:>11} {tag:>5} {coh:12.3e} {blk:12.3e} {g:8.3f} '
              f'{sel.sum():9d}')

print()
print('  gamma is the share of the mean conversion that survives a harmonic')
print('  fit over the whole window. Values near one mean a phase locked,')
print('  coherent internal tide. Values far below one mean the phase drifts')
print('  within the window and the coherent fit averages it away.')

# =====================================================================
print()
print('=' * 72)
print('CONTEXT   integrated budget already in the file, for reference')
print('=' * 72)
for reg in ('shelf', 'slope'):
    for k, w in enumerate(wins):
        line = f'  {w:>11} {reg:>6}  '
        for tag in ('ref', 'ctrl'):
            C = float(ds[f'C_{reg}_{tag}'].isel(window=k).values)
            line += f'{tag} C {C:7.1f} MW   '
        print(line)

print()
print('WHAT THIS FILE CANNOT ANSWER')
print('  Refitting the peak window in shorter sub windows needs the per')
print('  block normal equations, which were reduced to scalars before')
print('  saving. Set SAVE_BLOCK_ACCUM = True in fig03_compute.py and the')
print('  next run stores them, about 94 MB per run, after which any window')
print('  length can be refitted here with no further model reads.')

# =====================================================================
if PLOT:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axs[0]
    x = np.arange(len(wins))
    for tag, col in (('ref', '#0b3d91'), ('ctrl', '#c1440e')):
        ax.plot(x, [effA[(tag, w)][1] for w in wins], 'o-', color=col,
                lw=2, label=f'{tag.upper()}')
    ax.set_xticks(x); ax.set_xticklabels(wins)
    ax.set_ylabel(r'$A_{p_{bc}(-H)}$ over the band (Pa)')
    ax.set_title('bottom baroclinic pressure amplitude', fontsize=10)
    ax.grid(alpha=0.25, lw=0.4); ax.legend()

    ax = axs[1]
    for tag, col in (('ref', '#0b3d91'), ('ctrl', '#c1440e')):
        ax.plot(x, [effA[(tag, w)][2] for w in wins], 'o-', color=col,
                lw=2, label=f'{tag.upper()}')
    ax.set_xticks(x); ax.set_xticklabels(wins)
    ax.set_ylabel(r'$E = C\,/\,(0.5|\nabla h| A_{p})$  (m s$^{-1}$)')
    ax.set_title('conversion efficiency, tracks $A_{u_{bt}}\\cos\\Delta\\phi$',
                 fontsize=10)
    ax.grid(alpha=0.25, lw=0.4); ax.legend()

    ax = axs[2]
    for tag, col in (('ref', '#0b3d91'), ('ctrl', '#c1440e')):
        ax.plot(x, [gam.get((tag, w), np.nan) for w in wins], 'o-',
                color=col, lw=2, label=f'{tag.upper()}')
    ax.axhline(1.0, color='k', lw=0.9, ls='--')
    ax.set_xticks(x); ax.set_xticklabels(wins)
    ax.set_ylabel(r'$\gamma$ = coherent / block')
    ax.set_title('coherent fraction of the conversion', fontsize=10)
    ax.grid(alpha=0.25, lw=0.4); ax.legend()

    fig.tight_layout()
    fig.savefig(OUTFIG, dpi=300, bbox_inches='tight')
    print(f'\nwrote {OUTFIG}')