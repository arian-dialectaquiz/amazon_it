###########################
"""
fig05_compute.py
=================================================================
Figure 5  -  Coherence made at the source, lost in the plume
Amazon shelf internal tide manuscript

Computes and stores everything the panels need:

  (a to c) maps of the coherent fraction of the mode 1 to 3 flux,
           peak window, REF
  (d)      map of the mode 1 coherent fraction difference, REF minus CTRL
  (e)      coherent modal SSH amplitude against fit length at stations
           along the propagation path from site A, both runs
  (f)      coherent fraction against fit length per mode, averaged over
           the shelf band and over the deep band, both runs
  (g)      coherent fraction along the propagation path per mode
  (h, i)   decorrelation timescale per mode against discharge and
           against |d hp / dt|
  (j)      coherent and incoherent M2 baroclinic SSH per mode inside the
           SWOT track 20 box, against Tchilibou et al. (2025)

Output:  fig05_data.nc   (read by fig05_plot.py)

Method
------
Same two pass architecture as fig04. A strided pass gives the window
mean stratification and the vertical modes, normalised so that
(1/H) int phi_n^2 dz = 1. A full streaming pass projects the baroclinic
velocity and pressure onto those modes at each record and accumulates
the M2 normal equations IN BLOCKS of BLOCK_HOURS at every wet column.
Each block is solved the moment its last record has been seen, so the
accumulator never exceeds one block, and what survives the pass is the
complex modal amplitude per block

    Z_n = a_n - i b_n ,  x(t) = Re{ Z_n exp(i omega t) }

for u_n, v_n and p_n, n = 1 to 3.

WHY COHERENCE IS MEASURED THIS WAY
----------------------------------
fig03 established that generation is phase locked, so any loss of
coherence downstream is acquired in propagation and the plume is the
only medium property that differs between the runs. The coherent
fraction over a set of blocks is the vector mean of the block
amplitudes over the mean of their magnitudes,

    gamma_n = |<Z_n>| / <|Z_n|>                         (amplitude)
    gamma_n^F = |<F_n>| / <|F_n|>                       (flux)

with F_n = H/2 Re(Z_u conj(Z_p)) the block modal flux vector and the
coherent flux built from the vector mean amplitudes. Both are bounded
by unity and both reduce to the classical coherent over total amplitude
ratio of Nash et al. (2012). Sliding the fit length L over the block
series gives gamma_n(L), whose 1/e crossing is the decorrelation
timescale reported per mode.

The modal surface expression uses eta_n = p_n phi_n(0) / (rho0 g),
which is the quantity altimetry observes, so panel (j) is directly
comparable with the mode resolved coherent and incoherent structure
reported for this shelf by Tchilibou et al. (2025).
=================================================================
"""

import os
import time as _time
import warnings
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from fig01_compute import (
	REALISTIC, CONTROL, WINDOWS, GEN_SITES, GEN_SITES_MAIN,
	BREAK_ISOBATH, G, RHO0, OMEGA_M2,
	open_run, build_tree, nearest_ji, C1_MIN, eos_provenance,
)
from fig03_compute import (
	uv_to_rho, baroclinic_pressure, design_matrix, solve_normal,
	window_index, density_chunk, window_mean_density,
	EOS_MODE, BLOCK_HOURS, TIME_CHUNK,
)
from fig04_compute import normalised_modes, unit_normal_offshore


# =====================================================================
# CONFIGURATION
# =====================================================================
OUTFILE = 'fig05_data.nc'
FIG04FILE = 'fig04_data.nc'        # supplies the peak window modal flux

NMODE = 3
BLOCK_CONSTITUENTS = {'M2': OMEGA_M2}

# fit lengths, in blocks. With BLOCK_HOURS = 50 this spans 2.1 to 66.7 d,
# which brackets both the discharge pulse timescale and the mesoscale
# timescale reported offshore.
FIT_BLOCKS = np.array([1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32])
GAMMA_CUT = float(np.exp(-1.0))    # 1/e crossing defines the timescale

# propagation path, traced through the peak window mode 1 flux field of
# REF, seeded either side of site A along the bathymetric normal so that
# the shoreward and the seaward legs are both followed downstream
PATH_SITE = 'A'
PATH_SEED = 25.0e3                 # m, seed offset each side of the site
PATH_DS = 4.0e3                    # m, integration step
PATH_NSTEP = 110                   # steps per leg, about 440 km
PATH_FMIN = 1.0                    # W m-1, stop where the flux vanishes
STATION_KM = np.array([-240., -170., -110., -50., 0., 60., 140., 240.])

# analysis bands, defined on bathymetry alone so that the two runs are
# compared over identical columns
SHELF_BAND = (30., 300.)
DEEP_MIN = 1000.
SWOT_BOX = (-45.5, -41.0, -2.5, 6.0)   # lon0, lon1, lat0, lat1
S_REF = 36.0                           # reference salinity for Vfw

# River discharge, m3 s-1. These are the twelve monthly means of the
# 1990 hydrological year that build the forcing, before the sixty to
# forty split between the North and South Channels, so the series below
# is the total delivered to the domain. They are anchored on the dates
# set by DISCHARGE_ANCHOR and interpolated to a daily series, and the
# block means are read off that series. CTRL holds the discharge at
# CTRL_Q, which is the January to July mean of this series to within
# half a per cent, so the two runs share the same period mean forcing.
DISCHARGE_MONTHLY = np.array([175000., 206000., 240000., 240000.,
							  213000., 177000., 143000., 114000.,
							  107000., 116000., 133000., 154000.])
DISCHARGE_YEAR = 1990
# 'legacy' reproduces the anchoring of the forcing script, the month
# ends from the previous December onwards shifted forward by thirty
# days, which lands near the thirtieth of each month rather than mid
# month. 'mid' anchors on the fifteenth instead and exists only as a
# phase sensitivity test, since the block means inherit whichever phase
# is used here.
DISCHARGE_ANCHOR = 'legacy'
CTRL_Q = 2.0e5

RUNS = ('ref', 'ctrl')
PLOT_WINDOW = 'peak'
USE_CACHE = True


# =====================================================================
# HELPERS
# =====================================================================
def block_edges(ntime, dt_h):
	"""Block length in records and the number of blocks."""
	blk = max(int(round(BLOCK_HOURS/dt_h)), 8)
	return blk, int(np.ceil(ntime/blk))


def coherent_stats(Z):
	"""
	Vector mean, mean magnitude and coherent fraction of a complex block
	series. Z has the blocks on the leading axis.
	"""
	zm = np.nanmean(Z, axis=0)
	am = np.nanmean(np.abs(Z), axis=0)
	with np.errstate(invalid='ignore', divide='ignore'):
		g = np.where(am > 0, np.abs(zm)/am, np.nan)
	return zm, am, g


def modal_flux(Zu, Zv, Zp, H):
	"""Block modal flux vector, F_n = H/2 Re(Z_u conj(Z_p))."""
	Fx = 0.5*H*np.real(Zu*np.conj(Zp))
	Fy = 0.5*H*np.real(Zv*np.conj(Zp))
	return Fx, Fy


def sliding_coherence(Z, Lb, min_frac=0.7):
	"""
	Coherent amplitude and coherent fraction for every window of Lb
	consecutive blocks. Uses cumulative sums so the cost does not grow
	with Lb. Z is (nblock, ncol) complex. Returns (npos, ncol) arrays.
	"""
	nb = Z.shape[0]
	if Lb > nb:
		sh = (0,) + Z.shape[1:]
		return np.empty(sh), np.empty(sh)
	ok = np.isfinite(Z)
	zs = np.concatenate([np.zeros((1,) + Z.shape[1:], dtype=Z.dtype),
						 np.cumsum(np.where(ok, Z, 0), axis=0)])
	as_ = np.concatenate([np.zeros((1,) + Z.shape[1:]),
						  np.cumsum(np.where(ok, np.abs(Z), 0.), axis=0)])
	ns = np.concatenate([np.zeros((1,) + Z.shape[1:]),
						 np.cumsum(ok.astype(float), axis=0)])
	sz = zs[Lb:] - zs[:-Lb]
	sa = as_[Lb:] - as_[:-Lb]
	sn = ns[Lb:] - ns[:-Lb]
	good = sn >= max(1.0, min_frac*Lb)
	with np.errstate(invalid='ignore', divide='ignore'):
		acoh = np.where(good, np.abs(sz)/np.maximum(sn, 1.), np.nan)
		amag = np.where(good, sa/np.maximum(sn, 1.), np.nan)
		gam = np.where(amag > 0, acoh/amag, np.nan)
	return acoh, gam


def crossing_length(L_days, gam):
	"""
	Fit length at which the coherent fraction first falls through 1/e,
	by linear interpolation between the two bracketing fit lengths. NaN
	when the curve never crosses within the fit lengths available, which
	is reported as censored rather than as a large timescale.
	"""
	g = np.asarray(gam, dtype=float)
	ok = np.isfinite(g)
	if ok.sum() < 2:
		return np.nan, 1
	L = np.asarray(L_days, dtype=float)[ok]
	g = g[ok]
	below = np.where(g < GAMMA_CUT)[0]
	if below.size == 0:
		return np.nan, 1
	k = below[0]
	if k == 0:
		return L[0], 0
	g0, g1 = g[k - 1], g[k]
	if not np.isfinite(g0) or g0 == g1:
		return L[k], 0
	f = (g0 - GAMMA_CUT)/(g0 - g1)
	return L[k - 1] + f*(L[k] - L[k - 1]), 0


def trace_flux_path(Fx, Fy, lon, lat, mask, x0, y0, ds=PATH_DS,
					nstep=PATH_NSTEP, fmin=PATH_FMIN):
	"""
	Path followed downstream along the modal energy flux, integrated from
	the seed point along the unit vector of (Fx, Fy). The flux components
	are taken in the same frame as the break normal projection of fig04,
	so no rotation by the grid angle is applied here either.
	"""
	F = np.hypot(Fx, Fy)
	ok = np.isfinite(F) & (mask > 0) & (F > fmin)
	if ok.sum() < 10:
		return np.full((nstep, 2), np.nan)
	jj, ii = np.where(ok)
	tree = cKDTree(np.column_stack([lon[ok], lat[ok]]))
	ex = np.where(ok, Fx/np.maximum(F, 1e-12), np.nan)
	ey = np.where(ok, Fy/np.maximum(F, 1e-12), np.nan)

	path = np.full((nstep, 2), np.nan)
	plon, plat = float(x0), float(y0)
	for s in range(nstep):
		d, k = tree.query(np.array([[plon, plat]]))
		if d[0] > 0.6:
			break
		j, i = int(jj[k[0]]), int(ii[k[0]])
		tx, ty = float(ex[j, i]), float(ey[j, i])
		if not (np.isfinite(tx) and np.isfinite(ty)):
			break
		path[s] = (plon, plat)
		plat_new = plat + ty*ds/111.2e3
		plon = plon + (tx*ds/111.2e3)/max(np.cos(np.deg2rad(plat)), 0.1)
		plat = plat_new
	return path


def build_path(fig04, geom):
	"""
	Full propagation path through site A, shoreward leg first with
	negative distance, seaward leg with positive distance, plus the
	stations picked at the requested signed distances.
	"""
	lon, lat, h, mask = geom['lon'], geom['lat'], geom['h'], geom['mask']
	pm, pn = geom['pm'], geom['pn']
	kw = [str(w) for w in fig04.window.values].index(PLOT_WINDOW)
	Fx = fig04.Fx_ref.isel(window=kw, mode=0).values
	Fy = fig04.Fy_ref.isel(window=kw, mode=0).values

	nox, noy = unit_normal_offshore(h, pm, pn, mask)
	tree, jj, ii = build_tree(lon, lat, mask)
	slon, slat = GEN_SITES[PATH_SITE]
	j0, i0 = nearest_ji(tree, jj, ii, np.array([slon]), np.array([slat]))
	j0, i0 = int(j0[0]), int(i0[0])
	nx0, ny0 = float(nox[j0, i0]), float(noy[j0, i0])

	seed_on = (slon - (nx0*PATH_SEED/111.2e3)/np.cos(np.deg2rad(slat)),
			   slat - ny0*PATH_SEED/111.2e3)
	seed_off = (slon + (nx0*PATH_SEED/111.2e3)/np.cos(np.deg2rad(slat)),
				slat + ny0*PATH_SEED/111.2e3)

	leg_on = trace_flux_path(Fx, Fy, lon, lat, mask, *seed_on)
	leg_off = trace_flux_path(Fx, Fy, lon, lat, mask, *seed_off)
	leg_on = leg_on[np.isfinite(leg_on[:, 0])]
	leg_off = leg_off[np.isfinite(leg_off[:, 0])]

	def arclength(p):
		if p.shape[0] < 2:
			return np.zeros(p.shape[0])
		dx = np.diff(p[:, 0])*111.2*np.cos(np.deg2rad(p[:-1, 1]))
		dy = np.diff(p[:, 1])*111.2
		return np.concatenate([[0.], np.cumsum(np.hypot(dx, dy))])

	d_on = -(arclength(leg_on) + PATH_SEED/1e3)
	d_off = arclength(leg_off) + PATH_SEED/1e3
	pts = np.vstack([leg_on[::-1], np.array([[slon, slat]]), leg_off])
	dist = np.concatenate([d_on[::-1], [0.], d_off])

	pj, pi = nearest_ji(tree, jj, ii, pts[:, 0], pts[:, 1])
	st_idx = np.array([int(np.nanargmin(np.abs(dist - dk)))
					   for dk in STATION_KM])
	print(f'  path from site {PATH_SITE}, {pts.shape[0]} points, '
		  f'{dist.min():.0f} to {dist.max():.0f} km')
	return dict(lon=pts[:, 0], lat=pts[:, 1], dist=dist, j=pj, i=pi,
				st_idx=st_idx, st_dist=dist[st_idx])


def discharge_daily():
	"""
	Daily discharge series, built as the forcing builds it. The twelve
	monthly means are placed on their anchor dates and interpolated in
	time between them, with the first and last months held constant
	beyond the end anchors. Returns the daily times and values.
	"""
	import pandas as pd
	if DISCHARGE_ANCHOR == 'legacy':
		try:
			me = pd.date_range(start=f'{DISCHARGE_YEAR - 1}-12-01',
							   periods=12, freq='M')
		except ValueError:
			me = pd.date_range(start=f'{DISCHARGE_YEAR - 1}-12-01',
							   periods=12, freq='ME')
		anchors = me + pd.Timedelta(days=30)
	else:
		anchors = pd.to_datetime(
			[f'{DISCHARGE_YEAR}-{m:02d}-15' for m in range(1, 13)])

	days = pd.date_range(start=f'{DISCHARGE_YEAR}-01-01',
						 end=f'{DISCHARGE_YEAR}-12-31', freq='D')
	s = pd.Series(np.nan, index=days, dtype=float)
	inside = anchors.isin(days)
	if not inside.all():
		warnings.warn(f'{int((~inside).sum())} discharge anchors fall '
					  f'outside {DISCHARGE_YEAR} and are dropped')
	s.loc[anchors[inside]] = DISCHARGE_MONTHLY[inside]
	s = s.interpolate(method='time').ffill().bfill()
	return s.index.values, s.values


def load_discharge(tblk, tag):
	"""
	Discharge at the block centre times, m3 s-1. CTRL is held at CTRL_Q
	by construction, so its points collapse onto a single abscissa in the
	timescale panels, which is the comparison those panels are built on.
	"""
	if tag == 'ctrl':
		return np.full(tblk.size, CTRL_Q)
	td, qd = discharge_daily()
	x = (td - tblk[0])/np.timedelta64(1, 's')
	xq = (tblk - tblk[0])/np.timedelta64(1, 's')
	return np.interp(xq, x, qd)


# =====================================================================
# STREAMING PASS FOR ONE RUN
# =====================================================================
def process_run(ds, tag, wins, geom, path, bands):
	lon, lat, h, mask = geom['lon'], geom['lat'], geom['h'], geom['mask']
	pm, pn, flat = geom['pm'], geom['pn'], geom['flat']
	ny, nx = geom['ny'], geom['nx']
	nz = ds.sizes['s_rho']
	nwet = int(flat.sum())
	nwin = len(wins)

	t = ds.ocean_time.values
	t_s = (t - t[0])/np.timedelta64(1, 's')
	widx = window_index(t, WINDOWS)
	dt_h = float(np.median(np.diff(t_s))/3600.)
	blk, nblock = block_edges(t.size, dt_h)
	print(f'  {tag}: {t.size} records, {dt_h:.2f} h apart, '
		  f'{nblock} blocks of {blk} records')

	lon_f = lon.ravel()[flat]
	lat_f = lat.ravel()[flat]
	area_f = (1.0/(pm*pn)).ravel()[flat]
	h_f = h.ravel()[flat]

	print('  mean state pass and vertical modes')
	Tm, Sm, rho_m, alpha, beta = window_mean_density(ds, wins, widx, flat)

	cn_w = np.full((nwin, NMODE, nwet), np.nan)
	proj_w = np.full((nwin, nz, NMODE, nwet), np.nan, dtype=np.float32)
	phis_w = np.full((nwin, NMODE, nwet), np.nan)
	Hm_w = np.full((nwin, nwet), np.nan)

	for k, wname in enumerate(wins):
		t0, t1 = WINDOWS[wname]
		sub = ds.sel(ocean_time=slice(t0, t1))
		if sub.sizes['ocean_time'] == 0:
			continue
		temp = sub.temp.mean('ocean_time').values
		salt = sub.salt.mean('ocean_time').values
		zr = sub.z_rho.mean('ocean_time').values
		zw = sub.z_w.mean('ocean_time').values
		from fig01_compute import compute_N2
		N2 = compute_N2(temp, salt, zr, zw, lon, lat)
		N2f = N2.reshape(nz + 1, -1)[:, flat]
		zwf = zw.reshape(nz + 1, -1)[:, flat]
		cn, phi, dz, H = normalised_modes(N2f, zwf, NMODE)
		cn_w[k] = cn
		Hm_w[k] = H
		phis_w[k] = phi[-1]          # top rho level, the surface value
		proj_w[k] = (phi*dz[:, None, :]
					 / np.maximum(H, 1e-6)[None, None, :]).astype(np.float32)
		print(f'    {wname}: c1 median {np.nanmedian(cn[0]):.2f} m/s, '
			  f'phi1(0) median {np.nanmedian(phi[-1, 0]):.2f}')

	# ---- block accumulator, one block open at a time
	nbc = 1 + 2*len(BLOCK_CONSTITUENTS)
	nfield = 3*NMODE
	Zu = np.full((nblock, NMODE, nwet), np.nan, dtype=np.complex64)
	Zv = np.full((nblock, NMODE, nwet), np.nan, dtype=np.complex64)
	Zp = np.full((nblock, NMODE, nwet), np.nan, dtype=np.complex64)
	hp_sum = np.zeros((nblock, nwet), dtype=np.float64)
	hp_cnt = np.zeros(nblock)
	vfw_sum = np.zeros(nblock)
	tblk = np.empty(nblock, dtype=t.dtype)
	for g in range(nblock):
		tblk[g] = t[min(int((g + 0.5)*blk), t.size - 1)]

	bufG = np.zeros((nbc, nbc))
	bufY = np.zeros((nbc, nfield, nwet))
	open_g = -1
	shelf_col = (h_f < 250.)

	def flush(g):
		if g < 0 or bufG[0, 0] <= nbc:
			return
		c = solve_normal(bufG, bufY)
		a, b = c[1], c[2]
		Zu[g] = (a[0:NMODE] - 1j*b[0:NMODE]).astype(np.complex64)
		Zv[g] = (a[NMODE:2*NMODE] - 1j*b[NMODE:2*NMODE]).astype(np.complex64)
		Zp[g] = (a[2*NMODE:3*NMODE]
				 - 1j*b[2*NMODE:3*NMODE]).astype(np.complex64)

	chunk = TIME_CHUNK or blk
	tic = _time.time()
	nan_total = 0
	for s in range(0, t.size, chunk):
		e = min(s + chunk, t.size)
		sl = slice(s, e)
		nt_ = e - s
		kk = np.where(widx[sl] >= 0, widx[sl], 0)

		u = ds.u.isel(ocean_time=sl).values
		v = ds.v.isel(ocean_time=sl).values
		ur, vr = uv_to_rho(u, v)
		del u, v
		T = ds.temp.isel(ocean_time=sl).values
		S = ds.salt.isel(ocean_time=sl).values
		zw = ds.z_w.isel(ocean_time=sl).values
		zr = ds.z_rho.isel(ocean_time=sl).values

		ur = ur.reshape(nt_, nz, -1)[:, :, flat]
		vr = vr.reshape(nt_, nz, -1)[:, :, flat]
		T = T.reshape(nt_, nz, -1)[:, :, flat]
		S = S.reshape(nt_, nz, -1)[:, :, flat]
		zr = zr.reshape(nt_, nz, -1)[:, :, flat]
		zw = zw.reshape(nt_, nz + 1, -1)[:, :, flat]

		rho = density_chunk(ds, sl, T, S, zr, lon=lon_f, lat=lat_f,
							Tm=Tm, Sm=Sm, rho_m=rho_m,
							alpha=alpha, beta=beta, k=kk)
		p_bc = baroclinic_pressure(rho - rho_m[kk], zw)

		# plume thickness, depth of the maximum density gradient of the
		# field already carried by the pass. This sits on the same sigma
		# level as the N2 maximum of Figures 1 and 2 inside the plume and
		# avoids a second equation of state call per record.
		drho = np.diff(rho, axis=1)
		dzc = np.diff(zr, axis=1)
		with np.errstate(invalid='ignore', divide='ignore'):
			n2r = -(G/RHO0)*drho/np.where(np.abs(dzc) > 1e-6, dzc, np.nan)
		n2r = np.where(np.isfinite(n2r), n2r, -np.inf)
		kmax = np.argmax(n2r, axis=1)
		zc = 0.5*(zr[:, :-1] + zr[:, 1:])
		hp_t = -np.take_along_axis(zc, kmax[:, None, :], axis=1)[:, 0, :]

		dzr = np.diff(zw, axis=1)
		fw = np.sum(np.clip((S_REF - S)/S_REF, 0., None)*dzr, axis=1)
		vfw_t = np.nansum(fw[:, shelf_col]*area_f[shelf_col], axis=1)

		del T, S, zr, rho, dzc, drho, n2r, zc, fw

		dz = np.diff(zw, axis=1)
		H = np.sum(dz, axis=1)
		u_bt = np.sum(ur*dz, axis=1)/np.maximum(H, 1e-6)
		v_bt = np.sum(vr*dz, axis=1)/np.maximum(H, 1e-6)
		u_bc = ur - u_bt[:, None, :]
		v_bc = vr - v_bt[:, None, :]
		del ur, vr, u_bt, v_bt

		Y = np.empty((nt_, nfield, nwet), dtype=np.float32)
		for k in np.unique(kk):
			m = kk == k
			P = proj_w[k]
			Y[m, 0:NMODE] = np.einsum('tzm,znm->tnm', u_bc[m], P,
									  optimize=True)
			Y[m, NMODE:2*NMODE] = np.einsum('tzm,znm->tnm', v_bc[m], P,
											optimize=True)
			Y[m, 2*NMODE:3*NMODE] = np.einsum('tzm,znm->tnm', p_bc[m], P,
											  optimize=True)
		del u_bc, v_bc, p_bc

		nbad = int(np.count_nonzero(~np.isfinite(Y)))
		if nbad:
			nan_total += nbad
			np.nan_to_num(Y, copy=False)

		Xb = design_matrix(t_s[sl], BLOCK_CONSTITUENTS)
		bid = np.arange(s, e)//blk
		for g in np.unique(bid):
			m = bid == g
			if g != open_g:
				flush(open_g)
				bufG[:] = 0.
				bufY[:] = 0.
				open_g = int(g)
			Xg = Xb[m]
			bufG += Xg.T @ Xg
			bufY += np.einsum('tc,tfm->cfm', Xg, Y[m], optimize=True)
			hp_sum[g] += np.nansum(hp_t[m], axis=0)
			hp_cnt[g] += int(m.sum())
			vfw_sum[g] += float(np.nansum(vfw_t[m]))
		del Y, hp_t, vfw_t

		if (s//chunk) % 10 == 0 or e == t.size:
			done = e/t.size
			el = _time.time() - tic
			print(f'    {100*done:5.1f}%   {el/60:6.1f} min elapsed   '
				  f'{el/max(done, 1e-6) - el:6.0f} s to go', flush=True)
	flush(open_g)

	if nan_total:
		frac = nan_total/(t.size*nfield*nwet)
		print(f'  WARNING {nan_total} non finite projected values '
			  f'({100*frac:.4f}% of samples), zeroed before accumulation')

	hp_blk = hp_sum/np.maximum(hp_cnt, 1)[:, None]
	vfw_blk = vfw_sum/np.maximum(hp_cnt, 1)/1e9        # km3
	del hp_sum, bufY

	return reduce_run(tag, wins, geom, path, bands, Zu, Zv, Zp,
					  cn_w, Hm_w, phis_w, hp_blk, vfw_blk, tblk)


# =====================================================================
# REDUCTION, everything the panels need from the block amplitudes
# =====================================================================
def reduce_run(tag, wins, geom, path, bands, Zu, Zv, Zp,
			   cn_w, Hm_w, phis_w, hp_blk, vfw_blk, tblk):
	print(f'  reducing {tag}')
	flat = geom['flat']
	nwet = int(flat.sum())
	nwin = len(wins)
	nblock = Zu.shape[0]
	wblk = window_index(tblk, WINDOWS)
	L_days = FIT_BLOCKS*BLOCK_HOURS/24.

	pw = geom['path_wet']
	npath = pw.size
	st = path['st_idx']
	nst = st.size
	shelf = bands['shelf']
	deep = bands['deep']
	box = bands['box']

	gammaF = np.full((nwin, NMODE, nwet), np.nan)
	gammaP = np.full((nwin, NMODE, nwet), np.nan)
	eta_coh = np.full((nwin, NMODE, nwet), np.nan)
	eta_inc = np.full((nwin, NMODE, nwet), np.nan)
	gam_path = np.full((nwin, NMODE, npath), np.nan)

	for k in range(nwin):
		m = wblk == k
		if m.sum() < 3:
			continue
		H = Hm_w[k]
		bad = ~np.isfinite(cn_w[k, 0]) | (cn_w[k, 0] < C1_MIN)
		for n in range(NMODE):
			zu, zv, zp = Zu[m, n], Zv[m, n], Zp[m, n]
			zpm, apm, gp = coherent_stats(zp)
			zum, _, _ = coherent_stats(zu)
			zvm, _, _ = coherent_stats(zv)

			Fxb, Fyb = modal_flux(zu, zv, zp, H[None, :])
			Fmag = np.nanmean(np.hypot(Fxb, Fyb), axis=0)
			Fxc = 0.5*H*np.real(zum*np.conj(zpm))
			Fyc = 0.5*H*np.real(zvm*np.conj(zpm))
			with np.errstate(invalid='ignore', divide='ignore'):
				gf = np.where(Fmag > 0, np.hypot(Fxc, Fyc)/Fmag, np.nan)

			ze = zp*(phis_w[k, n]/(RHO0*G))[None, :]
			ec = np.abs(np.nanmean(ze, axis=0))
			er = np.sqrt(np.nanmean(np.abs(ze)**2, axis=0))
			ei = np.sqrt(np.clip(er**2 - ec**2, 0., None))

			gammaP[k, n] = np.where(bad, np.nan, gp)
			gammaF[k, n] = np.where(bad, np.nan, np.clip(gf, 0., 1.))
			eta_coh[k, n] = np.where(bad, np.nan, ec*100./np.sqrt(2.))
			eta_inc[k, n] = np.where(bad, np.nan, ei*100./np.sqrt(2.))
			gam_path[k, n] = gammaP[k, n][pw]

	# ---- coherent amplitude against fit length, stations and bands
	Acoh_L = np.full((nst, NMODE, FIT_BLOCKS.size), np.nan)
	gam_L_st = np.full((nst, NMODE, FIT_BLOCKS.size), np.nan)
	gam_L_reg = np.full((2, NMODE, FIT_BLOCKS.size), np.nan)
	kpk = wins.index(PLOT_WINDOW) if PLOT_WINDOW in wins else 0
	phis_ref = phis_w[kpk]

	for n in range(NMODE):
		ze_all = Zp[:, n, :]*(phis_ref[n]/(RHO0*G))[None, :]
		for il, Lb in enumerate(FIT_BLOCKS):
			ac, gm = sliding_coherence(ze_all[:, pw[st]], int(Lb))
			if ac.size:
				Acoh_L[:, n, il] = np.nanmean(ac, axis=0)*100.
				gam_L_st[:, n, il] = np.nanmean(gm, axis=0)
			ac, gm = sliding_coherence(Zp[:, n, shelf], int(Lb))
			if gm.size:
				gam_L_reg[0, n, il] = np.nanmedian(np.nanmean(gm, axis=0))
			ac, gm = sliding_coherence(Zp[:, n, deep], int(Lb))
			if gm.size:
				gam_L_reg[1, n, il] = np.nanmedian(np.nanmean(gm, axis=0))

	# ---- decorrelation timescale per block centre, shelf band
	cent = np.arange(nblock)
	gam_cent = np.full((nblock, NMODE, FIT_BLOCKS.size), np.nan)
	Tdec = np.full((nblock, NMODE), np.nan)
	Tcens = np.ones((nblock, NMODE), dtype='int8')
	for ic in cent:
		for il, Lb in enumerate(FIT_BLOCKS):
			s0 = ic - int(Lb)//2
			s1 = s0 + int(Lb)
			if s0 < 0 or s1 > nblock:
				continue
			seg = Zp[s0:s1][:, :, shelf]
			zm = np.abs(np.nanmean(seg, axis=0))
			am = np.nanmean(np.abs(seg), axis=0)
			with np.errstate(invalid='ignore', divide='ignore'):
				g = np.where(am > 0, zm/am, np.nan)
			gam_cent[ic, :, il] = np.nanmedian(g, axis=-1)
		for n in range(NMODE):
			Tdec[ic, n], Tcens[ic, n] = crossing_length(L_days,
													   gam_cent[ic, n])

	# ---- forcing series against which the timescale is read
	hp_reg = np.nanmean(hp_blk[:, shelf], axis=1)
	hp_sm = np.convolve(np.nan_to_num(hp_reg), np.ones(3)/3., mode='same')
	dt_blk = np.gradient((tblk - tblk[0])/np.timedelta64(1, 'D'))
	dhpdt = np.abs(np.gradient(hp_sm)/np.maximum(dt_blk, 1e-6))
	Q = load_discharge(tblk, tag)
	dQdt = np.abs(np.gradient(Q)/np.maximum(dt_blk, 1e-6))

	# ---- SWOT box, coherent and incoherent modal sea level
	ssh_coh = np.full(NMODE, np.nan)
	ssh_inc = np.full(NMODE, np.nan)
	ssh_coh_w = np.full((nwin, NMODE), np.nan)
	ssh_inc_w = np.full((nwin, NMODE), np.nan)
	for n in range(NMODE):
		ze = Zp[:, n, box]*(phis_ref[n, box]/(RHO0*G))[None, :]
		ec = np.abs(np.nanmean(ze, axis=0))
		er = np.sqrt(np.nanmean(np.abs(ze)**2, axis=0))
		ssh_coh[n] = np.nanmean(ec)*100./np.sqrt(2.)
		ssh_inc[n] = np.nanmean(np.sqrt(np.clip(er**2 - ec**2, 0., None)))\
			* 100./np.sqrt(2.)
		for k in range(nwin):
			m = wblk == k
			if m.sum() < 3:
				continue
			zek = Zp[m][:, n, box]*(phis_w[k, n, box]/(RHO0*G))[None, :]
			eck = np.abs(np.nanmean(zek, axis=0))
			erk = np.sqrt(np.nanmean(np.abs(zek)**2, axis=0))
			ssh_coh_w[k, n] = np.nanmean(eck)*100./np.sqrt(2.)
			ssh_inc_w[k, n] = np.nanmean(
				np.sqrt(np.clip(erk**2 - eck**2, 0., None)))*100./np.sqrt(2.)

	return dict(gammaF=gammaF, gammaP=gammaP, eta_coh=eta_coh,
				eta_inc=eta_inc, gam_path=gam_path, Acoh_L=Acoh_L,
				gam_L_st=gam_L_st, gam_L_reg=gam_L_reg,
				gam_cent=gam_cent, Tdec=Tdec, Tcens=Tcens,
				hp_reg=hp_reg, dhpdt=dhpdt, Q=Q, dQdt=dQdt, vfw=vfw_blk,
				ssh_coh=ssh_coh, ssh_inc=ssh_inc,
				ssh_coh_w=ssh_coh_w, ssh_inc_w=ssh_inc_w,
				c1=cn_w[:, 0], tblk=tblk, wblk=wblk)


# =====================================================================
# GEOMETRY, CACHE
# =====================================================================
def build_geometry():
	dsr = open_run(REALISTIC, chunks={'ocean_time': BLOCK_HOURS})
	dsc = open_run(CONTROL, chunks={'ocean_time': BLOCK_HOURS})
	lon = np.asarray(dsr.lon_rho)
	lat = np.asarray(dsr.lat_rho)
	h = np.asarray(dsr.h)
	mask = np.asarray(dsr.mask_rho) if 'mask_rho' in dsr else np.ones_like(h)
	pm = np.asarray(dsr.pm)
	pn = np.asarray(dsr.pn)
	ny, nx = h.shape
	flat = mask.ravel() > 0
	geom = dict(lon=lon, lat=lat, h=h, mask=mask, pm=pm, pn=pn,
				flat=flat, ny=ny, nx=nx)
	return dsr, dsc, geom


def build_bands(geom, path):
	"""Column selections shared by the two runs."""
	lon, lat, h, flat = geom['lon'], geom['lat'], geom['h'], geom['flat']
	hf = h.ravel()[flat]
	lof = lon.ravel()[flat]
	laf = lat.ravel()[flat]
	shelf = np.where((hf > SHELF_BAND[0]) & (hf < SHELF_BAND[1]))[0]
	deep = np.where(hf > DEEP_MIN)[0]
	box = np.where((hf > DEEP_MIN) & (lof > SWOT_BOX[0]) & (lof < SWOT_BOX[1])
				   & (laf > SWOT_BOX[2]) & (laf < SWOT_BOX[3]))[0]
	colidx = np.full(geom['ny']*geom['nx'], -1, dtype=int)
	colidx[flat] = np.arange(int(flat.sum()))
	colidx = colidx.reshape(geom['ny'], geom['nx'])
	geom['path_wet'] = colidx[path['j'], path['i']]
	print(f'  bands: shelf {shelf.size}, deep {deep.size}, '
		  f'SWOT box {box.size} columns')
	return dict(shelf=shelf, deep=deep, box=box)


def _cache_path(tag):
	return f'fig05_cache_{tag}.npz'


def save_cache(tag, r):
	d = {k: v for k, v in r.items() if k != 'tblk'}
	d['tblk'] = r['tblk'].astype('datetime64[s]').astype('int64')
	np.savez_compressed(_cache_path(tag), **d)
	print(f'  cached {tag} to {_cache_path(tag)} '
		  f'({os.path.getsize(_cache_path(tag))/1e6:.0f} MB)')


def load_cache(tag):
	p = _cache_path(tag)
	if not os.path.exists(p):
		return None
	z = np.load(p, allow_pickle=False)
	r = {k: z[k] for k in z.files}
	r['tblk'] = r['tblk'].astype('datetime64[s]')
	print(f'  loaded {tag} from {p}, streaming pass skipped')
	return r


# =====================================================================
# ASSEMBLY
# =====================================================================
def assemble(res, geom, path, bands, wins, outfile=OUTFILE):
	lon, lat, h, mask = geom['lon'], geom['lat'], geom['h'], geom['mask']
	flat, ny, nx = geom['flat'], geom['ny'], geom['nx']
	nwin = len(wins)
	nblock = res[RUNS[0]]['tblk'].size
	L_days = FIT_BLOCKS*BLOCK_HOURS/24.

	def unflat(v):
		a = np.full(ny*nx, np.nan)
		a[flat] = v
		return a.reshape(ny, nx)

	print('assembling')
	maps = {}
	for tag in RUNS:
		for name in ('gammaF', 'gammaP', 'eta_coh', 'eta_inc'):
			arr = np.full((nwin, NMODE, ny, nx), np.nan)
			for k in range(nwin):
				for n in range(NMODE):
					arr[k, n] = unflat(res[tag][name][k, n])
			maps[f'{name}_{tag}'] = arr
		c1 = np.full((nwin, ny, nx), np.nan)
		for k in range(nwin):
			c1[k] = unflat(res[tag]['c1'][k])
		maps[f'c1_{tag}'] = c1

	coords = dict(
		window=('window', wins),
		mode=('mode', np.arange(1, NMODE + 1)),
		eta_rho=('eta_rho', np.arange(ny)),
		xi_rho=('xi_rho', np.arange(nx)),
		path_pt=('path_pt', np.arange(path['dist'].size)),
		station=('station', np.arange(path['st_idx'].size)),
		fitlen=('fitlen', L_days),
		block=('block', np.arange(nblock)),
		region=('region', ['shelf', 'deep']),
	)
	dv = {}
	for tag in RUNS:
		r = res[tag]
		dv[f'gammaF_{tag}'] = (('window', 'mode', 'eta_rho', 'xi_rho'),
							   maps[f'gammaF_{tag}'])
		dv[f'gammaP_{tag}'] = (('window', 'mode', 'eta_rho', 'xi_rho'),
							   maps[f'gammaP_{tag}'])
		dv[f'eta_coh_{tag}'] = (('window', 'mode', 'eta_rho', 'xi_rho'),
								maps[f'eta_coh_{tag}'])
		dv[f'eta_inc_{tag}'] = (('window', 'mode', 'eta_rho', 'xi_rho'),
								maps[f'eta_inc_{tag}'])
		dv[f'c1_{tag}'] = (('window', 'eta_rho', 'xi_rho'), maps[f'c1_{tag}'])
		dv[f'gam_path_{tag}'] = (('window', 'mode', 'path_pt'), r['gam_path'])
		dv[f'Acoh_L_{tag}'] = (('station', 'mode', 'fitlen'), r['Acoh_L'])
		dv[f'gam_L_st_{tag}'] = (('station', 'mode', 'fitlen'),
								 r['gam_L_st'])
		dv[f'gam_L_reg_{tag}'] = (('region', 'mode', 'fitlen'),
								  r['gam_L_reg'])
		dv[f'gam_cent_{tag}'] = (('block', 'mode', 'fitlen'), r['gam_cent'])
		dv[f'Tdec_{tag}'] = (('block', 'mode'), r['Tdec'])
		dv[f'Tcens_{tag}'] = (('block', 'mode'), r['Tcens'])
		dv[f'hp_reg_{tag}'] = ('block', r['hp_reg'])
		dv[f'dhpdt_{tag}'] = ('block', r['dhpdt'])
		dv[f'Q_{tag}'] = ('block', r['Q'])
		dv[f'dQdt_{tag}'] = ('block', r['dQdt'])
		dv[f'vfw_{tag}'] = ('block', r['vfw'])
		dv[f'ssh_coh_{tag}'] = ('mode', r['ssh_coh'])
		dv[f'ssh_inc_{tag}'] = ('mode', r['ssh_inc'])
		dv[f'ssh_coh_w_{tag}'] = (('window', 'mode'), r['ssh_coh_w'])
		dv[f'ssh_inc_w_{tag}'] = (('window', 'mode'), r['ssh_inc_w'])
	dv.update(dict(
		lon_rho=(('eta_rho', 'xi_rho'), lon),
		lat_rho=(('eta_rho', 'xi_rho'), lat),
		h=(('eta_rho', 'xi_rho'), h),
		mask_rho=(('eta_rho', 'xi_rho'), mask),
		path_lon=('path_pt', path['lon']),
		path_lat=('path_pt', path['lat']),
		path_dist=('path_pt', path['dist']),
		station_index=('station', path['st_idx']),
		station_dist=('station', path['st_dist']),
		station_lon=('station', path['lon'][path['st_idx']]),
		station_lat=('station', path['lat'][path['st_idx']]),
		block_time=('block', res[RUNS[0]]['tblk']),
	))

	ds_out = xr.Dataset(dv, coords=coords)
	ds_out.attrs.update(dict(
		realistic=REALISTIC, control=CONTROL, plot_window=PLOT_WINDOW,
		nmode=NMODE, block_hours=BLOCK_HOURS, gamma_cut=GAMMA_CUT,
		eos_mode=EOS_MODE, eos=eos_provenance(),
		path_site=PATH_SITE, swot_box=str(SWOT_BOX),
		shelf_band=str(SHELF_BAND), deep_min=DEEP_MIN,
		discharge_source=f'{DISCHARGE_YEAR} monthly means interpolated '
		f'to daily, {DISCHARGE_ANCHOR} anchoring, CTRL held at '
		f'{CTRL_Q:.3g} m3 s-1',
		discharge_monthly=' '.join(f'{q:.0f}' for q in DISCHARGE_MONTHLY),
		windows=';'.join(f'{k}:{v[0]}..{v[1]}' for k, v in WINDOWS.items()),
		note='gamma is the vector mean of the block M2 modal amplitudes '
			 'over the mean of their magnitudes, bounded by unity. '
			 'gammaF uses the modal flux built from the vector mean '
			 'amplitudes over the mean block flux magnitude. Sea level '
			 'is eta_n = p_n phi_n(0)/(rho0 g), reported as the standard '
			 'deviation in cm so that it compares directly with the '
			 'SWOT bands of Tchilibou et al. (2025).',
	))
	print('writing', outfile)
	try:
		enc = {v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars}
		ds_out.to_netcdf(outfile, encoding=enc)
	except ValueError as err:
		warnings.warn(f'compressed write failed ({err}), writing plain')
		ds_out.to_netcdf(outfile)

	summary(res, wins, L_days)


def summary(res, wins, L_days):
	print('\n--- discharge seen by the blocks, m3 s-1 ---')
	wb = res['ref']['wblk']
	for k, w in enumerate(wins):
		m = wb == k
		if not m.any():
			continue
		q = res['ref']['Q'][m]
		dq = res['ref']['dQdt'][m]
		print(f'{w:>11}  REF mean {np.nanmean(q):8.0f}  range '
			  f'{np.nanmin(q):8.0f} to {np.nanmax(q):8.0f}  '
			  f'mean |dQ/dt| {np.nanmean(dq):7.0f} per day   '
			  f'CTRL {CTRL_Q:8.0f}')
	qr = res['ref']['Q']
	print(f'      record  REF mean {np.nanmean(qr):8.0f}, '
		  f'CTRL minus REF mean {CTRL_Q - np.nanmean(qr):+8.0f}')

	print('\n--- summary, coherent fraction of the M2 modal flux ---')
	for k, w in enumerate(wins):
		for tag in RUNS:
			g = res[tag]['gammaF'][k]
			line = f'{w:>11} {tag:>5}  shelf median '
			line += ' '.join(f'M{n+1} {np.nanmedian(g[n]):5.3f}'
							 for n in range(NMODE))
			print(line)
	print('\n--- coherent fraction against fit length, shelf band ---')
	for tag in RUNS:
		for n in range(NMODE):
			g = res[tag]['gam_L_reg'][0, n]
			print(f'{tag:>5} mode {n+1}  ' +
				  '  '.join(f'{L:4.1f}d {v:4.2f}' for L, v in
							zip(L_days, g)))
	print('\n--- coherent fraction against fit length, deep band ---')
	for tag in RUNS:
		for n in range(NMODE):
			g = res[tag]['gam_L_reg'][1, n]
			print(f'{tag:>5} mode {n+1}  ' +
				  '  '.join(f'{L:4.1f}d {v:4.2f}' for L, v in
							zip(L_days, g)))
	print('\n--- decorrelation timescale, days, shelf band ---')
	for tag in RUNS:
		T = res[tag]['Tdec']
		C = res[tag]['Tcens']
		for n in range(NMODE):
			ok = C[:, n] == 0
			med = np.nanmedian(T[ok, n]) if ok.any() else np.nan
			print(f'{tag:>5} mode {n+1}  median {med:6.2f}   '
				  f'resolved in {100*ok.mean():5.1f}% of the centres')
	print('\n--- M2 baroclinic sea level in the SWOT box, cm std ---')
	for tag in RUNS:
		c = res[tag]['ssh_coh']
		i = res[tag]['ssh_inc']
		tot = np.sqrt(c**2 + i**2)
		for n in range(NMODE):
			frac = 100*(c[n]**2)/max(tot[n]**2, 1e-12)
			print(f'{tag:>5} mode {n+1}  coherent {c[n]:5.3f}  '
				  f'incoherent {i[n]:5.3f}  total {tot[n]:5.3f}  '
				  f'coherent share of variance {frac:5.1f}%')
	print('  SWOT track 20 for reference, total SLA std 1.03, 0.58 and '
		  '0.74 cm for mode 1, mode 2 and the higher modes, with 24, 16 '
		  'and 4 per cent of the variance removed by the coherent atlas '
		  '(Tchilibou et al. 2025, their Table 2)')


def main():
	print('opening runs')
	dsr, dsc, geom = build_geometry()
	wins = list(WINDOWS.keys())

	if not os.path.exists(FIG04FILE):
		raise SystemExit(f'{FIG04FILE} not found, run fig04_compute first, '
						 'the propagation path is traced through its peak '
						 'window modal flux field')
	f4 = xr.open_dataset(FIG04FILE)
	path = build_path(f4, geom)
	f4.close()
	bands = build_bands(geom, path)

	res = {}
	for tag, ds in (('ref', dsr), ('ctrl', dsc)):
		if tag not in RUNS:
			continue
		cached = load_cache(tag) if USE_CACHE else None
		if cached is not None:
			res[tag] = cached
			continue
		tic = _time.time()
		res[tag] = process_run(ds, tag, wins, geom, path, bands)
		print(f'  {tag} done in {(_time.time() - tic)/60:.1f} min')
		try:
			save_cache(tag, res[tag])
		except Exception as err:
			warnings.warn(f'could not cache {tag}: {err}')

	assemble(res, geom, path, bands, wins)


if __name__ == '__main__':
	import sys
	if '--assemble-only' in sys.argv:
		print('assemble only, using the cached reductions')
		dsr, dsc, geom = build_geometry()
		wins = list(WINDOWS.keys())
		f4 = xr.open_dataset(FIG04FILE)
		path = build_path(f4, geom)
		f4.close()
		bands = build_bands(geom, path)
		res = {t: load_cache(t) for t in RUNS}
		missing = [t for t, v in res.items() if v is None]
		if missing:
			raise SystemExit(f'no cache for {missing}, run without the flag')
		assemble(res, geom, path, bands, wins)
	else:
		main()