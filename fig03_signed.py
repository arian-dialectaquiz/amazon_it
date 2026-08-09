
# =====================================================================
# FIGURE SKELETON
# =====================================================================
fig = plt.figure(figsize=(16.5, 19.5))
gs = gridspec.GridSpec(4, 6, figure=fig,
                       height_ratios=[1.0, 1.0, 0.62, 0.62],
                       hspace=0.2, wspace=0.50,
                       left=0.05, right=0.98, top=0.972, bottom=0.045)

gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :],
                                          wspace=TOP_WSPACE)
gs_mid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :],
                                          wspace=TOP_WSPACE)

axa = fig.add_subplot(gs_top[0], projection=ccrs.PlateCarree())
axb = fig.add_subplot(gs_top[1], projection=ccrs.PlateCarree())
axc = fig.add_subplot(gs_top[2], projection=ccrs.PlateCarree())
axd = fig.add_subplot(gs_mid[0], projection=ccrs.PlateCarree())
axe = fig.add_subplot(gs_mid[1], projection=ccrs.PlateCarree())
axf = fig.add_subplot(gs_mid[2], projection=ccrs.PlateCarree())
axg = fig.add_subplot(gs[2, :])
axh = fig.add_subplot(gs[3, 0:2])
axi = fig.add_subplot(gs[3, 2:4])
axj = fig.add_subplot(gs[3, 4:6])


# =====================================================================
# (a b c)  CONVERSION, THREE WINDOWS, REF
# =====================================================================
norm_C = mcolors.TwoSlopeNorm(vmin=-C_LIM, vcenter=0, vmax=C_LIM)
for ax, wname, tag in ((axa, 'rising', 'a'), (axb, 'peak', 'b'),
                       (axc, 'decreasing', 'c')):
    if wname not in wins:
        continue
    k = wins.index(wname)
    C = ds.C_ref.isel(window=k).values
    pcm = ax.pcolormesh(lon, lat, np.where(mask > 0, C, np.nan),
                        cmap='RdBu_r', norm=norm_C, shading='auto',
                        zorder=1, rasterized=True)
    dress(ax, labels_left=(tag == 'a'))
    ax.set_title(wname, fontsize=11, pad=4)
    ax.text(0.03, 0.96, tag, transform=ax.transAxes, fontsize=15,
            fontweight='bold', va='top', ha='left', zorder=20,
            bbox=dict(fc='w', ec='0.4', boxstyle='round,pad=0.2', alpha=0.9))
    if CBAR_IN_EACH or tag == 'a':
        cbar_on_land(ax, pcm, r'$C$ (W m$^{-2}$)')


# =====================================================================
# (d e f)  DEPTH INTEGRATED BAROCLINIC FLUX WITH PROPAGATION VECTORS
# =====================================================================
norm_F = mcolors.Normalize(vmin=0, vmax=F_MAX)
st = VEC_STEP
for ax, wname, tag in ((axd, 'rising', 'd'), (axe, 'peak', 'e'),
                       (axf, 'decreasing', 'f')):
    if wname not in wins:
        continue
    k = wins.index(wname)
    Fxi = ds.Fxi_ref.isel(window=k).values
    Feta = ds.Feta_ref.isel(window=k).values
    # grid components to east and north for plotting
    Fe = Fxi*np.cos(ang) - Feta*np.sin(ang)
    Fn = Fxi*np.sin(ang) + Feta*np.cos(ang)
    Fm = np.hypot(Fe, Fn)/1e3                      # kW m-1

    pcf = ax.pcolormesh(lon, lat, np.where(mask > 0, Fm, np.nan),
                        cmap=cmo.matter, norm=norm_F, shading='auto',
                        zorder=1, rasterized=True)
    sel = (mask > 0) & np.isfinite(Fm) & (Fm > 0.05*F_MAX)
    q = ax.quiver(lon[::st, ::st], lat[::st, ::st],
                  np.where(sel, Fe, np.nan)[::st, ::st],
                  np.where(sel, Fn, np.nan)[::st, ::st],
                  color='0.15', scale=VEC_SCALE*1e3, width=0.003,
                  zorder=12, alpha=0.85)
    if tag == 'f':
        ax.quiverkey(q, 0.78, 0.06, 2e3, r'2 kW m$^{-1}$',
                     labelpos='E', coordinates='axes', fontproperties={'size': 7})
    dress(ax, labels_left=(tag == 'd'))
    ax.text(0.03, 0.96, tag, transform=ax.transAxes, fontsize=15,
            fontweight='bold', va='top', ha='left', zorder=20,
            bbox=dict(fc='w', ec='0.4', boxstyle='round,pad=0.2', alpha=0.9))
    if CBAR_IN_EACH or tag == 'd':
        cbar_on_land(ax, pcf, r'$|\mathbf{F}_{bc}|$ (kW m$^{-1}$)',
                     extend='max')


# =====================================================================
# (g)  SHELF BREAK CONVERSION TIME SERIES
# =====================================================================
tb = ds.block_time.values
Cr = ds.C_block_ref.values*1e4          # 10^-4 W m-2
Cc = ds.C_block_ctrl.values*1e4

if np.isfinite(NOISE):
    band = NOISE*1e4
    m = np.nanmean(Cc)
    axg.axhspan(m - band, m + band, color='0.65', alpha=0.30, lw=0, zorder=0)
    #axg.text(0.015, 0.05, 'CTRL fortnightly variability',
             #transform=axg.transAxes, fontsize=7.5, color='0.35')

axg.plot(tb, Cc, color=C_CTRL, lw=1.6, label='CTRL, constant discharge')
axg.plot(tb, Cr, color=C_REF, lw=1.8, label='REF, variable discharge')

# spring and neap from the sea surface height envelope
env = ds.springneap_env.values
tt = ds.time.values
if np.isfinite(env).any():
    e = env/np.nanmax(env)
    pk = (e[1:-1] > e[:-2]) & (e[1:-1] > e[2:])
    tr = (e[1:-1] < e[:-2]) & (e[1:-1] < e[2:])
    for x in tt[1:-1][pk]:
        axg.axvline(x, color='crimson', lw=0.8, ls=':', alpha=0.7, zorder=1)
    for x in tt[1:-1][tr]:
        axg.axvline(x, color='royalblue', lw=0.8, ls=':', alpha=0.7, zorder=1)
    axg.plot([], [], color='crimson', lw=0.8, ls=':', label='spring')
    axg.plot([], [], color='royalblue', lw=0.8, ls=':', label='neap')

wcol = {'rising': '#8ecae6', 'peak': '#ffb703', 'decreasing': '#90be6d'}
for item in ds.attrs.get('windows', '').split(';'):
    if ':' not in item:
        continue
    name, v = item.split(':'); t0, t1 = v.split('..')
    axg.axvspan(np.datetime64(t0), np.datetime64(t1),
                color=wcol.get(name, '0.7'), alpha=0.16, lw=0, zorder=0)
    axg.text(np.datetime64(t0) + (np.datetime64(t1) - np.datetime64(t0))/2,
             0.96, name, transform=axg.get_xaxis_transform(), ha='center',
             va='top', fontsize=8, color='0.25', fontweight='bold')

axg.set_ylabel(r'$C$ over the shelf break (10$^{-4}$ W m$^{-2}$)')
axg.set_xlabel(None)
for l in axg.get_xticklabels():
    l.set_rotation(20); l.set_ha('right')
axg.legend(loc=3, ncol=4, framealpha=0.9)
axg.grid(alpha=0.22, lw=0.4)
axg.set_title(f'area average over the {BAND[0]:.0f} to {BAND[1]:.0f} m band',
              fontsize=9, color='0.35', loc='right')
axg.text(0.012, 0.965, 'g', transform=axg.transAxes, fontsize=15,
         fontweight='bold', va='top', ha='left')


# =====================================================================
# (h) (i) (j)  SIGNED DECOMPOSITION OF THE CONVERSION BY DEPTH BAND
# =====================================================================
# Positive C is barotropic to baroclinic transfer, negative C the reverse
# via pressure work. Where the two nearly cancel, the net is a residual of
# two large numbers and is not by itself a measure of generation, so the
# positive part and the cancellation index are shown alongside it.
A = cell_area(lon, lat)

REGIONS = [('shelf, $h<250$ m',            0.,   250., 'h'),
           ('upper slope, 250 to 1000 m',  250., 1000., 'i'),
           ('deep, 1000 to 3500 m',        1000., 3500., 'j')]

C_POS = '#2a6fb5'
C_NEG = '#d1495b'


def signed_integrals(C, m):
    w = np.where(m & np.isfinite(C), A, 0.0)
    c = np.nan_to_num(C)
    pos = np.nansum(np.where(c > 0, c, 0.)*w)/1e6
    neg = np.nansum(np.where(c < 0, c, 0.)*w)/1e6
    absC = pos - neg
    kap = 1 - abs(pos + neg)/absC if absC else np.nan
    return pos, neg, pos + neg, kap


for ax, (rname, h0, h1, tag) in zip((axh, axi, axj), REGIONS):
    m = (h >= h0) & (h < h1) & (mask > 0)
    xs, labs, kaps = [], [], []
    x = 0
    for k, w in enumerate(wins):
        for tg, hatch in (('ref', ''), ('ctrl', '///')):
            C = ds[f'C_{tg}'].isel(window=k).values
            pos, neg, net, kap = signed_integrals(C, m)
            ax.bar(x, pos, width=0.78, color=C_POS, hatch=hatch,
                   edgecolor='k', lw=0.4,
                   label=r'$\int C^{+}$' if x == 0 else None)
            ax.bar(x, neg, width=0.78, color=C_NEG, hatch=hatch,
                   edgecolor='k', lw=0.4,
                   label=r'$\int C^{-}$' if x == 0 else None)
            ax.plot(x, net, marker='D', ms=7, mfc='k', mec='w', mew=1.0,
                    ls='none', zorder=6,
                    label='net' if x == 0 else None)
            ax.annotate(f'{net:.0f}', xy=(x, net), xytext=(0, 11),
                        textcoords='offset points', ha='center',
                        fontsize=6.5, fontweight='bold')
            xs.append(x); labs.append(f'{w[:4]}\n{tg.upper()}')
            kaps.append((x, kap))
            x += 1
        x += 0.5

    for xk, kap in kaps:
        ax.annotate(f'$\\kappa$={kap:.2f}', xy=(xk, -400),
                    xytext=(0, -14), textcoords='offset points',
                    ha='center', fontsize=6.5, color='0.3',
                    annotation_clip=False)

    ax.axhline(0, color='k', lw=0.9)
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel('MW')
    ax.set_title(rname, fontsize=10)
    ax.grid(alpha=0.22, lw=0.4, axis='y')
    ax.margins(y=0.24)
    ax.text(0.02, 0.965, tag, transform=ax.transAxes, fontsize=15,
            fontweight='bold', va='top', ha='left')
    if tag == 'i':
        ax.plot([], [], ' ', label='hatched, CTRL')
        ax.legend(loc='upper center', framealpha=0.9, fontsize=7.5, ncol=2)

#axj.text(1.0, -0.30, r'$\kappa = 1 - |\int C| / \int |C|$, the cancellation '
                     #r'index', transform=axj.transAxes, ha='right',
         #va='top', fontsize=8, color='0.3')


# =====================================================================
fig.savefig(OUTFIG, dpi=DPI, bbox_inches='tight')
print('wrote', OUTFIG)