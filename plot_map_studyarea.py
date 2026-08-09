import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import netCDF4 as nc
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry as sgeom
from datetime import datetime, timedelta
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patheffects import Stroke
import shapely.geometry as sgeom
#import cartopy.feature as cf
from matplotlib import colormaps
from matplotlib import cm
import xroms
import matplotlib.colors as mcolors
import cmocean.cm as cmo
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.dates as mdates

# ploting grid with locations 
fpath_gebco = '/home/rafaela/ROMS/pyroms_tools_arian/data/raw/gebco_2020_005.nc'
fpath_grid = '/home/rafaela/ROMS/pyroms_tools_arian/data/grid/pca_grd_river_open.nc'
figpath = '/home/rafaela/ROMS/projects/tese/'
path = '/data1/amasseds_run/const/'
roms = 'roms_am2_redo.nc'

path_master = '/data0/rafaela/ROMS/projects/amasseds_redo/input/'
era_file = 'era5_1990b.nc'

area_slice = {'eta_rho': slice(165, 295), 'xi_rho': slice(45, 100)}
area_cnb =  {'eta_rho': slice(165, 295)}

ds_gebco = xr.open_dataset(fpath_gebco)
ds_grid = xr.open_dataset(fpath_grid)

# bathymetry
latb=ds_gebco.lat.data
lonb=ds_gebco.lon.data
zbat=ds_gebco.elevation.data
LObat,LAbat=np.meshgrid(lonb,latb)



grid = ds_grid.copy()

##################### Mean wind inside the volume ########################

grilon = grid.lon_rho.isel(**area_slice)
grilat = grid.lat_rho.isel(**area_slice)
wind = xr.open_dataset(path_master+ era_file).sel(lon= slice(grilon.min(), grilon.max()), lat= slice(grilat.min(),grid.lat_rho.max()))
wind = wind.mean(dim=['lon', 'lat'])

###################### Discharge #######################################

# 1. Seus dados originais
t_a = [  175000, 206000, 240000, 240000, 
        213000, 177000, 143000, 114000, 107000, 116000,133000, 154000]

# 2. Criar as datas correspondentes ao meio de cada mês de 1990
datas_mensais = pd.date_range(start='1989-12-1', periods=12, freq='M') + pd.Timedelta(days=30)

# 3. Criar o esqueleto do dataframe diário (01/01/1990 até 31/12/1990)
datas_diarias = pd.date_range(start='1990-01-01', end='1990-12-31', freq='D')
df = pd.DataFrame(index=datas_diarias)

# 4. Inserir as médias mensais nos dias centrais correspondentes
df_mensal = pd.Series(t_a, index=datas_mensais)
df['media'] = df_mensal

# 5. Interpolar
# O método 'pchip' ou 'cubic' garante suavidade. 
# 'limit_direction="both"' preenche os dias antes de 15/jan e após 15/dez.
df['media_interpolada'] = df['media'].interpolate(method='pchip', limit_direction='both')
df = df['1990-01-01':'1990-07-31']

#glofas = df['media_interpolada'][v_daily[0].ocean_time[0].values : v_daily[0].ocean_time[-1].values]

# --- Configurações Estéticas Globais ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12

# 1. Padronização do Tempo
common_time = df.index
wind_aligned = wind.interp(time=common_time)

#localização do maregrafo Santana está em grau min seg
las= [-0,3,41] 
los = [-51,9,57]

m_lat1 = las[0]-(las[1]/60) -(las[2]/3600)
m_lon1 = los[0]-(los[1]/60) -(los[2]/3600)

#localização do maregrafo Belem está em grau min seg
la= [-1,18,10] # 1°18'10" S
lo = [-48,30,16]

m_lat2 = la[0]-(la[1]/60) -(la[2]/3600)
m_lon2 = lo[0]-(lo[1]/60) -(lo[2]/3600)

#localização do maregrafo Ribamar
m_lon3 = -44.051739
m_lat3 = -2.563409

# M1 amasseds
M_lat1=3 + 4.51/60
M_lon1=-50 -18.80/60

# M2
M_lat2=3 +23.12/60
M_lon2=-49 -56.23/60

# M3
M_lat3 = 4 + 4.29/60
M_lon3 = -49 - 37.35/60


# determining features for plot
land = cfeature.NaturalEarthFeature('physical', 'land', scale='10m')
states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m')
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale = '10m')
                                    
# dict with text for plot

Rivers = {  'Rio Amazonas': [ -52.176166178928966, -1.541752298824613, 'Rio Amazonas'],
		  	'Pará River': [ -49.640230,-2.181433, 'Rio Pará'],
            }



# dict with text for plot


names2 = {  'M1': [ M_lon1-0.1, M_lat1, 'M1'],
            'M2': [ M_lon2-0.1, M_lat2, 'M2'],
            'M3': [ M_lon3-0.1, M_lat3, 'M3'],
            'P1': [ m_lon1-0.1, m_lat1, 'P1']}
names3 = {  'P2': [ m_lon2+0.1, m_lat2, 'P2'],
            'P3': [ m_lon3+0.1, m_lat3, 'P3']}

dots3 = {   'M1': [ M_lon1, M_lat1, '.'],
            'M2': [ M_lon2, M_lat2, '.'],
            'M3': [ M_lon3, M_lat3, '.'],
            'P1': [ m_lon1, m_lat1, '.'],
            'P2': [ m_lon2, m_lat2, '.'],
            'P3': [ m_lon3, m_lat3, '.']}
texto = {'Domínio Numerico':[-46.302300,3.181787, 'Domínio Numérico']}

import matplotlib.colors as mcolors
import cmocean.cm as cmo
import matplotlib as mpl
rdrag = ds_grid.rdrag2.values/1000
vmin = np.min(rdrag)
vmax = np.max(rdrag)
levels = 11
norm = cm.colors.Normalize(vmin=vmin,vmax=vmax)	
mapi = 'PiYG'
# Ploting Map

fig = plt.figure(figsize=[8,10], constrained_layout=True) 
gs = gridspec.GridSpec(3, 1, figure=fig, 
                       width_ratios=[1], 
                       height_ratios=[1, .25, 0.25],
                       hspace=0.001, wspace=0.05)


ax1 =fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax1.set_aspect('auto') # Garante que ele ocupe a altura toda se necessário

ax1.contour(LObat, LAbat, zbat, [-3000,-2500,-2000,-1500,-1000,
            ---500,-400,-300], latlon=True, zorder=15,colors='dimgray', linestyles='-',
            linewidths=0.7, alpha=0.5)
l = ax1.contour(LObat, LAbat, zbat, [-200, -50], latlon=True, zorder=15, colors='black', linestyles='-', linewidths=0.7)
plt.clabel(l, fontsize=9, fmt='%1.0f')  
ax1.plot(ds_grid.lon_rho[0,:], ds_grid.lat_rho[0,:], color = 'darkblue', alpha=.7, linewidth=1.5, zorder = 3)     
ax1.plot(ds_grid.lon_rho[-1,:], ds_grid.lat_rho[-1,:], color = 'darkblue', alpha=.7,  linewidth=1.5, zorder = 3)  
ax1.plot(ds_grid.lon_rho[:,-1], ds_grid.lat_rho[:,-1], color = 'darkblue', alpha=.7,  linewidth=1.5, zorder = 3)  
ax1.plot(ds_grid.lon_rho[150,:], ds_grid.lat_rho[150,:], color = 'red', alpha=.7,  linewidth=3.5, zorder = 3)  
ax1.set_title('A',x=0.05,y=0.02, fontsize=18,color='black', fontweight='bold', zorder = 11 )
ax1.set_extent([lonb.min(), lonb.max()-4, latb.min()+.5, latb.max()-3.5], crs=ccrs.PlateCarree())
ax1.set_aspect('auto', adjustable='box') # Isso força o mapa a esticar para a largura do grid
ax1.coastlines(resolution='10m', linewidths=0.5)
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha = 0.2, linewidth=0.6, color='grey', linestyle='--', zorder = 10)
ax1.add_feature(land, facecolor='wheat', edgecolor='grey', linewidths=0.5, zorder = 4)
ax1.add_feature(states, facecolor='wheat', edgecolor='grey', linewidths=0.5, zorder = 5)

gl.xlabels_top = True
gl.ylabels_right = False
gl.xlines = True
gl.ylines = True
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

_ = [ax1.text(*names2[k], color='black',fontweight='bold', fontsize=10, zorder=16,va='top', ha='right') for k in names2]
_ = [ax1.text(*names3[k], color='black',fontweight='bold', fontsize=10, zorder=16,va='top', ha='left') for k in names3]
_ = [ax1.plot(*dots3[k], color='black', zorder=16, mew=3, ) for k in dots3]

ax1.spines['geo'].set_edgecolor('grey')
ax1.spines['geo'].set_linewidth(1)
ax1.spines['geo'].set_zorder(30) # Ensure it stays on top of other layers

[ax1.text(*Rivers[k], color='brown', fontsize=12.5, zorder=16,rotation = 45, ha = 'left', va='bottom', fontweight='bold') for k in Rivers]
[ax1.text(*texto[k], color='darkblue', zorder=20, rotation = -45, ha = 'center', va='center', fontweight='bold') for k in texto]

# ax2 = fig1.add_subplot(spec[1])
def clean_axis(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.grid(True, which='major', alpha=0.3, linewidth=0.6)
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.1, linewidth=0.4)
    ax.tick_params(axis='x', labelbottom=True)

ax0 = fig.add_subplot(gs[1, 0])

# --- PAINEL 2: VENTO ERA5 ---

q_wind = ax0.quiver(common_time[::5], 0, wind_aligned.u10[::5], wind_aligned.v10[::5], 
                   color='navy', scale=60, width=0.005)

# Escala do vento com ms-1
ax0.quiverkey(q_wind, X=0.95, Y=1.05, U=5, label=r'5 m s$^{-1}$', 
              labelpos='E', coordinates='axes', fontproperties={'size': 10})

ax0.set_ylabel(r'm s$^{-1}$', fontsize=10)
ax0.set_ylim(-1, 1)
clean_axis(ax0)

ax0.set_title('B',x=0.05,y=0.09, fontsize=18,color='black', fontweight='bold' )




ax3 = fig.add_subplot(gs[2, 0])

# --- PAINEL 4: DESCARGA GLOFAS ---

ax3.plot(common_time, df/100000, color='black', lw=1.8)
ax3.set_ylabel( r'$\times 10^5$ m$^{3}$ s$^{-1}$', fontsize=10)
clean_axis(ax3)

# --- FORMATAÇÃO PADRONIZADA DE DATAS ---
axes_list = [ax0, ax3]
for ax in axes_list:
    ax.set_xlim(common_time.values[0], common_time.values[-1])
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.setp(ax.get_xticklabels(), rotation=15, ha='center', fontsize='small')

fig.align_ylabels(axes_list)
ax3.set_aspect('auto', adjustable='box')#('auto')
ax3.set_title('C',x=0.05,y=0.02, fontsize=18,color='black', fontweight='bold', zorder = 11 )


plt.savefig('grid_and_discharge.jpeg', dpi=120)


