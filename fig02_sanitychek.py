"""Check the Figure 2 diagnostics against analytic limits."""
import numpy as np, sys, types
for m in ['xroms','gsw','cartopy','cmocean','matplotlib']:
    sys.modules.setdefault(m, types.ModuleType(m))
src2 = open('fig02_compute.py').read()
src1 = open('fig01_compute.py').read()
ns = {'np': np, 'G': 9.81, 'RHO0': 1025.0, 'OMEGA_E': 7.2921e-5,
      'R_EARTH': 6.371e6, 'MAX_R': 5000., 'N2_FLOOR':1e-9, 'CHUNK_COLS':4000,
      'NMODES':4, 'HAS_GSW': False}
exec(src1[src1.index('def vertical_modes'):src1.index('def grad_h')], ns)
for fn in ['def phi_from_Phi','def surface_trapping','def deformation_radii',
           'def two_layer_c1']:
    a = src2.index(fn)
    b = min(x for x in [src2.index('\ndef ', a+1), len(src2)] if x > a)
    exec(src2[a:b], ns)

vm, phi_from_Phi = ns['vertical_modes'], ns['phi_from_Phi']
trap, defrad = ns['surface_trapping'], ns['deformation_radii']

print('='*70)
print('TEST 1  equatorial vs midlatitude deformation radius')
print('='*70)
beta0 = 2*7.2921e-5/6.371e6
for c1 in (0.3, 0.6, 1.0):
    print(f'  c1 = {c1:.1f} m/s,  Req = sqrt(c1/beta) = '
          f'{np.sqrt(c1/beta0)/1e3:5.0f} km')
    for la in (0.0, 0.5, 1.0, 2.0, 4.0, 6.0):
        Rf, Req, Reff = defrad(np.array([c1]), np.array([la]))
        which = 'equatorial' if Reff[0] == Req[0] else 'midlatitude'
        print(f'    lat {la:4.1f}  Rf {Rf[0]:7.0f}  Req {Req[0]:5.0f}  '
              f'Reff {Reff[0]:5.0f} km   {which}')
    print()

print('='*70)
print('TEST 2  surface trapping index, analytic limits')
print('='*70)
H, Nw = 100., 61
zw = np.linspace(-H, 0., Nw)[:, None]
zr = 0.5*(zw[1:] + zw[:-1])
# uniform N2 -> mode 1 is bottom controlled, energy spread over the column
N2 = np.full((Nw,1), 1e-4)
c, Phi = vm(N2, zw, nmodes=3, free_surface=True)
ph = phi_from_Phi(Phi[:,1,:], zw, c[1])
for hp in (20., 50.):
    T = trap(ph, zr, zw, np.array([hp]))
    print(f'  uniform N2, hp={hp:4.0f} m: T = {T[0]:5.3f}  '
          f'(depth fraction {hp/H:4.2f})')
# strong 20 m halocline -> mode 1 becomes surface trapped
N2 = np.full((Nw,1), 2e-5); N2[np.abs(zw+20.)<=2.5] = 5e-3
c, Phi = vm(N2, zw, nmodes=3, free_surface=True)
ph = phi_from_Phi(Phi[:,1,:], zw, c[1])
T = trap(ph, zr, zw, np.array([20.]))
print(f'  20 m plume,  hp=  20 m: T = {T[0]:5.3f}   c1 = {c[1,0]:.3f} m/s')
print('\n  For a uniform column T tracks the depth fraction. A strong')
print('  halocline should push T well above it, which is the panel (f2) claim.')

print()
print('='*70)
print('TEST 3  two layer estimate against the full solution')
print('='*70)
for H, hp, drho in ((60., 15., 4.), (100., 20., 5.), (200., 25., 6.)):
    Nw = 81
    zw = np.linspace(-H, 0., Nw)[:, None]
    gp = 9.81*drho/1025.
    N2 = np.full((Nw,1), 1e-6); d = 3.0
    N2[np.abs(zw+hp) <= d/2] = gp/d
    c, _ = vm(N2, zw, nmodes=3, free_surface=True)
    ana = np.sqrt(gp*hp*(H-hp)/H)
    print(f'  H={H:5.0f} hp={hp:4.0f}: full {c[1,0]:5.3f}  two layer {ana:5.3f}'
          f'  err {100*(c[1,0]-ana)/ana:+5.1f} %')