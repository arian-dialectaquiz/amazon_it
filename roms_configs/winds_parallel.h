/*
** svn $Id: upwelling.h 1001 2020-01-10 22:41:16Z arango $
*******************************************************************************
** Copyright (c) 2002-2020 The ROMS/TOMS Group                               **
**   Licensed under a MIT/X style license                                    **
**   See License_ROMS.txt                                                    **
*******************************************************************************
**
** Options for Upwelling Test.
**
** Application flag:   WINDS_PARALLEL
** Input script:       roms_parallel.in
*/

#define UV_ADV		/*to turn ON advection terms */
#define UV_COR		/*to turn ON Coriolis term  */
#define UV_VIS2	/*to turn ON harmonic horizontal mixing */
#define NONLIN_EOS
/*#define UV_SMAGORINSKY*/	/* to turn ON Smagorinsky-like viscosity*/
#define UV_DRAG_GRID	/*spatially varying bottom friction parameters*/
#define UV_QDRAG	/*to turn ON quadratic bottom friction*/
#undef MIX_S_UV	/*if mixing along constant S-surfaces*/
#define MIX_GEO_UV	/*turn off mixing constant Z surfaces*/
#define SPLINES_VVISC	/*splines reconstruction of vertical viscosity*/
#define TS_DIF2	/*to turn ON harmonic horizontal mixing*/
#undef  TS_DIF4	/*to turn OFF harmonic horizontal mixing*/
/*#define TS_SMAGORINSKY*/	/*to turn ON Smagorinsky-like diffusion */
#define MIX_GEO_TS	/*turn OFF mixing on constant Z surfaces*/
#undef MIX_S_TS	/*if mixing along constant S-surfaces(tirar quando rodar o baroclinico)*/
#define RADIATION_2D
#define SALINITY	/*if having salinity*/
#define SPLINES_VDIFF	/*if splines reconstruction of vertical diffusion*/
#define DJ_GRADPS	/*if splines density Jacobian (Shchepetkin, 2000)*/
#define SOLVE3D	/*if solving 3D primitive equations*/
#define MASKING	/*if land/sea masking */
#define AVERAGES	/*if writing out NLM time-averaged data*/
/*#define AVERAGES_DETIDE*/
#define DIAGNOSTICS_UV	/*if writing out momentum diagnostics*/
#define DIAGNOSTICS_TS	/*if writing out tracer diagnostics*/

#define TIDES_ASTRO
#define POT_TIDES
#define SSH_TIDES   /*if imposing tidal elevation*/
#define UV_TIDES	/*if imposing tidal currents*/
#define RAMP_TIDES	/*if ramping (over one day) tidal forcing*/
#define ADD_FSOBC	/*to add tidal elevation to processed OBC data*/
#define ADD_M2OBC	/*to add tidal currents  to processed OBC data*/
/*#define ANA_FSOBC	/*if analytical free-surface boundary conditions*/
/*#define ANA_M2OBC 	/*if analytical 2D momentum boundary conditions*/
/*#define ANA_TOBC*/   /* if analytical tracer boundary conditions*/ 
#define WET_DRY	/*to activate wetting and drying*/
#define CURVGRID
/*#define ANA_GRID*/
/*#define ANA_INITIAL	/*if analytical initial conditions*/
/*#define ANA_SMFLUX	/* deactivate if u wanna use a input file for u and v wind stress */
#define ANA_STFLUX
#define ANA_SSFLUX
#define ANA_BTFLUX
#define ANA_BSFLUX
/*#define ANA_PSOURCE*/

#define MY25_MIXING	/*Mellor/Yamada Level-2.5 closure*/
#if defined GLS_MIXING || defined MY25_MIXING
# define KANTHA_CLAYSON	/*Kantha and Clayson stability function*/
# define N2S2_HORAVG		/*horizontal smoothing of buoyancy/shear*/
# define RI_SPLINES		/*splines reconstruction for vertical sheer*/
#else
# define ANA_VMIX
#endif

#if defined BIO_FENNEL  || defined ECOSIM || \
    defined NPZD_POWELL || defined NEMURO
# define ANA_BIOLOGY
# define ANA_SPFLUX
# define ANA_BPFLUX
# define ANA_SRFLUX
#endif

#if defined NEMURO
# define HOLLING_GRAZING
# undef  IVLEV_EXPLICIT
#endif

#ifdef BIO_FENNEL
# define CARBON
# define DENITRIFICATION
# define BIO_SEDIMENT
# define DIAGNOSTICS_BIO
#endif

#ifdef PERFECT_RESTART
# undef  AVERAGES
# undef  DIAGNOSTICS_BIO
# undef  DIAGNOSTICS_TS
# undef  DIAGNOSTICS_UV
# define OUT_DOUBLE
#endif

