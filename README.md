Example for using weno5 scheme on non-uniform grids
1D -> linear advection with a stretched grid

2D -> Isentropic vortex convection in a periodic domain on 
      a wavy grid. Implemented both finite difference and 
      finite volume style residuals to highlight implementation
      differences.

To run:
# finite volume error check
python -i vortex_weno.py --residual fv --execution=errorcheck
# finite difference error check
python -i vortex_weno.py --residual fd --execution=errorcheck
# finite volume vortex transport with contours
python -i vortex_weno.py --residual fv 
# finite difference vortex transport with contours
python -i vortex_weno.py --residual fd 
