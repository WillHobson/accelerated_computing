import numpy as np
cimport numpy as np
from libc.math cimport cos
cpdef one_energy(np.ndarray[double, ndim=2] arr, int ix, int iy, int nmax):
    cdef double en, ang
    cdef int ixp, ixm, iyp, iym
    
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #

    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    return en


cpdef all_energy(np.ndarray[double, ndim=2] arr, int nmax):
    cdef double enall
    enall = 0.0
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall