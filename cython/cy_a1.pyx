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

cpdef get_order(np.ndarray[double, ndim=2] arr, int nmax):
    cdef np.ndarray[double, ndim=2] Qab = np.zeros((3,3))
    cdef np.ndarray[double, ndim=2] delta = np.eye(3,3)
    cdef np.ndarray[double, ndim=3] lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()