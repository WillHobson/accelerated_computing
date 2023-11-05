import numpy as np
cimport numpy as np
from libc.math cimport cos, exp
from libc.math cimport pow

cpdef double one_energy(np.ndarray[double, ndim=2] arr, int ix, int iy, int nmax):
    cdef double en, ang
    cdef int ixp, ixm, iyp, iym
    
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #

    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*pow(cos(ang),2))
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*pow(cos(ang),2))
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*pow(cos(ang),2))
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*pow(cos(ang),2))
    return en


cpdef double all_energy(np.ndarray[double, ndim=2] arr, int nmax):
    cdef double enall
    enall = 0.0
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall

cpdef double get_order(np.ndarray[double, ndim=2] arr, int nmax):
    cdef np.ndarray[double, ndim=2] Qab = np.zeros((3,3))
    cdef np.ndarray[double, ndim=2] delta = np.eye(3,3)
    cdef np.ndarray[double, ndim=3] lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    cdef np.ndarray[double, ndim=1] eigenvalues
    cdef np.ndarray[double, ndim=2] eigenvectors
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




cpdef double MC_step(np.ndarray[double, ndim=2] arr, double Ts, int nmax, seed=None):
    cdef double scale=0.1+Ts
    cdef double accept=0
    rng = np.random.default_rng(seed)
    cdef np.ndarray[np.int_t, ndim=2] xran=rng.integers(low=0,high=nmax,size=(nmax,nmax))
    rng = np.random.default_rng(seed)
    cdef np.ndarray[np.int_t, ndim=2] yran=rng.integers(low=0,high=nmax,size=(nmax,nmax))
    rng = np.random.default_rng(seed)
    cdef np.ndarray[double, ndim=2] aran = rng.normal(scale=scale, size=(nmax,nmax))

    cdef int ix,iy
    cdef double ang, en0, en1, boltz
    
    
    for i in range(nmax):
        for j in range(nmax):
            ix = xran[i,j]
            iy = yran[i,j]
            ang = aran[i,j]
            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)
            if en1<=en0:
                accept += 1
            else:
                boltz = exp( -(en1 - en0) / Ts )
                rng = np.random.default_rng(seed)
                if boltz >= rng.uniform(0.0,1.0):
                    accept += 1
                else:
                    arr[ix,iy] -= ang
    return accept/float(nmax*nmax)







































