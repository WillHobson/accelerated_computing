import numpy as np
cimport numpy as np
from libc.math cimport cos
from libc.math cimport pow

cpdef one_energy(np.ndarray[double, ndim=2] arr, int ix, int iy, int nmax):
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




cpdef MC_step(np.ndarray[double, ndim=2] arr, double Ts,int nmax):
    #
    # Pre-compute some random numbers.  This is faster than
    # using lots of individual calls.  "scale" sets the width
    # of the distribution for the angle changes - increases
    # with temperature.
    cdef double scale=0.1+Ts
    cdef int accept=0
    cdef np.ndarray[np.int_t, ndim=2] xran=np.random.randint(0,high=nmax, size=(nmax,nmax))
    cdef np.ndarray[np.int_t, ndim=2] yran=np.random.randint(0,high=nmax, size=(nmax,nmax))
    
    cdef np.ndarray[double, ndim=2] aran = np.random.normal(scale=scale, size=(nmax,nmax))
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
            # Now apply the Monte Carlo test - compare
            # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = np.exp( -(en1 - en0) / Ts )

                if boltz >= np.random.uniform(0.0,1.0):
                    accept += 1
                else:
                    arr[ix,iy] -= ang
    return accept/(nmax*nmax)




















