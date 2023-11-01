import numpy as np
import time
cimport numpy as np
from cython.parallel cimport prange
cimport openmp
cimport cython
from libc.math cimport cos, exp, pow
ctypedef np.float64_t dtype_t

@cython.nogil
@cython.wraparound(False)
@cython.boundscheck(False)
cdef dtype_t one_energy(double[:,::1] arr, Py_ssize_t ix, 
                        Py_ssize_t iy,Py_ssize_t nmax) nogil:
    cdef dtype_t en, ang
    cdef Py_ssize_t ixp, ixm, iyp,iym
    
    
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


@cython.nogil
@cython.wraparound(False)
@cython.boundscheck(False)
cdef dtype_t all_energy(double[:,::1] arr, Py_ssize_t nmax, Py_ssize_t threads):
    cdef dtype_t enall
    enall = 0.0
    cdef Py_ssize_t i,j,num_threads
    for i in prange(nmax, nogil=True, num_threads=threads):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall

@cython.nogil
@cython.wraparound(False)
@cython.boundscheck(False)
cdef double get_order(double[:,::1] arr, Py_ssize_t nmax,Py_ssize_t threads):
    cdef double[:,::1] Qab = np.zeros((3,3))
    cdef double[:,::1] delta = np.eye(3,3)
    cdef double[:,:,::1] lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    cdef np.ndarray[double, ndim=1] eigenvalues
    cdef np.ndarray[double, ndim=2] eigenvectors
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    cdef Py_ssize_t a,b,i,j, num_threads
    for a in range(3):
        for b in range(3):
            for i in prange(nmax, nogil=True, num_threads=threads):
            #for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = np.asarray(Qab)/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()




cpdef double MC_step(double[:,::1] arr, double Ts,int nmax):

    cdef dtype_t scale=0.1+Ts
    cdef Py_ssize_t accept=0
    cdef Py_ssize_t[:,::1] xran=np.random.randint(0,high=nmax, size=(nmax,nmax))
    cdef Py_ssize_t[:,::1] yran=np.random.randint(0,high=nmax, size=(nmax,nmax))
    
    cdef double[:,::1] aran = np.random.normal(scale=scale, size=(nmax,nmax))
    cdef Py_ssize_t ix,iy
    cdef dtype_t ang, en0, en1, boltz
    
    
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
                boltz = exp( -(en1 - en0) / Ts )

                if boltz >= np.random.uniform(0.0,1.0):
                    accept += 1
                else:
                    arr[ix,iy] -= ang
    return accept/(nmax*nmax)

cdef double[:,::1] initdat(Py_ssize_t nmax):
    cdef double[:,::1] arr
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr

cpdef main(str program, int nsteps, int nmax, double temp, int pflag,Py_ssize_t threads):
    # Create and initialise lattice
    cdef double[:,::1]  lattice
    cdef double[::1]  energy = np.zeros(nsteps+1,dtype=float)
    cdef double[::1] ratio = np.zeros(nsteps+1,dtype=float)
    cdef double[::1] order = np.zeros(nsteps+1,dtype=float)
    cdef dtype_t initial, final, runtime

    lattice = initdat(nmax)
    # Set initial values in arrays
    energy[0] = all_energy(lattice,nmax, threads)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,nmax, threads)

    # Begin doing and timing some MC steps.
    initial = openmp.omp_get_wtime()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,nmax)
        energy[it] = all_energy(lattice,nmax, threads)
        order[it] = get_order(lattice,nmax, threads)
    final = openmp.omp_get_wtime()
    runtime = final-initial
    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s, Threads: {:d}".format(program, nmax,nsteps,temp,order[nsteps-1],runtime, threads))
    # Plot final frame of lattice and generate output file