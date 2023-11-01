import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from par_cy import main

#=======================================================================


#=======================================================================

#=======================================================================

#=======================================================================

#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 6:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        THREADS = int(sys.argv[5])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, THREADS)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADS>".format(sys.argv[0]))
#=======================================================================
