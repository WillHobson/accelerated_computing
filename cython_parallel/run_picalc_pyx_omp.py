import sys
from picalc_pyx_omp import main

if int(len(sys.argv)) == 3:
    main(int(sys.argv[1]),int(sys.argv[2]))
else:
    print("Usage: {} <ITERATIONS> <THREADS>".format(sys.argv[0]))

