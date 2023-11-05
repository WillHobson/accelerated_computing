#uncomment the filename you would like to test.

from numba_parallel import all_energy, one_energy, get_order
#from numba_parallel import all_energy, one_energy, get_order

import numpy as np
import pytest

ts=0.5
nmax = 5
expect_energy = -10.060086610264408
expect_order = 0.4052857677309655


arr = np.array([[3.70336867, 1.58168628 ,2.88213064 ,0.89033764 ,0.82916109],
 [0.94006858, 2.5531697,  0.89415859, 5.09274011, 1.99080292],
 [3.98956589, 4.53250906, 0.42724291, 0.72890776, 2.72445064],
 [0.62110402, 0.27112905, 0.50261626, 5.1446963,  1.57099636],
 [5.26174731, 3.49660635, 1.9680378 , 1.00542645 ,2.71437755]])


@pytest.mark.parametrize("arr, nmax, expect",[([arr, nmax, expect_energy])])
def test_all_energy(arr, nmax, expect):
    output = all_energy(arr, nmax)
    output = round(output,5)
    expect = round(expect,5)
    assert output == expect

@pytest.mark.parametrize("arr, nmax, expect",[([arr, nmax, expect_order])])
def test_get_order(arr,nmax, expect):
    output = get_order(arr,nmax)
    output = round(output,5)
    expect = round(expect,5)
    assert output == expect


