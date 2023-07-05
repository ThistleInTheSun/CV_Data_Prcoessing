import ctypes

import numpy as np
from numpy.ctypeslib import ndpointer
OCSort = ctypes.CDLL("cvdata/core/trk/ocsort.so")



def Convert1DToCArray(TYPE, ary):
    arow = TYPE(*ary.tolist())
    return arow


def Convert2DToCArray(ary):
    ROW = ctypes.c_double * len(ary[0])
    rows = []
    for i in range(len(ary)):
        rows.append(Convert1DToCArray(ROW, ary[i]))
    MATRIX = ROW * len(ary)
    return MATRIX(*rows)


if __name__ == "__main__":
    lis = [
        [1, 2, 3, 4, 0.3], 
        [3, 4, 5, 6, 0.8],
    ]
    
    # dets = np.array(lis, dtype=np.float64)
    # dets = Convert2DToCArray(dets)

    # print(dets)
    # res = OCSort.update(dets)
    # print(res)


    
    # lis = [1, 2, 3, 4, 0.3]
    dets_array = np.array(lis, dtype=np.uint64)
    # dets = Convert2DToCArray(dets_array)



    m, n = dets_array.shape
    # m = ctypes.c_int(m)
    # n = ctypes.c_int(n)
    dets = dets_array.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
    # print(len(dets), len(dets[0]))

    # # ctypes.POINTER(ctypes.c_double), 
    # np.ctypeslib.ndpointer
    # OCSort.test.argtypes = (ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int, ctypes.c_int)
    # OCSort.test.restype = None
    # print(dets, m, n)
    # dets1 = ctypes.cast(dets, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
    # print("dets1:", dets1)
    # dets2 = dets_array.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
    # print("dets2:", dets2)
    # OCSort.test(dets2, m, n)

    # OCSort.test.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int, ctypes.c_int]
    # OCSort.test.restype = None
    # print(dets)
    # OCSort.test(*dets, m, n)


    c_array = (ctypes.POINTER(ctypes.c_double) * m)()
    for i in range(m):    
        c_array[i] = dets_array[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    OCSort.test(c_array, m, n)

    double** c_array = new double*[m]; 
    for (int i = 0; i < m; i++) { 
        c_array[i] = new double[n]; 
    }