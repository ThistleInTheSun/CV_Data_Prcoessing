import ctypes

import numpy as np

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
    dets = np.array(lis, dtype=np.float64)
    dets = Convert2DToCArray(dets)
    m, n = len(dets), len(dets[0])
    m = ctypes.c_int(m)
    n = ctypes.c_int(n)
    # dets = dets.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    print(dets)
    for x in dets:
        for j in x:
            print(j)
    print(len(dets), len(dets[0]))

    # ctypes.POINTER(ctypes.c_double), 
    OCSort.test.argtypes = [ctypes.c_int, ctypes.c_int]
    OCSort.test.restype = None
    OCSort.test(m, n)