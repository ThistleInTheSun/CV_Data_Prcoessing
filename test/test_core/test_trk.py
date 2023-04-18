import ctypes

OCSort = ctypes.cdll.LoadLibrary("cvdata/core/trk/ocsort.so")
# from cvdata.core.trk import OCSort


if __name__ == "__main__":
    # print(OCSort)
    # p = OCSort.OCSort()
    # print(p)

    res_int = OCSort.display()