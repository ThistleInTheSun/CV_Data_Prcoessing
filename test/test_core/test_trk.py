import ctypes

OCSort = ctypes.CDLL("cvdata/core/trk/ocsort.so")


if __name__ == "__main__":
    OCSort.update()