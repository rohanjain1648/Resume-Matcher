import os

# keep BLAS/torch thread buffers small so the matcher also runs on low-RAM laptops
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
