# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Deterministic numerics: one BLAS thread, fixed hash seed. Without these two
# the runs are reproducible only up to thread scheduling, and the numbers in
# the tables move in the fourth decimal between machines.
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MPLBACKEND=Agg \
    RESULTS_DIR=/work/results

# pyshtools ships manylinux wheels, so no Fortran toolchain is needed. The
# build tools below are a fallback for architectures without a prebuilt wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran libopenblas-dev libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /work/requirements.txt

# pyteda comes from PyPI, pinned to an exact version in requirements.txt. The
# version actually used is recorded in every manifest, so a run can always be
# traced to the release it was produced with.
ENV PYTHONPATH=/work:/work/experiments

COPY cvloc /work/cvloc
COPY experiments /work/experiments
COPY scripts /work/scripts
COPY tests /work/tests
COPY notebooks /work/notebooks
COPY paper /work/paper
COPY README.md Makefile setup.py /work/

RUN mkdir -p /work/results && chmod +x /work/scripts/*.sh

# Fails the build if the method does not register into pyteda, if the
# objective can see the truth, or if a parameterization stops being nested.
# A broken image then never reaches the cluster.
RUN python -m pytest tests -q

# Progress is written to stdout, line buffered and timestamped, so a remote
# run can be followed with:
#     docker logs -f cvloc-fpa-enkf
#     docker compose logs -f experiments
# and a persistent copy is left in results/run_<scale>.log
CMD ["bash", "/work/scripts/run_all.sh"]
