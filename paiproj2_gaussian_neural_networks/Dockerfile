FROM mambaorg/micromamba:latest@sha256:e1476182cd40eb1f7c0cfae4adec9bfd78f3d7f05c61368306abfdc8939c1198 AS checker
COPY env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python -m pip install --no-cache-dir pyarmor==6.7.4 && \
    python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.0.1

WORKDIR /code
COPY pytransform /code/pytransform
COPY checker_client.py map_weights.pt *.npz test_ys.bytes /code/
ENV LD_LIBRARY_PATH=/code/pytransform
ENV OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
COPY util.py solution.py /code/
CMD ["python", "-u", "checker_client.py"]
