FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime AS gpu
RUN python -m pip install --no-cache-dir "numpy<2" tqdm matplotlib scikit-learn
ENV PAI_DEVICE=cuda CUBLAS_WORKSPACE_CONFIG=:4096:8
WORKDIR /results
CMD ["python", "-u", "solution.py"]

FROM mambaorg/micromamba:latest AS checker
COPY env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
RUN pip install pyarmor==6.7.4 torch==2.0.1 scikit-learn==1.3.0
WORKDIR /code
ADD * /code/
ADD pytransform /code/pytransform
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/code/pytransform
WORKDIR /code
CMD python -u checker_client.py
