ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
ARG MODEL_SIZE=large

FROM ${BASE_IMAGE}

# Gunicorn environment variables
ENV GUNICORN_WORKERS=1
ENV GUNICORN_THREADS=2
ENV GUNICORN_PORT=5000

# SAM 2 environment variables
ENV PYTHONUNBUFFERED=1
ENV SAM2_BUILD_CUDA=0
ENV MODEL_SIZE=${MODEL_SIZE}

# Install system requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    build-essential \
    libffi-dev \
    git

RUN pip install --upgrade pip setuptools
WORKDIR /
RUN git clone https://github.com/facebookresearch/sam2.git
WORKDIR /sam2
RUN pip install -e .
RUN mkdir /workspace/checkpoints
RUN cp sam2/configs/sam2.1/sam2.1_hiera_l.yaml /workspace/checkpoints/
WORKDIR /workspace
COPY ./ /workspace
RUN pip install -r requirements.txt

# https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/issues/69#issuecomment-1826764707
RUN rm /opt/conda/bin/ffmpeg && ln -s /bin/ffmpeg /opt/conda/bin/ffmpeg

# Make app directory. This directory will host all files required for the
# backend and SAM 2 inference files.

# Download SAM 2.1 checkpoints
# ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_tiny.pt
# ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_small.pt
# ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_base_plus.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt /workspace/checkpoints/sam2.1_hiera_large.pt

# https://pythonspeed.com/articles/gunicorn-in-docker/
# CMD gunicorn --worker-tmp-dir /dev/shm \
#     --worker-class gthread app:app \
#     --log-level info \
#     --access-logfile /dev/stdout \
#     --log-file /dev/stderr \
#     --workers ${GUNICORN_WORKERS} \
#     --threads ${GUNICORN_THREADS} \
#     --bind 0.0.0.0:${GUNICORN_PORT} \
#     --timeout 60
CMD streamlit run main.py