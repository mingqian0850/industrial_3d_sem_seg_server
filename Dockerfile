FROM pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    HF_HOME=/tmp/huggingface

ARG POINTCEPT_REPO=https://github.com/mingqian0850/Pointcept.git
ARG POINTCEPT_COMMIT=d7272257058337969292865e0892b91d2f362b6a

# Pin the model implementation used for training. The checkpoint config is
# downloaded separately from Hugging Face at container startup.
RUN git clone --filter=blob:none "${POINTCEPT_REPO}" /opt/pointcept \
    && git -C /opt/pointcept checkout "${POINTCEPT_COMMIT}" \
    && rm -rf /opt/pointcept/.git

WORKDIR /opt/server

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY tests/ tests/
COPY docker/entrypoint.sh /usr/local/bin/in3d-entrypoint
RUN chmod +x /usr/local/bin/in3d-entrypoint \
    && mkdir -p /models/ptv3-industrial-23cls \
    && chmod a+rwX /models/ptv3-industrial-23cls
RUN python -m unittest discover -s tests -v

ENV PYTHONPATH=/opt/pointcept:/opt/server \
    MODEL_DIR=/models/ptv3-industrial-23cls \
    HF_MODEL_REPO=min99ian/ptv3-industrial-23cls \
    MODEL_CONFIG=/models/ptv3-industrial-23cls/config.py \
    MODEL_WEIGHT=/models/ptv3-industrial-23cls/ptv3-industrial-23cls.pth \
    MODEL_DEVICE=cuda \
    HOST_GPU_INDEX=0 \
    MAX_VALID_POINTS=350000 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"

ENTRYPOINT ["in3d-entrypoint"]
