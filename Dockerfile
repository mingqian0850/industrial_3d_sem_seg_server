FROM pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/tmp/cache \
    HF_HOME=/tmp/huggingface \
    TORCH_HOME=/tmp/torch \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    TRITON_CACHE_DIR=/tmp/triton

ARG VOLT_REPO=https://github.com/mingqian0850/Volt.git
ARG VOLT_COMMIT=089cc38d8b32e7c695dd939057f787f3c60dd35e

# Pin the exact Volt implementation used to train the industrial checkpoint.
# The lighter Pointcept CUDA 12.4 runtime has been forward-tested against the
# EMA checkpoint and avoids requiring Conda or Python on the deployment host.
RUN git clone --filter=blob:none "${VOLT_REPO}" /opt/volt \
    && git -C /opt/volt checkout "${VOLT_COMMIT}" \
    && rm -rf /opt/volt/.git

WORKDIR /opt/server

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY tests/ tests/
COPY docker/entrypoint.sh /usr/local/bin/in3d-entrypoint
RUN chmod +x /usr/local/bin/in3d-entrypoint \
    && mkdir -p /models/volt-industrial-23cls \
    && chmod a+rwX /models/volt-industrial-23cls

ENV PYTHONPATH=/opt/volt:/opt/server
RUN python -c "import flash_attn, torch; from pointcept.models import build_model; print(torch.__version__, flash_attn.__version__)" \
    && python -m unittest discover -s tests -v

ENV MODEL_DIR=/models/volt-industrial-23cls \
    HF_MODEL_REPO=min99ian/volt-industrial-23cls \
    HF_REVISION=1020052f8b4968235c727414e70103d43e78169b \
    MODEL_CONFIG=/models/volt-industrial-23cls/config.py \
    MODEL_WEIGHT=/models/volt-industrial-23cls/volt-industrial-23cls.pth \
    MODEL_CONFIG_SHA256=6d5385d38c5d6060eacd3d226124c86804b8cee7e129664b4c365f3bd6e9ed36 \
    MODEL_WEIGHT_SHA256=65ba22f299847f2c4356a59f1f110160f865c7e9d82816d9e980161fe515c198 \
    MODEL_DEVICE=cuda \
    HOST_GPU_INDEX=0 \
    MAX_VALID_POINTS=250000 \
    MAX_VOXEL_POINTS=60000 \
    MAX_VOLT_TOKENS=6000 \
    VOXEL_SAMPLE_SEED=0 \
    REQUEST_QUEUE_TIMEOUT_SECONDS=30 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"

ENTRYPOINT ["in3d-entrypoint"]
