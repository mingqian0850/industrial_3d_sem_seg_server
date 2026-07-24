# Deployment image for the DITR industrial 3D semantic segmentation server.
# Based on the official Pointcept image (public on Docker Hub, pulled
# automatically) so all compiled deps (spconv, flash-attn, pointops) match
# the checkpoint. The DITR additions below mirror the training image
# (ws_Pointcept/Dockerfile.ditr).
FROM pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel

RUN pip install --no-cache-dir \
    sharedarray \
    imageio \
    albumentations==1.4.21 \
    "yapf<0.40.2"

ARG DITR_REPO=https://github.com/mingqian0850/ditr.git
ARG DITR_COMMIT=f66d1dadb82e97ded7750567a043e4783305b01c

# Model code (Pointcept fork with DITR), pinned to the training commit.
RUN git clone "${DITR_REPO}" /opt/ditr \
    && git -C /opt/ditr checkout "${DITR_COMMIT}" \
    && rm -rf /opt/ditr/.git
ENV PYTHONPATH=/opt/ditr

# Writable caches regardless of the runtime user.
ENV HF_HOME=/opt/cache/huggingface \
    TORCH_HOME=/opt/cache/torch \
    XDG_CACHE_HOME=/opt/cache

# Bake the frozen DINOv2-small weights into the image so the container
# does not need internet access at startup (only used by timm).
RUN python -c "import timm; timm.create_model(\
'vit_small_patch14_reg4_dinov2', pretrained=True, dynamic_img_size=True, \
num_classes=0, global_pool='')" \
    && chmod -R a+rX /opt/cache

WORKDIR /opt/server
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY tests/ tests/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /models/ditr-industrial-aligned-23cls \
    && chmod a+rwX /models/ditr-industrial-aligned-23cls \
    && python -m unittest discover -s tests -v

# Where the checkpoint + config live at runtime; if the directory is empty
# the entrypoint downloads HF_MODEL_REPO into it first.
ENV MODEL_DIR=/models/ditr-industrial-aligned-23cls \
    HF_MODEL_REPO=min99ian/ditr-industrial-aligned-23cls \
    MODEL_CONFIG=/models/ditr-industrial-aligned-23cls/config.py \
    MODEL_WEIGHT=/models/ditr-industrial-aligned-23cls/ditr-industrial-aligned-23cls.pth \
    MODEL_DEVICE=cuda \
    HOST_GPU_INDEX=0 \
    MAX_VALID_POINTS=350000 \
    PORT=8000

EXPOSE 8000
ENTRYPOINT ["entrypoint.sh"]
