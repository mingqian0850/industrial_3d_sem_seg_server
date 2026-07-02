# Deployment image for the DITR industrial 3D semantic segmentation server.
# Based on the training image so all compiled deps (spconv, flash-attn,
# pointops) are guaranteed to match the checkpoint.
FROM pointcept/pointcept:ditr-pytorch2.5.0-cuda12.4

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
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && mkdir -p /models && chmod a+rwX /models

# Where the checkpoint + config live at runtime; if the directory is empty
# the entrypoint downloads HF_MODEL_REPO into it first.
ENV MODEL_DIR=/models/ditr-industrial \
    HF_MODEL_REPO=min99ian/ditr-industrial \
    PORT=8000

EXPOSE 8000
ENTRYPOINT ["entrypoint.sh"]
