# =============================================================================
# NDVI Extraction Pipeline
# Multi-stage build (ตาม pattern ของพี่):
#   Stage 1 builder : build GDAL จาก source + install deps ใน venv
#   Stage 2 runtime : copy venv + GDAL libs เท่านั้น → image เล็กสะอาด
# =============================================================================

# -------- Stage 1: Builder --------
FROM python:3.11-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="Durian"
LABEL description="Sentinel-2 NDVI point-extraction pipeline"

# System deps สำหรับ build GDAL + geospatial libs
RUN apt-get update && apt-get install -y \
    curl git build-essential cmake ninja-build pkg-config \
    python3 python3-pip python3-venv python3-dev \
    libsqlite3-dev libcurl4-openssl-dev libssl-dev \
    libtiff-dev libjpeg-dev libpng-dev libgeos-dev \
    libproj-dev libspatialite-dev libwebp-dev libxml2-dev \
    libopenjp2-7-dev zlib1g-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Build GDAL 3.8.5 จาก source (ให้ตรงกับ requirements.txt)
WORKDIR /tmp
RUN curl -L -o gdal.tar.gz https://github.com/OSGeo/gdal/archive/refs/tags/v3.8.5.tar.gz && \
    tar -xzf gdal.tar.gz && \
    cd gdal-3.8.5 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && make install && ldconfig

ENV CPLUS_INCLUDE_PATH=/usr/local/include
ENV C_INCLUDE_PATH=/usr/local/include

# ติดตั้ง uv (เร็วกว่า pip มาก)
RUN pip install --upgrade pip && pip install uv

# Install Python deps ใน venv
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv && \
    uv pip install --python=/opt/venv/bin/python setuptools numpy==1.26.4 && \
    uv pip install --python=/opt/venv/bin/python --no-build-isolation GDAL==3.8.5 && \
    uv pip install --python=/opt/venv/bin/python --no-cache-dir -r /tmp/requirements.txt && \
    /opt/venv/bin/python -c "from osgeo import gdal, gdal_array; print('✅ GDAL OK:', gdal.VersionInfo(), '| gdal_array OK')" && \
    rm -rf /root/.cache /tmp/* && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + && \
    find /opt/venv -type f -name "*.pyc" -delete

# -------- Stage 2: Runtime --------
FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive

# Runtime libs เท่านั้น (ไม่ต้อง build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 libcurl4 libssl3 \
    libtiff-dev libjpeg62-turbo libpng16-16 \
    libgeos-dev libproj-dev \
    libspatialite-dev \
    libwebp7 libxml2 \
    libopenjp2-7 zlib1g \
    python-is-python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy venv + GDAL จาก builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/include /usr/local/include
RUN ldconfig

ENV PATH="/opt/venv/bin:/usr/local/bin:/usr/bin:$PATH"
ENV PYTHONPATH="/opt/venv/lib/python3.11/site-packages"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GDAL_CACHEMAX=512 \
    OGR_SQLITE_SYNCHRONOUS=OFF

WORKDIR /workspaces
COPY ndvi_pipeline.py /workspaces/ndvi_pipeline.py

VOLUME ["/fs2", "/output"]

CMD ["python3", "/workspaces/ndvi_pipeline.py"]