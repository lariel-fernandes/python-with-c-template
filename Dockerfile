ARG OS_VERSION
ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-devel-${OS_VERSION} AS build-stage

# Install Python
ARG PYTHON_VERSION
RUN apt-get update && \
    apt-get install -y build-essential pkg-config software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv && \
    rm -rf /var/lib/apt/lists/*

# Install build-time OS dependencies
RUN --mount=type=bind,source=build/packages-build.txt,target=packages-build.txt \
    apt-get update && \
    grep -v '^#' packages-build.txt | grep -v '^$' | xargs apt-get install -y && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=cache,mode=0755,target=/root/.cache/uv \
    --mount=type=bind,source=build/requirements.txt,target=requirements.txt \
    uv pip install --no-deps -r requirements.txt --python=/usr/bin/python${PYTHON_VERSION}

# Build project wheel from sources
COPY src/ /build/src/
COPY setup.py pyproject.toml MANIFEST.in /build/
WORKDIR /build
RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=cache,mode=0755,target=/root/.cache/uv \
    uv build --wheel --python=/usr/bin/python${PYTHON_VERSION}

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${OS_VERSION}

# Install Python
ARG PYTHON_VERSION
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv && \
    rm -rf /var/lib/apt/lists/*

# Install runtime OS dependencies
RUN --mount=type=bind,source=build/packages-runtime.txt,target=packages-runtime.txt \
    apt-get update && \
    grep -v '^#' packages-runtime.txt | grep -v '^$' | xargs -r apt-get install -y && \
    rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from build stage
COPY --from=build-stage /usr/local/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages

# Install project wheel from build stage
RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=cache,mode=0755,target=/root/.cache/uv \
    --mount=type=bind,from=build-stage,source=/build/dist,target=/dist \
    uv pip install --python=/usr/bin/python${PYTHON_VERSION} --no-deps /dist/*.whl
