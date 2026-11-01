# Hermes Core Runtime: Python only; optional tool runtimes are installed separately.
FROM python:3.14-slim AS dependencies
WORKDIR /opt/hermes
COPY pm/ pm/
COPY hermes_platform/ hermes_platform/
COPY hermes_cli/__init__.py hermes_cli/runtime_state.py hermes_cli/
COPY hermes_constants*.py hermes_yaml.py hermes_dir_acl.py ./
COPY pyproject.toml uv.lock README.md ./
RUN python -m pm.build_env --source /opt/hermes --python /usr/local/bin/python \
    --out /opt/hermes/.venv --no-install-project --extra runtime --sealed

FROM python:3.14-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git ripgrep tini && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10000 hermes && useradd --uid 10000 --gid hermes --create-home hermes \
    && install -d -o hermes -g hermes -m 0700 /var/lib/hermes /workspace
WORKDIR /opt/hermes
COPY --from=dependencies /opt/hermes/.venv /opt/hermes/.venv
COPY . .
RUN printf '#!/bin/sh\nexec /opt/hermes/.venv/bin/python -m hermes_core "$@"\n' > /usr/local/bin/hermes \
    && chmod 0755 /usr/local/bin/hermes
ENV PATH="/opt/hermes/.venv/bin:${PATH}" \
    HERMES_HOME=/var/lib/hermes \
    HERMES_RUNTIME_DIR=/var/lib/hermes/runtime \
    PYTHONUNBUFFERED=1
USER hermes
VOLUME ["/var/lib/hermes", "/workspace"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD ["python", "-m", "hermes_core", "health"]
ENTRYPOINT ["/usr/bin/tini", "--", "python", "-m", "hermes_core"]
CMD ["run"]
