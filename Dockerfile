FROM registry.access.redhat.com/ubi8/ubi-minimal:latest as builder

RUN microdnf update -y && \
    microdnf install -y \
        git python39-pip && \
    pip3 install --upgrade --no-cache-dir pip && \
    microdnf clean all

RUN python3 -m venv /opt/caikit/

ENV VIRTUAL_ENV=/opt/caikit
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY dist/caikit_nlp*.whl /tmp/
RUN pip install --no-cache /tmp/caikit_nlp*.whl && rm /tmp/caikit_nlp*.whl


FROM registry.access.redhat.com/ubi8/ubi-minimal:latest as deploy

RUN microdnf update -y && \
    microdnf install -y \
        shadow-utils python39 && \
    microdnf clean all

COPY --from=builder /opt/caikit /opt/caikit
COPY LICENSE /opt/caikit/
COPY README.md /opt/caikit/

RUN groupadd --system caikit --gid 1001 && \
    adduser --system --uid 1001 --gid 0 --groups caikit \
    --home-dir /caikit --shell /sbin/nologin \
    --comment "Caikit User" caikit

ENV VIRTUAL_ENV=/opt/caikit
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

USER caikit

ENV RUNTIME_LIBRARY=caikit_nlp
# Optional: use `CONFIG_FILES` and the /caikit/ volume to explicitly provide a configuration file and models
# ENV CONFIG_FILES=/caikit/caikit.yml
VOLUME ["/caikit/"]
WORKDIR /caikit

CMD ["python"]
