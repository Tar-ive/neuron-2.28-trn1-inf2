# Trn1/Inf2 Neuron 2.28 compile/dev stack (compiler, torch-neuronx, NxD Inference, NKI; CPU torch), fully pinned.
# For vLLM serving use the AWS 2.28 vLLM container pinned by digest instead (README). Builds on any x86_64 host (or Apple silicon under emulation).
# Compiling (neuronx-cc, torch_neuronx.trace, NxDI --compile-only) and the NKI simulator work without a Neuron device.
# To run on a trn1/inf2 host, the host needs the matching driver (aws-neuronx-dkms 2.26.5.0, see setup-host.sh) and:
#   docker run --device=/dev/neuron0 ... (one --device per chip)
#
#   docker build --platform linux/amd64 -t neuron-2.28-trn1-inf2 .
#   docker run --rm --platform linux/amd64 neuron-2.28-trn1-inf2 python /stack/verify.py
FROM --platform=linux/amd64 ubuntu:24.04@sha256:9a04f13b3f80ef056361e30868593b1cfd2279fe19e4081cb93a90bf56819d6a
ARG DEBIAN_FRONTEND=noninteractive
COPY versions.env constraints.txt requirements.lock verify.py /stack/

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg git \
      python3.12 python3.12-venv python3.12-dev build-essential libarchive13t64 \
 && . /stack/versions.env \
 && curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | gpg --dearmor -o /usr/share/keyrings/neuron.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com noble main" > /etc/apt/sources.list.d/neuron.list \
 && apt-get update && apt-get install -y --no-install-recommends \
      aws-neuronx-runtime-lib=$AWS_NEURONX_RUNTIME_LIB aws-neuronx-collectives=$AWS_NEURONX_COLLECTIVES \
      aws-neuronx-tools=$AWS_NEURONX_TOOLS \
 && apt-mark hold aws-neuronx-runtime-lib aws-neuronx-collectives aws-neuronx-tools \
 && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/neuron-2.28 PATH=/opt/neuron-2.28/bin:/opt/aws/neuron/bin:$PATH \
    PIP_EXTRA_INDEX_URL=https://pip.repos.neuron.amazonaws.com PIP_CONSTRAINT=/stack/constraints.txt
RUN python3.12 -m venv $VIRTUAL_ENV \
 && pip install --no-cache-dir -U "pip==25.2" \
 && pip install --no-cache-dir --no-deps -r /stack/requirements.lock \
 && pip check
WORKDIR /work
