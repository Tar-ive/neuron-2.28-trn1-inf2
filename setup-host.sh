#!/usr/bin/env bash
# Installs the pinned Trn1/Inf2 Neuron 2.28 stack on a plain Ubuntu 22.04/24.04 x86_64 host (trn1, inf2, or a CPU
# compile box) and holds every Neuron package so `apt upgrade` / `pip install -U` cannot move it to 2.29+.
# Not needed on the 2.28 DLAMI (it already has this stack in /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13).
#
#   sudo ./setup-host.sh            # driver (on Neuron hosts) + runtime/tools + venv at /opt/neuron-2.28
#   VENV=$HOME/neuron-2.28 ./setup-host.sh --no-apt     # venv only (compile-only use, no root)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/versions.env"
VENV=${VENV:-/opt/neuron-2.28}
APT=1; [ "${1:-}" = --no-apt ] && APT=0

if [ $APT = 1 ]; then
  . /etc/os-release
  curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | gpg --dearmor --yes -o /usr/share/keyrings/neuron.gpg
  echo "deb [signed-by=/usr/share/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com $VERSION_CODENAME main" > /etc/apt/sources.list.d/neuron.list
  apt-get update
  # libarchive is required by neuronx-cc's backend (walrus_driver) but declared by no pip package.
  LIBARCHIVE=$(apt-cache show libarchive13t64 >/dev/null 2>&1 && echo libarchive13t64 || echo libarchive13)
  # torch-xla / libneuronxla link against the shared libpython, which a bare host lacks.
  LIBPY=$(apt-cache show libpython${PYTHON}t64 >/dev/null 2>&1 && echo libpython${PYTHON}t64 || echo libpython${PYTHON})
  PKGS="aws-neuronx-runtime-lib=$AWS_NEURONX_RUNTIME_LIB aws-neuronx-collectives=$AWS_NEURONX_COLLECTIVES aws-neuronx-tools=$AWS_NEURONX_TOOLS"
  if lspci -n 2>/dev/null | grep -qi "1d0f:" ; then   # Annapurna Labs PCI vendor = Neuron device present
    apt-get install -y "linux-headers-$(uname -r)" || true
    PKGS="$PKGS aws-neuronx-dkms=$AWS_NEURONX_DKMS"
  fi
  apt-get install -y --allow-downgrades python${PYTHON} python${PYTHON}-venv git "$LIBARCHIVE" "$LIBPY" $PKGS
  for p in aws-neuronx-runtime-lib aws-neuronx-collectives aws-neuronx-tools aws-neuronx-dkms; do
    dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" && apt-mark hold "$p"
  done
fi

python${PYTHON} -m venv "$VENV"
"$VENV/bin/pip" install -q --no-cache-dir -U pip==25.2
"$VENV/bin/pip" install --no-cache-dir --no-deps -r "$HERE/requirements.lock"
"$VENV/bin/pip" check
# Every later `pip install` in this venv is constrained to the 2.28 pins.
printf '[global]\nconstraint = %s\nextra-index-url = https://pip.repos.neuron.amazonaws.com\n' "$HERE/constraints.txt" > "$VENV/pip.conf"
echo "installed: $VENV  (activate: . $VENV/bin/activate; check: python $HERE/verify.py)"
