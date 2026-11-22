# A stable Neuron stack for Trn1 and Inf2 (SDK 2.28, frozen)

Trn1 (Trainium1) and Inf2 (Inferentia2) use the same NeuronCore-v2 generation and lost support together, so one
stack serves both. The 2.28 package list is identical for the two, except that `neuronx_distributed_training` is
Trn1-only (not included here). Compiled artifacts are chip-specific: compile with `--target inf2` for Inf2 and
`--target trn1` for Trn1. `verify.py` compiles for both and runs on whichever chip is present.

Starting with Neuron SDK 2.29 (April 2026), NxD Inference, vllm-neuron and NKI 0.3+ kernels support only Trainium2
and newer. The release notes tell Trn1/Inf2 users to stay on SDK 2.28. Later releases (2.30 to 2.33.1, Nov 2026)
do not bring Trn1/Inf2 back, and AWS offers no LTS line. **2.28 is the last stable version for Trn1/Inf2.**
This directory freezes it so that it installs the same way every time and cannot drift to 2.29+.

## Pick one

| Use | What | Pin |
|---|---|---|
| Run or serve on trn1/inf2 with the least effort | AWS DLAMI *Deep Learning AMI Neuron PyTorch Inference vLLM 0.13 (Ubuntu 24.04) 20260227* | AMI IDs in `versions.env`; AWS deprecates them on **2028-02-27** (copy the AMI into your account to keep it) |
| vLLM serving in a container | `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx@sha256:b92b70aa52a9d8cd2a025260f2bc00de57f89ab95728be292ce7e16b2280a4c1` | Pin by **digest**, not tag (see pitfall 3) |
| Compile, NKI simulation, CI, any x86 box or Apple silicon | `Dockerfile` here (Ubuntu 24.04, CPU torch, no vLLM) | Base image digest + `requirements.lock` |
| Plain Ubuntu 22.04/24.04 host (trn1, inf2 or CPU) | `sudo ./setup-host.sh` | apt packages held, venv locked, `pip.conf` constraint |

All four have the same Neuron components:

| Component | Version |
|---|---|
| neuronx-cc | 2.23.6484.0 |
| torch / torch-xla / torch-neuronx | 2.9.0 / 2.9.0 / 2.9.0.2.12.22436 |
| libneuronxla | 2.2.15515.0 |
| neuronx-distributed / -inference | 0.17.26814 / 0.8.16251 |
| nki | 0.2.0 |
| transformers | 4.57.6 |
| vllm / vllm-neuron | 0.13.0 / 0.4.1 (DLAMI and container only) |
| runtime-lib / collectives / tools / dkms (apt) | 2.30.51.0 / 2.30.59.0 / 2.28.23.0 / 2.26.5.0 |

## Use

```bash
# Container (compile / simulate anywhere)
git clone https://github.com/Tar-ive/neuron-2.28-trn1-inf2 && cd neuron-2.28-trn1-inf2
docker build --platform linux/amd64 -t neuron-2.28-trn1-inf2 .
docker run --rm --platform linux/amd64 neuron-2.28-trn1-inf2 python /stack/verify.py
# On a trn1/inf2 host, pass the devices through (host driver: aws-neuronx-dkms 2.26.5.0):
docker run --rm --device=/dev/neuron0 neuron-2.28-trn1-inf2 python /stack/verify.py

# Bare Ubuntu host
sudo ./setup-host.sh && . /opt/neuron-2.28/bin/activate && python verify.py

# 2.28 DLAMI: nothing to install, just check
/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/python verify.py
```

`verify.py` checks:
- the exact pip and apt versions;
- compiling a model with `torch_neuronx.trace` for both `trn1` and `inf2`;
- an NKI kernel in the CPU simulator;
- the NxD Inference model-builder imports;
- on a Neuron host, running the traced model and the NKI kernel on the device and comparing them with CPU.

## Pitfalls this stack handles (each one hit in practice)

1. **`pip install` silently upgrades to 2.29+.** The Neuron pip index serves the newest wheels by default: `nki 0.3+`,
   `neuronx-cc 2.24+`, `neuronx-distributed-inference 0.9+`, none of which run Trn1/Inf2 kernels. `setup-host.sh` writes
   a `pip.conf` with `constraint = constraints.txt`, so `pip install nki==0.3.0` fails to resolve and
   `pip install -U transformers` stays at 4.57.6. Both were tested.
2. **Never pip-install NxD Inference on top of a DLAMI venv.** The torch/torch-xla wheels get mismatched and import
   fails with undefined `_XLAC` symbols.
3. **The AWS "sdk2.28.0" containers are labeled with 2.27-era versions.** Their build args say neuronx-cc 2.22,
   NxDI 0.7 and NKI 0.1, but the Dockerfile installs `neuronx-cc>=2.0` and an unpinned NxDI from the Neuron index.
   The pushed image (built 2026-02-26) actually contains the 2.28 set; we listed its layers. Rebuilding that
   Dockerfile today would install 2.33 packages, so pin the published image by digest.
4. **`libarchive.so.13` is missing.** neuronx-cc's backend (`walrus_driver`) needs it and no pip package declares it.
   Compiles then fail with `neuronx-cc failed with 70`.
5. **`libpython3.12.so.1.0` is missing.** torch-xla and libneuronxla link against the shared libpython, which a bare
   host doesn't have.
6. **`FileNotFoundError: libneuronpjrt-path`.** torch_neuronx runs helpers from the venv's `bin/`. Activate the
   venv, or put its `bin/` on `PATH`.
7. **`Unsupported Platform` when importing NxD Inference off a Neuron instance.** Set `NEURON_PLATFORM_TARGET_OVERRIDE=trn1` (or
   `inf2`) on compile boxes and in CI.
8. **Compiled artifacts load only on the exact same SDK.** Compile on the same 2.28 stack you run on.
9. **The tools are 2.28 too.** Version 2.28's `neuron-explorer view` lacks the flags the current VS Code extension
   needs. Use a newer explorer for viewing profiles only, never for compiling.

## Changing the stack
Edit `versions.env` and `constraints.txt` together, run `./resolve.sh` to regenerate `requirements.lock`, rebuild,
and rerun `verify.py` on CPU and on a device.

## Status
- **CI:** every push builds the image and runs `verify.py` on GitHub's x86 runners, and installs with `setup-host.sh`
  on a fresh Ubuntu 24.04 runner.
- **Verified on CPU (2026-11-22).** Both the Docker image and `setup-host.sh` on bare Ubuntu 24.04 pass versions,
  trn1 and inf2 compiles, the NKI simulator and the NxDI imports, under amd64 emulation on Apple silicon.
- **Device check pending.** The same components ran Llama-3.1-8B benchmarks and profiles on trn1.2xlarge with the DLAMI (Oct-Nov
  2026), but `verify.py`'s device step has not yet run on hardware.

## License
Apache-2.0 for the files in this repository. The Neuron SDK, PyTorch and the other packages installed have their own
licenses. This project is not affiliated with or endorsed by AWS.
