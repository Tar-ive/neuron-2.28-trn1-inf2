#!/usr/bin/env python3
"""Checks that this environment is the pinned Trn1/Inf2 Neuron 2.28 stack, and that it works.

  1. versions: every pip package in versions.env (and the apt packages, when dpkg is present) matches exactly;
  2. compile: torch_neuronx.trace compiles a small model for trn1 and for inf2 (no device needed);
  3. NKI: a kernel runs in the NKI 0.2 CPU simulator and matches NumPy;
  4. NxD Inference: the model-builder modules import;
  5. device (only if /dev/neuron0 exists): the traced model and the NKI kernel run on the NeuronCore and match CPU.

Exit status 0 only if every applicable check passes. Prints one line per check, plus a JSON summary with --json.
Works in the Dockerfile image, in the 2.28 DLAMI venv and in the AWS 2.28 vLLM container.
"""
import glob
import importlib.metadata as md
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PIP = {"NEURONX_CC": "neuronx-cc", "TORCH_NEURONX": "torch-neuronx", "LIBNEURONXLA": "libneuronxla",
       "NEURONX_DISTRIBUTED": "neuronx-distributed", "NEURONX_DISTRIBUTED_INFERENCE": "neuronx-distributed-inference",
       "NKI": "nki", "TORCH": "torch", "TORCH_XLA": "torch-xla", "TORCHVISION": "torchvision",
       "TRANSFORMERS": "transformers"}
OPTIONAL_PIP = {"VLLM": "vllm"}  # only in the serving container / DLAMI
APT = {"AWS_NEURONX_RUNTIME_LIB": "aws-neuronx-runtime-lib", "AWS_NEURONX_COLLECTIVES": "aws-neuronx-collectives",
       "AWS_NEURONX_TOOLS": "aws-neuronx-tools", "AWS_NEURONX_DKMS": "aws-neuronx-dkms"}

results = []


def check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        status = "skip" if isinstance(detail, str) and detail.startswith("skip") else "ok"
    except Exception as e:  # noqa: BLE001
        status, detail = "FAIL", f"{type(e).__name__}: {e}"
    results.append(dict(check=name, status=status, detail=detail, seconds=round(time.time() - t0, 1)))
    print(f"[{status:>4}] {name}: {detail}", flush=True)


def load_versions():
    v = {}
    for line in open(os.path.join(HERE, "versions.env")):
        m = re.match(r'^([A-Z0-9_]+)=("?)([^"#]*?)\2\s*(#.*)?$', line.strip())
        if m:
            v[m.group(1)] = m.group(3).strip()
    return v


def versions():
    want = load_versions()
    bad = []
    for key, pkg in {**PIP, **OPTIONAL_PIP}.items():
        try:
            have = md.version(pkg)
        except md.PackageNotFoundError:
            if key in OPTIONAL_PIP:
                continue
            bad.append(f"{pkg} missing (want {want[key]})")
            continue
        exp = want[key]
        ok = have == exp or (pkg in ("torch", "torchvision") and have.split("+")[0] == exp)
        if not ok:
            bad.append(f"{pkg} {have} != {exp}")
    have_dpkg = subprocess.run(["which", "dpkg-query"], capture_output=True).returncode == 0
    if have_dpkg:
        for key, pkg in APT.items():
            r = subprocess.run(["dpkg-query", "-W", "-f=${Status} ${Version}", pkg], capture_output=True, text=True)
            installed = r.returncode == 0 and "ok installed" in r.stdout  # "install ok installed" or "hold ok installed"
            r.stdout = r.stdout.split()[-1] if installed else ""
            if not installed:
                if pkg == "aws-neuronx-dkms" and not os.path.exists("/dev/neuron0"):
                    continue  # driver lives on the host, not in containers
                if pkg in ("aws-neuronx-runtime-lib", "aws-neuronx-collectives", "aws-neuronx-tools") and not os.path.exists("/opt/aws/neuron"):
                    continue
                bad.append(f"{pkg} not installed")
            elif r.stdout.strip() != want[key]:
                bad.append(f"{pkg} {r.stdout.strip()} != {want[key]}")
    if bad:
        raise AssertionError("; ".join(bad))
    return f"{len(PIP)} pip pins match Neuron {want['NEURON_SDK']}" + (" (+apt)" if have_dpkg else "")


def tiny_model():
    import torch

    class M(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.a, s.b = torch.nn.Linear(256, 512), torch.nn.Linear(512, 128)

        def forward(s, x):
            return torch.softmax(s.b(torch.nn.functional.gelu(s.a(x))), -1)

    torch.manual_seed(0)
    return M().eval(), torch.randn(4, 256)


TRACED = {}


def compile_for(target):
    def run():
        import torch_neuronx
        m, x = tiny_model()
        with tempfile.TemporaryDirectory() as d:
            traced = torch_neuronx.trace(m, x, compiler_workdir=d, compiler_args=["--target", target])
            neffs = glob.glob(os.path.join(d, "**", "*.neff"), recursive=True)
            size = sum(os.path.getsize(n) for n in neffs)
        TRACED[target] = traced
        return f"traced for {target}, {len(neffs)} NEFF, {size / 1024:.0f} KiB"
    return run


def nki_sim():
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl

    @nki.jit
    def add_kernel(a, b):
        out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
        nl.store(out, value=nl.add(nl.load(a), nl.load(b)))
        return out

    a, b = np.random.rand(128, 512).astype(np.float32), np.random.rand(128, 512).astype(np.float32)
    got = nki.simulate_kernel(add_kernel, a, b)
    assert np.array_equal(got, a + b), "simulated add differs from NumPy"
    globals()["_nki_kernel"] = (add_kernel, a, b)
    return "simulate_kernel add matches NumPy"


def nxdi_import():
    # NxDI reads the instance type at import; on a non-Neuron host (compile box, laptop, CI) say which chip to target.
    if not os.path.exists("/dev/neuron0"):
        os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    import neuronx_distributed_inference as nxdi
    from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig  # noqa: F401
    from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM  # noqa: F401
    return f"NxD Inference {md.version('neuronx-distributed-inference')} model builder imports ({os.path.dirname(nxdi.__file__)})"


def device_type():
    try:
        out = subprocess.run(["neuron-ls", "--json-output"], capture_output=True, text=True, timeout=60).stdout
        info = json.loads(out)
        return info[0].get("instance_type") or info[0].get("device_type") or "neuron"
    except Exception:  # noqa: BLE001
        return "neuron"


def on_device():
    if not os.path.exists("/dev/neuron0"):
        return "skip (no /dev/neuron0)"
    import numpy as np
    import torch
    import torch_neuronx  # noqa: F401
    m, x = tiny_model()
    itype = device_type()
    target = "inf2" if "inf2" in str(itype) else "trn1"
    if target not in TRACED:
        compile_for(target)()
    got = TRACED[target](x)
    err = (got - m(x)).abs().max().item()
    assert err < 1e-2, f"device vs CPU max error {err}"
    kernel, a, b = globals()["_nki_kernel"]
    out = kernel(a, b)  # nki.jit with NumPy inputs runs in baremetal mode on the NeuronCore
    assert np.allclose(np.asarray(out), a + b), "NKI on device differs"
    return f"{itype}: traced model max |err| {err:.1e} vs CPU; NKI add on device OK"


if __name__ == "__main__":
    # torch_neuronx shells out to helpers in the venv's bin/ (libneuronpjrt-path), so a venv python run without
    # activation fails with FileNotFoundError. Put the interpreter's bin/ on PATH, as `activate` would.
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    check("versions", versions)
    check("compile trn1", compile_for("trn1"))
    check("compile inf2", compile_for("inf2"))
    check("nki simulator", nki_sim)
    check("nxd inference", nxdi_import)
    check("device", on_device)
    if "--json" in sys.argv:
        print(json.dumps(results, indent=1))
    sys.exit(0 if all(r["status"] != "FAIL" for r in results) else 1)
