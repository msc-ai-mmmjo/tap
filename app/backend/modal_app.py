"""
Modal deployment wrapper for the Hydra inference backend.

Two Modal entry points live here:

- :func:`download_weights` (one-off) -- populates the ``tap-olmo-weights``
  Volume with OLMo-2-7B weights and a warm HF cache for ModernBERT-NLI.
  Run once per weights bump via ``modal run app/backend/modal_app.py::download_weights``.
- :class:`HydraBackend` (always-on, GPU-backed) -- preloads Hydra in the
  ``@modal.enter()`` hook (so the FastAPI lifespan can skip the load) and
  serves :data:`app.backend.server.app` as an ASGI app.

The container image is built from the project's ``pixi.lock`` so the
runtime CUDA/torch/flash-attn stack matches local dev exactly. The
``CONDA_OVERRIDE_CUDA=12.4`` env var is required because the build
container has no GPU and pixi's ``__cuda`` virtual package check would
otherwise fail; runtime containers get a real GPU from Modal.
"""

import os

import modal

# Paths and identifiers.
WEIGHTS_MOUNT = "/weights"
MODEL_DIR = f"{WEIGHTS_MOUNT}/olmo2-7b"
HF_CACHE_DIR = f"{WEIGHTS_MOUNT}/hf-cache"
HF_MODEL_ID = "allenai/OLMo-2-1124-7B-Instruct"

PROJECT_DIR = "/app"
PIXI_ENV_BIN = f"{PROJECT_DIR}/.pixi/envs/cuda/bin"

# Reuse prod's pixi cuda environment inside the container: one source of truth
# (pixi.lock), same conda-forge torch + flash-attn that pixi resolves locally,
# no wheel/CUDA juggling. add_local_dir ships the project tree so pixi can
# install the editable msc-ai-group-project package and build the env in place.
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "curl", "ca-certificates")
    .run_commands(
        "curl -fsSL https://pixi.sh/install.sh | bash",
        "ln -s /root/.pixi/bin/pixi /usr/local/bin/pixi",
    )
    .workdir(PROJECT_DIR)
    .add_local_dir(
        ".",
        PROJECT_DIR,
        copy=True,
        ignore=[
            ".git/**",
            ".pixi/**",
            ".venv/**",
            "app/frontend/**",
            "node_modules/**",
            "**/__pycache__/**",
            "**/*.pyc",
            ".env",
            ".env.*",
            "docs/**",
            "olmo_tap/data/**",
            "olmo_tap/weights/prod_outdated/**",
        ],
    )
    # The build container has no GPU, so pixi's __cuda virtual package check
    # fails without an override; runtime containers get a real GPU from Modal.
    .run_commands(
        f"cd {PROJECT_DIR} && CONDA_OVERRIDE_CUDA=12.4 pixi install --environment cuda --locked"
    )
    .env(
        {
            "PATH": f"{PIXI_ENV_BIN}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .run_commands("python -c \"import nltk; nltk.download('punkt_tab', quiet=True)\"")
)

weights_vol = modal.Volume.from_name("tap-olmo-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

app = modal.App("tap-backend", image=image)


@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    secrets=[hf_secret],
    timeout=60 * 60,
)
def download_weights() -> None:
    """
    One-off Modal job that populates the weights Volume.

    Downloads OLMo-2-7B from the HF Hub and warms the ModernBERT-NLI cache
    by calling :func:`app.backend.bert_inference.load_bert` inside the
    container. Both downloads happen in a single function so one
    ``weights_vol.commit()`` at the end makes the lot available to all
    subsequent containers.

    Invoke with::

        modal run app/backend/modal_app.py::download_weights
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        HF_MODEL_ID,
        local_dir=MODEL_DIR,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"OLMo weights downloaded to {MODEL_DIR}")

    # Point bert_inference at the Volume before importing it; constants.py
    # reads HF_CACHE_DIR at module load time.
    os.environ["HF_CACHE_DIR"] = HF_CACHE_DIR
    from app.backend.bert_inference import load_bert

    model, tokenizer = load_bert(device="cpu")
    if model is None or tokenizer is None:
        raise RuntimeError("BERT load failed during cache population")
    print(f"BERT cache populated at {HF_CACHE_DIR}")

    weights_vol.commit()
    print("Volume committed")


@app.cls(
    gpu=["H100", "A100-40GB"],
    volumes={WEIGHTS_MOUNT: weights_vol},
    secrets=[hf_secret],
    scaledown_window=300,
    max_containers=3,
)
class HydraBackend:
    """
    Always-on Modal class that serves the FastAPI app on a GPU container.

    GPU preference is ``H100`` first then ``A100-40GB`` (see
    :data:`project_modal_gpu_choice` for the benchmarking rationale).
    ``scaledown_window=300`` keeps a warm container around for 5 minutes
    after the last request, masking cold-starts during interactive use.
    """

    @modal.enter()
    def preload(self) -> None:
        """
        Modal lifecycle hook that runs once per fresh container.

        Loads Hydra into :data:`app.backend.server._models` *before* the
        ASGI lifespan starts, so the lifespan detects the preloaded model
        and skips the duplicate load (~30s saving). The env vars are set
        before the imports because :mod:`olmo_tap.constants` and
        :mod:`app.backend.constants` read them at module load time.
        """
        # Set env before imports: olmo_tap.constants and app.backend.constants
        # read WEIGHTS_DIR and HF_CACHE_DIR at module load time.
        os.environ.setdefault("OLMO_WEIGHTS_DIR", MODEL_DIR)
        os.environ.setdefault("HF_CACHE_DIR", HF_CACHE_DIR)
        os.environ.setdefault("DEVICE", "cuda")

        from app.backend import server
        from app.backend.hydra_inference import load_hydra

        model, tokenizer = load_hydra(device="cuda")
        if model is None or tokenizer is None:
            raise RuntimeError("Hydra preload failed; refusing to start container")
        server._models["hydra"] = model
        server._tokenizers["hydra"] = tokenizer
        server._device = "cuda"

    @modal.asgi_app()
    def fastapi_app(self):
        """
        Return the FastAPI ASGI app for Modal to serve.

        The lifespan sees the Hydra model preloaded by :meth:`preload` and
        skips it, then loads BERT from the Volume-backed HF cache.

        :returns: The :class:`fastapi.FastAPI` instance defined in
            :mod:`app.backend.server`.
        """
        from app.backend.server import app as fastapi_app

        return fastapi_app
