"""Download a pinned, non-executable model snapshot with a strict 2 GB size ceiling."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / "local" / "hf-cache")
os.environ["HF_HUB_DISABLE_XET"] = "1"


def main():
    from huggingface_hub import HfApi, snapshot_download
    model = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    info = HfApi().model_info(model, files_metadata=True)
    files = [s for s in info.siblings if s.rfilename.endswith((".json", ".safetensors", ".txt", ".model"))]
    if any(s.size is None for s in files):
        raise RuntimeError("Cannot establish the download size")
    total = sum(s.size for s in files)
    if total > 2_000_000_000:
        raise RuntimeError("Download exceeds the explicitly authorized 2 GB ceiling")
    target = ROOT / "models" / "qwen2.5-1.5b-4bit"
    snapshot_download(model, revision=info.sha, local_dir=str(target),
                      allow_patterns=[s.rfilename for s in files])
    provenance = {"repository": model, "revision": info.sha, "bytes": total,
                  "files": [s.rfilename for s in files], "license": "apache-2.0"}
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Hub exceptions can contain signed download URLs. Do not print their body.
        print("Model download failed:", type(exc).__name__)
        raise SystemExit(1) from None

