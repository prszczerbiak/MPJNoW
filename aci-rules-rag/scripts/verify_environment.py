"""Sprawdza środowisko testami offline i zapisuje wersje zainstalowanych pakietów."""
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import atomic_write, write_json


def main():
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Python 3.11.x is required; target is 3.11.11")
    subprocess.run([sys.executable, "-m", "pip", "check"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "--junitxml=data/manifests/tests.xml"], cwd=ROOT, check=True)
    packages = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    import torch
    header = f"# Środowisko sprawdzone testami integracyjnymi offline\n# Python: {platform.python_version()}\n# Platforma: {platform.platform()}\n# Torch CUDA: {torch.version.cuda}\n"
    if "+cpu" in torch.__version__:
        header += "# Środowisko testowe CPU. Przed eksperymentem zapisz osobny, zweryfikowany plik zależności GPU.\n--extra-index-url https://download.pytorch.org/whl/cpu\n"
    atomic_write(ROOT / "requirements.lock", header + "".join(f"{name}=={version}\n" for name, version in packages))
    write_json(ROOT / "data/manifests/environment_validation.json",
               {"python_target": "3.11.11", "python_actual": platform.python_version(),
                "platform": platform.platform(), "cuda": torch.version.cuda,
                "packages": dict(packages), "test_scope": "offline synthetic and dependency integration",
                "clinical_experiment_executed": False, "full_generator_inference_verified": False})


if __name__ == "__main__":
    main()
