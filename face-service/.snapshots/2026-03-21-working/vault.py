import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

VAULT_PATH = Path("out/vault.json")

def get_file_hash(path: str) -> str:
    """Calculate MD5 hash of a file to detect changes."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

class FaceVault:
    def __init__(self, vault_path: Path = VAULT_PATH):
        self.vault_path = vault_path
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self):
        if self.vault_path.exists():
            try:
                with open(self.vault_path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def save(self):
        self.vault_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.vault_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_image_data(self, image_path: str, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Return cached faces if image hash and parameters match."""
        abs_path = str(Path(image_path).absolute())
        if abs_path not in self.data:
            return None
        
        current_hash = get_file_hash(image_path)
        cached = self.data[abs_path]
        
        # Check hash AND params (jitters, etc)
        if cached.get("hash") == current_hash and cached.get("params") == params:
            return cached.get("faces")
        
        return None

    def add_image_data(self, image_path: str, faces: List[Dict[str, Any]], params: Dict[str, Any]):
        """Add image results to vault."""
        abs_path = str(Path(image_path).absolute())
        current_hash = get_file_hash(image_path)
        self.data[abs_path] = {
            "hash": current_hash,
            "params": params,
            "timestamp": os.path.getmtime(image_path),
            "faces": faces
        }
        self.save()
