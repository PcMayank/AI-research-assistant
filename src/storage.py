"""
storage.py — Persistent storage via HuggingFace Dataset repo.

On startup  → pull latest vectorstore from HF dataset to local disk
On update   → push local vectorstore back to HF dataset

This gives HuggingFace Spaces persistent storage for free.
"""
from __future__ import annotations
import os
import shutil
import zipfile
from pathlib import Path
from src.logger import logger


def _get_hf_config():
    token = os.getenv("HF_TOKEN", "")
    repo = os.getenv("HF_DATASET_REPO", "")
    return token, repo


def is_persistent_storage_enabled() -> bool:
    token, repo = _get_hf_config()
    return bool(token and repo)


def pull_vectorstore(local_dir: str) -> bool:
    """
    Download vectorstore from HF dataset repo to local_dir.
    Returns True if successfully restored, False if nothing to restore.
    """
    if not is_persistent_storage_enabled():
        logger.info("Persistent storage not configured — skipping pull")
        return False

    token, repo = _get_hf_config()

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

        # Check if vectorstore.zip exists in the repo
        try:
            files = list(list_repo_files(repo, repo_type="dataset", token=token))
        except RepositoryNotFoundError:
            logger.warning(f"HF dataset repo '{repo}' not found — skipping pull")
            return False

        if "vectorstore.zip" not in files:
            logger.info("No vectorstore.zip in HF dataset — fresh start")
            return False

        logger.info(f"Pulling vectorstore from {repo}...")
        zip_path = hf_hub_download(
            repo_id=repo,
            filename="vectorstore.zip",
            repo_type="dataset",
            token=token,
        )

        # Clear existing local dir and extract
        local_path = Path(local_dir)
        if local_path.exists():
            shutil.rmtree(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(local_path)

        logger.info(f"Vectorstore restored to '{local_dir}'")
        return True

    except Exception as e:
        logger.error(f"Failed to pull vectorstore: {e}")
        return False


def push_vectorstore(local_dir: str) -> bool:
    """
    Upload local vectorstore to HF dataset repo as vectorstore.zip.
    Returns True on success.
    """
    if not is_persistent_storage_enabled():
        logger.info("Persistent storage not configured — skipping push")
        return False

    token, repo = _get_hf_config()
    local_path = Path(local_dir)

    if not local_path.exists():
        logger.warning(f"Local vectorstore dir '{local_dir}' does not exist — skipping push")
        return False

    try:
        from huggingface_hub import HfApi

        # Zip the vectorstore directory
        zip_path = Path("/tmp/vectorstore.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in local_path.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(local_path))

        # Upload to HF dataset repo
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(zip_path),
            path_in_repo="vectorstore.zip",
            repo_id=repo,
            repo_type="dataset",
        )

        logger.info(f"Vectorstore pushed to {repo}")
        zip_path.unlink(missing_ok=True)
        return True

    except Exception as e:
        logger.error(f"Failed to push vectorstore: {e}")
        return False