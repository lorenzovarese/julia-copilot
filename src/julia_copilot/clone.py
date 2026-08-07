"""Clone GitHub repositories whose primary language is Julia, then strip them.

Cleaning keeps only ``*.jl`` files: the corpus builder never looks at anything
else, and dropping the rest takes the on-disk footprint from tens of gigabytes
to a few hundred megabytes.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from julia_copilot.config import (
    EXCLUDED_REPOS,
    JULIA_CSV,
    REPOS_DIR,
    REPOS_ZIP,
    init_pandarallel,
)

logger = logging.getLogger(__name__)

CLEANED_MARKER = "cleaned"
"""Sentinel file written into a repository once cleaning finished.

Cloning ~10k repositories takes hours and is routinely interrupted; the marker
makes the whole stage resumable.
"""

JUNK_DIRS = (".git", ".github", ".vscode")


def clone(repo: str, repos_dir: Path = REPOS_DIR, *, force: bool = False) -> bool:
    """Shallow-clone ``owner/name`` into ``repos_dir``.

    Args:
        repo: Repository in ``owner/name`` form.
        repos_dir: Destination root.
        force: Re-clone even if the target directory already exists.

    Returns:
        ``True`` if the repository is present on disk afterwards.
    """
    target = Path(repos_dir) / repo
    if not force and target.exists():
        return True

    proc = subprocess.run(
        ["git", "clone", "--depth", "1", f"https://github.com/{repo}", str(target)],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        logger.error("Failed to clone %s: %s", repo, proc.stderr.decode(errors="replace").strip())
        return False
    return True


def clean(path: Path) -> None:
    """Remove everything that is not a Julia source file from ``path``.

    Deletes the ``.git``/``.github``/``.vscode`` directories, every file not
    ending in ``.jl``, and any directory left empty afterwards. Writes a
    ``cleaned`` marker on success.
    """
    path = Path(path)
    if not path.is_dir():
        logger.warning("%s is not a directory, skipping cleaning.", path)
        return

    for junk in JUNK_DIRS:
        shutil.rmtree(path / junk, ignore_errors=True)

    for entry in path.rglob("*"):
        if entry.is_file() and entry.suffix != ".jl":
            with contextlib.suppress(OSError):
                entry.unlink()

    # Deepest first, so that a directory emptied by the pass above can itself go.
    directories = sorted(
        (p for p in path.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for directory in directories:
        with contextlib.suppress(OSError):  # raised when the directory is not empty
            directory.rmdir()

    (path / CLEANED_MARKER).touch()


def is_cleaned(repo: str, repos_dir: Path = REPOS_DIR) -> bool:
    """Whether ``repo`` was already cloned *and* cleaned."""
    return (Path(repos_dir) / repo / CLEANED_MARKER).exists()


def clone_and_clean(repo: str, repos_dir: Path = REPOS_DIR) -> None:
    """Clone ``repo`` and immediately strip it of non-Julia files."""
    if clone(repo, repos_dir):
        clean(Path(repos_dir) / repo)


def _archive(repos_dir: Path, zip_path: Path) -> None:
    """Compress every Julia file under ``repos_dir`` into ``zip_path``."""
    files = [p for p in repos_dir.rglob("*") if p.is_file() and p.name != CLEANED_MARKER]

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for file in tqdm(files, desc="Zipping files", unit="file"):
            try:
                zipf.write(file, file.relative_to(repos_dir))
            except OSError as exc:  # unreadable file, broken symlink, ...
                logger.warning("Could not archive %s: %s", file, exc)


def zipped_repos(
    zip_path: Path = REPOS_ZIP,
    csv_with_repos: Path = JULIA_CSV,
    *,
    keep_repos_dir: bool = False,
    force: bool = False,
    repos_dir: Path = REPOS_DIR,
) -> zipfile.ZipFile:
    """Clone, clean and archive every repository listed in ``csv_with_repos``.

    Args:
        zip_path: Where the archive is written.
        csv_with_repos: CSV with a ``name`` column of ``owner/repo`` entries.
        keep_repos_dir: Keep the working checkout directory after archiving.
        force: Rebuild the archive even if it already exists.
        repos_dir: Working directory for the checkouts.

    Returns:
        The archive, opened for reading.
    """
    zip_path, csv_with_repos, repos_dir = Path(zip_path), Path(csv_with_repos), Path(repos_dir)

    if not force and zip_path.exists():
        logger.info("Archive %s already exists, reusing it.", zip_path)
        return zipfile.ZipFile(zip_path, "r")

    init_pandarallel(progress_bar=True)

    logger.info("Reading repository list from %s", csv_with_repos)
    df = pd.read_csv(csv_with_repos)

    before = len(df)
    df = df[~df.name.isin(EXCLUDED_REPOS)]
    logger.info("Dropped %d repositories from the exclusion list.", before - len(df))

    before = len(df)
    to_clone = df[~df.name.parallel_apply(lambda repo: is_cleaned(repo, repos_dir))]
    logger.info("Skipping %d already-cleaned repositories.", before - len(to_clone))

    if not to_clone.empty:
        logger.info("Cloning and cleaning %d repositories...", len(to_clone))
        to_clone.name.parallel_apply(lambda repo: clone_and_clean(repo, repos_dir))

    logger.info("Archiving repositories into %s", zip_path)
    _archive(repos_dir, zip_path)

    if not keep_repos_dir:
        logger.info("Removing working directory %s", repos_dir)
        shutil.rmtree(repos_dir, ignore_errors=True)

    return zipfile.ZipFile(zip_path, "r")
