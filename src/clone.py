import pandas as pd
import subprocess
import os, sys
import zipfile
import argparse
from pandarallel import pandarallel
import multiprocessing
from tqdm import tqdm

REPOS_DIR = "data/repos"

EXCLUSION_LIST = [
    "lanl-ansi/MINLPLib.jl",
    "Mehrnoom/Cryptocurrency-Pump-Dump",
]

def clone(repo: str, force: bool = False):
  """
    Clones a GitHub repository into a local directory.

    Args:
        repo (str): The repository to clone, in the format "owner/repo_name".
        force (bool, optional): If `True`, re-clones the repository even if it already exists. Defaults to `False`.
    """
    if not force and os.path.exists(os.path.join(REPOS_DIR, repo)):
        return 
    proc = subprocess.run(
        ["git", "clone", "--depth", "1", f"https://github.com/{repo}", os.path.join(REPOS_DIR, repo)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if proc.returncode != 0:
        print(f"Failed to clone {repo}", file=sys.stderr)
        print(f"Error message was:", file=sys.stderr)
        print(proc.stderr.decode(), file=sys.stderr)

def clean(path: str):
    """
    Cleans a specified directory by removing known unnecessary directories, keeping only Julia files and removing empty directories.

    Args:
        path (str): The directory path to clean.

    Notes:
        - Deletes the directories `.git/`, `.github/`, and `.vscode/` if present.
        - Removes all files not ending with `.jl`.
        - Removes empty directories after file deletions.
        - Creates a `cleaned` file in the directory as a marker of completion. 
          Useful for when the cloning and cleaning process is interrupted.

    Raises:
        Prints an error to standard error if a file or directory cannot be removed.
    """
    # Check if the given path is a directory
    if not os.path.isdir(path):
        print(f"The path {path} is not a valid directory.")
        return

    # Remove known directories that are not needed
    to_remove = [".git/", ".github/", ".vscode/"]
    for item in to_remove:
        try:
            os.system(f"rm -rf {path}/{item}")
        except Exception as e:
            print(f"Could not remove: {path}/{item}. Error: {e}", file=sys.stderr)

    # Walk through the directory tree to delete non-Julia files
    for root, dirs, files in os.walk(path, topdown=False):
        for filename in files:
            # Construct the full file path
            file_path = os.path.join(root, filename)
            
            # Check if the file does not end with '.jl'
            if not filename.endswith('.jl'):
                try:
                    # Remove the file
                    os.remove(file_path)
                    # print(f"Removed file: {file_path}")
                except Exception as e:
                    print(f"Could not remove file: {file_path}. Error: {e}", file=sys.stderr)
        
        # After removing files, remove empty directories
        for dirname in dirs:
            dir_path = os.path.join(root, dirname)
            try:
                # Attempt to remove the directory if it is empty
                os.rmdir(dir_path)
                # print(f"Removed empty directory: {dir_path}")
            except OSError:
                # Directory is not empty
                pass

    # touch file to indicate that cleaning is done
    open(os.path.join(path, "cleaned"), "w").close()

def clone_and_clean(repo: str):
    clone(repo)
    clean(os.path.join(REPOS_DIR, repo))

def already_cleaned(repo: str) -> bool:
    return os.path.exists(os.path.join(REPOS_DIR, repo, "cleaned"))

def zipped_repos(
        zip_path="data/repos.zip", 
        csv_with_repos="data/julia.csv.gz",
        keep_repos_dir=False,
        force=False, 
        verbose=False,
    ) -> zipfile.ZipFile:
    """
    Clones, cleans, and compresses a set of repositories specified in a CSV file into a zip archive.

    Args:
        zip_path (str): The path where the zip file will be saved. Defaults to "data/repos.zip".
        csv_with_repos (str): Path to the CSV file containing repository names. Defaults to "data/julia.csv.gz".
        keep_repos_dir (bool): If `False`, removes the repositories directory after zipping. Defaults to `False`.
        force (bool): If `True`, recreates the zip file even if it already exists. Defaults to `False`.
        verbose (bool): If `True`, outputs detailed process information. Defaults to `False`.

    Returns:
        zipfile.ZipFile: The created zip file object.

    Notes:
        - Uses an exclusion list to filter repositories before cloning.
        - Clones and cleans repositories before adding them to the zip file.
        - Deletes intermediate repository files unless `keep_repos_dir` is set to `True`.

    Raises:
        Prints errors to standard error for issues encountered during file or directory operations.
    """
    if not force and os.path.exists(zip_path):
        if verbose: print(f"Zip file {zip_path} already exists, loading that one.")
        return zipfile.ZipFile(zip_path, "r")
    
    pandarallel.initialize(nb_workers=min(50, multiprocessing.cpu_count()-1), progress_bar=True, verbose=2 if verbose else 0)

    if verbose: print(f"Zip file {zip_path} does not exist (or force was set to True), creating it.")

    if verbose: print(f"Reading CSV file {csv_with_repos}")
    df = pd.read_csv(args.csv)

    if verbose: print("Filtering already cleaned repositories")
    n_repos_before = df.shape[0]
    df = df[~df.name.isin(EXCLUSION_LIST)]
    n_repos_after = df.shape[0]
    if verbose: print(f"Filtered {n_repos_before - n_repos_after} repositories from the exclusion list")

    if verbose: print("Filtering already cleaned repositories")
    n_repos_before = df.shape[0]
    to_clone = df[~df.name.parallel_apply(already_cleaned)]
    n_repos_after = to_clone.shape[0]
    if verbose: print(f"Filtered {n_repos_before - n_repos_after} already cleaned repositories")

    if verbose: print("Cloning and cleaning repositories")
    if to_clone.shape[0] > 0:
        to_clone.name.parallel_apply(clone_and_clean)

    if verbose: print("Zipping repositories")
    file_count = sum(len(files) for file, _, files in os.walk(REPOS_DIR) if file != "cleaned")
    final_zip_path = "data/repos.zip"
    with zipfile.ZipFile(final_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        with tqdm(total=file_count, desc="Zipping files", unit="file") as pbar:
            for root, _, files in os.walk(REPOS_DIR):
                for file in files:
                    try:
                        if file == "cleaned":
                            continue
                        zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), REPOS_DIR))
                    except Exception:
                        pass
                    pbar.update(1)

    if verbose: print(f"All repositories have been zipped into {final_zip_path}")

    if not keep_repos_dir:
        if verbose: print(f"Removing {REPOS_DIR}")
        os.system(f"rm -rf {REPOS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clone julia repositories and clean them")
    parser.add_argument("-c", "--csv", default="data/julia.csv.gz", help="Path to the csv file containing the repositories. Default is data/julia.csv.gz")
    parser.add_argument("-z", "--zip", default="data/repos.zip", help="Path to the zip file to store the repositories. Default is data/repos.zip")
    parser.add_argument("-k", "--keep", help="Keep the repositories directory after zipping", action="store_true")
    parser.add_argument("-f", "--force", help="Force the operation", action="store_true")
    parser.add_argument("--force-clone", help="Force the cloning of the repositories", action="store_true")
    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
    args = parser.parse_args()

    zipped_repos(args.zip, args.csv, args.keep, args.force, args.verbose)

