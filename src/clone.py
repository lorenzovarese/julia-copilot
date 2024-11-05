import pandas as pd
import subprocess
import os, sys
import zipfile
import argparse
from pandarallel import pandarallel
import multiprocessing
from tqdm import tqdm

pandarallel.initialize(nb_workers=min(50, multiprocessing.cpu_count()-1), progress_bar=True)

REPOS_DIR = "data/repos"

EXCLUSION_LIST = [
    "lanl-ansi/MINLPLib.jl",
    "Mehrnoom/Cryptocurrency-Pump-Dump",
]

def clone(repo: str, force: bool = False):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clone julia repositories and clean them")
    parser.add_argument("-c", "--csv", help="Path to the csv file containing the repositories", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    print("Filtering already cleaned repositories")
    n_repos_before = df.shape[0]
    df = df[~df.name.isin(EXCLUSION_LIST)]
    n_repos_after = df.shape[0]
    print(f"Filtered {n_repos_before - n_repos_after} repositories from the exclusion list")

    to_clone = df[~df.name.parallel_apply(already_cleaned)]
    n_repos_after = to_clone.shape[0]
    print(f"Filtered {n_repos_before - n_repos_after} already cleaned repositories")

    print("Cloning and cleaning repositories")
    if to_clone.shape[0] > 0:
        to_clone.name.parallel_apply(clone_and_clean)

    file_count = sum(len(files) for _, _, files in os.walk(REPOS_DIR))
    # zip the "data/repos" directory
    final_zip_path = "data/repos.zip"
    with zipfile.ZipFile(final_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        with tqdm(total=file_count, desc="Zipping files", unit="file") as pbar:
            for root, dirs, files in os.walk(REPOS_DIR):
                for file in files:
                    try:
                        zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), REPOS_DIR))
                    except Exception as e:
                        pass
                    pbar.update(1)

    print(f"All repositories have been zipped into {final_zip_path}")
