import pandas as pd
import subprocess
import os, sys
import argparse
from pandarallel import pandarallel
import multiprocessing
import time

pandarallel.initialize(nb_workers=min(50, multiprocessing.cpu_count()-1), progress_bar=True)

REPOS_DIR = "data/repos"

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

def clone_and_clean(repo: str):
    clone(repo)
    clean(os.path.join(REPOS_DIR, repo))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clone julia repositories and clean them")
    parser.add_argument("-c", "--csv", help="Path to the csv file containing the repositories", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.name.progress_apply(clone_and_clean)
    print("Cloning and cleaning repositories")
    df.name.parallel_apply(clone_and_clean)
