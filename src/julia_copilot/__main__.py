"""Allow ``python -m julia_copilot`` alongside the ``julia-copilot`` script."""

from julia_copilot.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
