# Present so that `pytest tests` works from a clean clone, not only
# `python -m pytest`. Without it the repository root is absent from sys.path
# and `from src.config import ...` fails on someone else's machine.
