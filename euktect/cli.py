# euktect/cli.py
"""CLI entry points for the euktect conda package."""
import sys
import os


def predict_main():
    """Entry point for the euktect-predict command."""
    # Ensure the package root is in sys.path for predict.py to find src.*
    pkg_root = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(pkg_root)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from predict import main
    main()


def refine_main():
    """Entry point for the euktect-refine command."""
    pkg_root = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(pkg_root)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from refine import main
    main()
