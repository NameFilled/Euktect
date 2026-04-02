#!/usr/bin/env python3
"""Install flash_attn_cpu stub as flash_attn in current Python environment."""
import os
import sys
import shutil
import site


def main():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flash_attn_cpu")
    if not os.path.isdir(src):
        print(f"ERROR: flash_attn_cpu directory not found at {src}", file=sys.stderr)
        sys.exit(1)

    # Try writable site-packages first
    candidates = site.getsitepackages() if hasattr(site, 'getsitepackages') else []
    user_site = site.getusersitepackages() if hasattr(site, 'getusersitepackages') else None

    target_dir = None
    for sp in candidates:
        if os.access(sp, os.W_OK):
            target_dir = sp
            break
    if target_dir is None and user_site:
        os.makedirs(user_site, exist_ok=True)
        target_dir = user_site

    if target_dir is None:
        print("ERROR: No writable site-packages found.", file=sys.stderr)
        sys.exit(1)

    dest = os.path.join(target_dir, "flash_attn")
    if os.path.exists(dest):
        print(f"Removing existing flash_attn at {dest} ...")
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    print(f"Installed flash_attn CPU stub -> {dest}")


if __name__ == "__main__":
    main()
