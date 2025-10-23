"""
Utility functions for running end-to-end pipelines.
"""

import subprocess
import sys
import os


def run_script(script_path, args=None, description=None, capture_output=False, project_root=None):
    """
    Run a Python script with optional arguments.

    Args:
        script_path: Path to the script relative to project root
        args: List of command-line arguments
        description: Description to print before running
        capture_output: If True, capture and return stdout as a string
        project_root: Project root directory (defaults to current working directory)

    Returns:
        str or None: Captured stdout if capture_output=True, otherwise None
    """
    if project_root is None:
        project_root = os.getcwd()

    if description:
        print(f"\n{'='*80}")
        print(f"  {description}")
        print(f"{'='*80}\n")

    full_path = os.path.join(project_root, script_path)
    cmd = [sys.executable, full_path]
    if args:
        cmd.extend(args)

    print(f"Running: {' '.join(cmd)}\n")

    if capture_output:
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        # Print the output so the user can see it
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    else:
        result = subprocess.run(cmd, cwd=project_root)

    if result.returncode != 0:
        print(f"\n❌ Error: Script failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n✓ Completed successfully\n")

    if capture_output:
        return result.stdout
    return None
