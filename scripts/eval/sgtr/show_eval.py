"""
Quick visualization of SGTR detection results.

Usage:
    python scripts/eval/sgtr/visualize_detection.py <filepath>
"""

import json
import argparse
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Path to detection JSON file")
    args = parser.parse_args()

    with open(args.filepath) as f:
        data = json.load(f)

    total = len(data)
    counts = Counter(data.values())

    print(f"Total: {total}")
    for source, count in counts.most_common():
        print(f"  {source}: {count} ({count/total*100:.1f}%)")


if __name__ == "__main__":
    main()
