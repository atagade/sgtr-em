import argparse
import json
import pandas as pd

def flip_labels(input_file, output_file):
    with open(input_file, 'r') as f:
        data = pd.read_json(f, lines=True)

    for index, item in data.iterrows():
        item["messages"][2]["content"] = f"{3 - int(item["messages"][2]["content"])}"

    data.to_json(output_file, orient='records', lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flip labels in SGTR JSON file.")
    parser.add_argument("-i", "--input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("-o", "--output_file", type=str, help="Path to the output JSON file.")
    args = parser.parse_args()

    flip_labels(args.input_file, args.output_file)