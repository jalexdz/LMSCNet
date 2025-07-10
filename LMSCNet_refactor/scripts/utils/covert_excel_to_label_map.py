import pandas as pd
import yaml
import sys
import os

def convert_excel_to_label_map(excel_path, output_path):
    df = pd.read_excel(excel_path)

    # Drop rows where standard_name is blank
    df = df[df['standard_name'].notnull() & (df['standard_name'] != '')]
    unique_names = sorted(df['standard_name'].unique())
    name_to_id = {name: i + 1 for i, name in enumerate(unique_names)}

    label_map = {'labels': {}, 'color_map': {}}
    default_colors = [
        [0, 0, 0],         # black (unlabeled)
        [0, 0, 255],       # red-ish
        [245, 150, 100],   # brown
        [245, 230, 100],   # yellow
        [250, 80, 100],    # pink
        [150, 60, 30],     # dark brown
        [255, 0, 0],       # blue
    ]

    for i, row in df.iterrows():
        name = str(row['standard_name'])
        raw_name = str(row['raw_name']).strip().lower().replace(' ', '_')
        remap_id = name_to_id[name]

        label_map['labels'][remap_id] = name
        label_map['color_map'][remap_id] = default_colors[remap_id % len(default_colors)]
        label_map.setdefault('raw_to_remap', {})[raw_name] = remap_id

    with open(output_path, 'w') as f:
        yaml.dump(label_map, f)

    print(f"Saved label map to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_label_map.py <input_excel> <output_yaml>")
        sys.exit(1)

    excel_path = sys.argv[1]
    output_yaml = sys.argv[2]

    convert_excel_to_label_map(excel_path, output_yaml)
