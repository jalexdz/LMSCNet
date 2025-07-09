import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import re

def collect_labels(dataset_dir):
    label_files = glob.glob(os.path.join(dataset_dir, '*/*/dense/*.labels'))
    label_names = set()

    for f in label_files:
        with open(f, 'r') as file:
            names = {line.strip() for line in file.readlines()}
            label_names.update(names)

    standard_names = ["floor", "wall", "table", "chair", 
                      "cabinet", "glovebox", "computer", "trashcan", 
                      "barrel", "container", "person", "shelf", "unlabeled"]
    
    df = pd.DataFrame({'raw_name': sorted(label_names)})

    def match_standard_name(raw):
        raw = raw.lower().replace('_', ' ').replace('-', ' ')
        for name in standard_names:
            pattern = r'\b' + re.escape(name.lower()) + r'\b'
            if re.search(pattern, raw):
                return name
        return 'unlabeled'

    df['standard_name'] = df['raw_name'].apply(match_standard_name) 
    df['remap_id'] = ''         # e.g., 2

    df.to_excel('label_mapping_template.xlsx', index=False)
    print("Wrote: label_mapping_template.xlsx")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 read_raw_labels.py <dataset_dir>")
        sys.exit(1)

    collect_labels(sys.argv[1])