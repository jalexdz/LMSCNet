import os
import glob
import sys

def process_single_label_file(label_file, label_map_file):
    # Load label map (index -> raw_name)
    label_map = {}
    with open(label_map_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            label_map[int(parts[0])] = ' '.join(parts[1:])

    # Load label indices
    with open(label_file, 'r') as f:
        label_indices = [int(line.strip()) for line in f]

    # Map each index to raw_name
    raw_names = [label_map.get(idx, 'unlabeled') for idx in label_indices]

    # Overwrite labels.txt with raw names
    with open(label_file, 'w') as f:
        for name in raw_names:
            f.write(f"{name}\n")

    print(f"Rewrote {label_file} with raw names.")

def rewrite_all_labels_with_raw_names(dataset_dir):
    label_files = glob.glob(os.path.join(dataset_dir, '*/*/dense/*.labels'))

    for label_file in label_files:
        label_map_file = os.path.join(os.path.dirname(label_file), 'label_map.txt')
        if os.path.exists(label_map_file):
            process_single_label_file(label_file, label_map_file)
        else:
            print(f"Missing label_map.txt for {label_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rewrite_labels_with_raw_names.py <dataset_dir>")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    rewrite_all_labels_with_raw_names(dataset_dir)
