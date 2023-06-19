import argparse
import csv
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",
                        type=str,
                        required=True,
                        help='File listing the whole training dataset')
    parser.add_argument("--val_csv",
                        type=str,
                        required=True,
                        help='Validation dataset')
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help='Directory to save output files')
    parser.add_argument("--fractions",
                        type=float,
                        nargs='+',
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    return parser.parse_args()

def read_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
        return header, rows

def save_csv(header, rows, path):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

def create_partial_datasets(train_path, val_path, output_dir, fractions):
    header, training_dataset = read_csv(train_path)
    for fraction in fractions:
        num_examples = int(len(training_dataset) * fraction)
        output_path = os.path.join(output_dir, f'train-{fraction}.csv')
        save_csv(header, training_dataset[:num_examples], output_path)

    shutil.copy(val_path, os.path.join(output_dir, 'val.csv'))

if __name__ == '__main__':
    args = parse_args()
    create_partial_datasets(args.train_csv, args.val_csv, args.output_dir,
        args.fractions)
