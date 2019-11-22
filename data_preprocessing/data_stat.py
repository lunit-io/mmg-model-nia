import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, default='/storage2/br/original/mmg/mg_nia/',
                    help='root to the data dir')

args, _ = parser.parse_known_args()

root_dir = args.root_dir
categories = ['Cancer', 'Benign', 'Normal', 'Benign_additional']
views = ['LCC', 'LMLO', 'RCC', 'RMLO']
comp_name = 'Uncompressed'

types = [0, 0, 0]
cat2type = {'Cancer': 2, 'Benign': 1, 'Benign_additional': 1, 'Normal': 0}
for category in categories:
    cat_dir = os.path.join(root_dir, comp_name, category)
    case_names = os.listdir(cat_dir)
    for case_name in case_names:
        dcm_files = os.listdir(os.path.join(cat_dir, case_name))
        if len(dcm_files) != 4:
            raise ValueError("Dicom is not 4-view: {}".format(case_name))
        types[cat2type[category]] += 1

print(types)

