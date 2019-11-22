import os
from tqdm import tqdm
import argparse
import pydicom as dicom
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, default='/path/to/nia/',
                    help='root to the data dir')

args, _ = parser.parse_known_args()

root_dir = args.root_dir
compression_name = ['Uncompressed', 'Compressed']
categories = ['Cancer', 'Benign', 'Normal', 'Benign_additional']
views = ['LCC', 'LMLO', 'RCC', 'RMLO']

def get_image_size(impath):
    im = Image.open(impath)
    return im.size

imsize_dict = {}
for category in categories:
    uncomp_cat_dir = os.path.join(root_dir, 'png', compression_name[0], category)
    comp_cat_dir = os.path.join(root_dir, 'png', compression_name[1], category)
    case_names = os.listdir(uncomp_cat_dir)
    for case_name in tqdm(case_names):
        dcm_files = [v + '.png' for v in views]
        for df in dcm_files:
            uncomp_imsize = get_image_size(os.path.join(uncomp_cat_dir, case_name, df))
            comp_imsize = get_image_size(os.path.join(comp_cat_dir, case_name, df))
            if uncomp_imsize != comp_imsize:
                raise ValueError("Uncompressed image size differs from compressed, {} vs {}"\
                        .format(uncomp_im_size, comp_im_size) + \
                        "in case {}/{}/{}".format(category, case_name, df))
            if uncomp_imsize in imsize_dict.keys():
                imsize_dict[uncomp_imsize] += 1
            else:
                imsize_dict[uncomp_imsize] = 1

print(imsize_dict)
