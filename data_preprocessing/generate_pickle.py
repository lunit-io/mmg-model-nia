import os
import json
import argparse
import pickle
from tqdm import tqdm
import pydicom as dicom
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, default='/path/to/data_root/',
                    help='root to the data dir')
parser.add_argument('--anno_dir', type=str, default='/path/to/annotations...success/', help='root to the annotation dir')
parser.add_argument('--output_path', type=str, default='./db/db.pkl')
parser.add_argument('--fast_build', default=False, action='store_true',
                    help='use png meta to extract image size')

args, _ = parser.parse_known_args()

root_dir = args.root_dir
compression_type = ['Uncompressed', 'Compressed']
categories = ['Cancer', 'Benign', 'Normal', 'Benign_additional']
views = ['LCC', 'LMLO', 'RCC', 'RMLO']

cat2type = {'Cancer': 2, 'Benign': 1, 'Benign_additional': 1, 'Normal': 0}
lesion_type_to_code = {'normal': 0, 'benign': 1, 'cancer': 2}

# list annotations, annotation file name is {category}_{case_name}.json
anno_list = [d for d in os.listdir(args.anno_dir) if d.endswith('.json')]
ckey2anno = {d[:-5] : os.path.join(args.anno_dir, d) for d in anno_list}

def read_annotation(case_key, view, image_size):
    # return a list of lesions
    # lesions is a dictionary, contains two key: 'contour', 'label'
    try:
        json_fn = ckey2anno[case_key]
    except KeyError:
        return None

    with open(json_fn, 'r') as fp:
        anno = json.load(fp)
    # check sanity
    assert anno["case_id"] == case_key, \
            "Loaded annotation doesn't match with the key, {} vs {}"\
            .format(anno["case_id"], case_key)
    case_annotation = {k: [] for k in views}
    height, width = image_size
    lesions = []
    for lesion_type, lesion_dict in anno["contour_list"].items():
        lesion_label = lesion_type_to_code[lesion_type] # 0,1,2
        if view.lower() in lesion_dict:
            view_dict = lesion_dict[view.lower()]
            for _, point_list in view_dict.items():
                contour = []
                for xy in point_list:
                    # convert coordinate system
                    r = xy['y'] + height // 2
                    c = xy['x'] + width // 2
                    r = min(max(r, 0), height - 1)
                    c = min(max(c, 0), width - 1)
                    contour.append([c, r])
                lesions.append({
                    'label': lesion_label,
                    'contour': contour
                })
    return lesions

db = {}
for category in categories:
    case_names = os.listdir(os.path.join(root_dir, 'dcm', compression_type[0], category))
    for case_name in tqdm(case_names):
        case_key = '{}_{}'.format(category, case_name)
        case_info = {}
        case_info['case_label'] = cat2type[category]
        # Make LMLO, LCC, RMLO, RCC information
        for view in views:
            view_info = {}
            view_info['uncompressed_image'] = os.path.join('png', compression_type[0], category, \
                                                                case_name, view + '.png')
            view_info['compressed_image'] = os.path.join('png', compression_type[1], category, \
                                                                case_name, view + '.png')
            uncompressed_dcm = os.path.join(root_dir, 'dcm', compression_type[0], category, \
                                            case_name, view + '.dcm')

            if args.fast_build:
                # use pre-extracted png
                im = Image.open(os.path.join(root_dir, view_info['uncompressed_image']))
                width, height = im.size
                image_size = height, width
            else:                    
                with dicom.dcmread(uncompressed_dcm) as ud:
                    image_size = ud.pixel_array.shape # height, width
            lesions = read_annotation(case_key, view, image_size)
            if lesions is None:
                lesions = []
                annotated = False
            else:
                annotated = True
            view_info['image_size'] = image_size
            view_info['lesions'] = lesions
            case_info[view] = view_info
        case_info['annotated'] = annotated
        db[case_key] = case_info

if not os.path.exists(os.path.dirname(args.output_path)):
    os.makedirs(os.path.dirname(args.output_path))

with open(args.output_path, 'wb') as fp:
    pickle.dump(db, fp)



