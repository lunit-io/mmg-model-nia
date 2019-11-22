import argparse

parser = argparse.ArgumentParser(description='Aggregate data from evaluated images')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34')
parser.add_argument('--fold-number', default=5, type=int, metavar='N',
                    help='The fold number in cross validation')

args = parser.parse_args()

result = {
    "compressed": {
        "auc": 0.0,
        "threshold": {
            0.1: {
                "accuracy": 0.0,
                "specificity": 0.0,
                "sensitivity": 0.0
            },
            0.15: {
                "accuracy": 0.0,
                "specificity": 0.0,
                "sensitivity": 0.0
            },
            0.2: {
                "accuracy": 0.0,
                "specificity": 0.0,
                "sensitivity": 0.0
            }
        }
    },
    "uncompressed": {
        "auc": 0.0,
        "threshold": {
            0.1: {
                "accuracy": 0.0,
                "specificity": 0.0,
                "sensitivity": 0.0
            },
            0.15: {
                "accuracy": 0.0,
                "specificity": 0.0,
                "sensitivity": 0.0
            },
            0.2: {
                "accuracy": 0.0,
                "specificity": 0.0,
                "sensitivity": 0.0
            }
        }
    }
}

for image_type in ["compressed", "uncompressed"]:
    for i in range(1, args.fold_number+1):
        with open("{}-{}-{}-{}".format(args.arch, args.fold_number, i, image_type), "r") as f:
            lines = f.readlines()
        result[image_type]['auc'] += float(lines[-1].split()[-1]) / args.fold_number
        for idx, threshold in enumerate([0.1, 0.15, 0.2]):
            stat = result[image_type]['threshold'][threshold]
            stat['accuracy'] += float(lines[-1-(3-idx)*4+1].split()[-1]) / args.fold_number
            stat['specificity'] += float(lines[-1-(3-idx)*4+2].split()[-1]) / args.fold_number
            stat['sensitivity'] += float(lines[-1-(3-idx)*4+3].split()[-1]) / args.fold_number


with open("{}-{}fold-result".format(args.arch, args.fold_number), "w") as f:
    for image_type in ["compressed", "uncompressed"]:
        f.write(image_type+"\n")
        for threshold in [0.1, 0.15, 0.2]:
            stat = result[image_type]['threshold'][threshold]
            f.write('   threshold : {}\n'.format(threshold))
            f.write('         calculated accuracy is {}\n'.format(stat['accuracy']))
            f.write('         calculated specificity is {}\n'.format(stat['specificity']))
            f.write('         calculated sensitivity is {}\n'.format(stat['sensitivity']))
        f.write("   calculated auc is {}\n".format(result[image_type]['auc']))
