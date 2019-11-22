import pickle
import argparse
import random
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--source_path', type=str, default='./db/fast_db.pkl')
parser.add_argument('--target_path', type=str, default='./db/shuffled_db.pkl')

args = parser.parse_args()
with open(args.source_path, "rb") as f:
    db = pickle.load(f, encoding='iso-8859-1')

random.seed(0)
shuffled_db = sorted(list(db.items()))
random.shuffle(shuffled_db)
shuffled_db = OrderedDict(shuffled_db)

with open(args.target_path, "wb") as f:
    pickle.dump(shuffled_db, f)
