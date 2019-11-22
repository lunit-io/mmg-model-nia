import pickle

total_fold_number = 5
with open('shuffled_db.pkl', 'rb') as fp:
    s_db = pickle.load(fp)

cur_fold_num = 0
for k, v in s_db.items():
    v['fold'] = cur_fold_num+1
    cur_fold_num += 1
    if cur_fold_num == total_fold_number:
        cur_fold_num = 0

with open('five_fold_db.pkl', 'wb') as fp:
    pickle.dump(s_db, fp)

