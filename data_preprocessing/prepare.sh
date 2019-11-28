#!/bin/bash

DATA_ROOT=$1
ANNOTATION_ROOT=$DATA_ROOT/annotation_result_1st/__results/dst_json/20190930172251_KST

python dcm2png.py --dcm-root $DATA_ROOT/dcm --dst-root $DATA_ROOT/png
python generate_pickle.py --root_dir $DATA_ROOT --anno_dir $ANNOTATION_ROOT --fast_build
python shuffle_pickle.py

