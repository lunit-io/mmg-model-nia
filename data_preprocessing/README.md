# Prepare dataset

## Download dataset

We assume the data is downloaded in $DATA_ROOT

Store data in $DATA_ROOT/dcm by following command,

```
cd $DATA_ROOT
mkdir dcm
```

## Data preprocess
To train our model, we have to first convert dicom files to image files.
Then, the information about the data(such as the path to image file or annotations) is gathered.

1. To convert dicoms to pngs, use following command
`$ python dcm2png.py --dcm-root $DATA_ROOT/dcm --dst-root $DATA_ROOT/png`

Note: this might take several hours

2. To gather informations,
`$ python generate_pickle.py --root_dir $DATA_ROOT --anno_dir $ANNOTATION_ROOT --fast_build`
where $ANNOTATION_ROOT is directory that contains annotated json files, e.g. $DATA_ROOT/annotation_result_1st/__results/dst_json/20190930172251_KST/success

Note: fast_build options is available only if png files are generated in advance.

3. Shuffle pickle file here(for fold generation)
`$ python shuffle_pickle.py`

