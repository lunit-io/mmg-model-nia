# Prepare dataset

## Library installation: GDCM
```
apt-get update

apt-get install -y --no-install-recommends curl build-essential cmake swig

curl -L -O https://sourceforge.net/projects/gdcm/files/gdcm%202.x/GDCM%202.8.9/gdcm-2.8.9.tar.gz

tar -xzf gdcm-2.8.9.tar.gz

mv gdcm-2.8.9 gdcm-src

mkdir gdcm-build

cd gdcm-build

cmake -DGDCM_WRAP_PYTHON=ON \
          -DGDCM_BUILD_SHARED_LIBS=ON \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          ../gdcm-src

make -j$(nproc) && make install && \

cd /usr/local/lib && cp _gdcmswig.so gdcmswig.py gdcm.py $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

cd - 

rm -rf gdcm-src gdcm-build gdcm-2.8.9.tar.gz
```

## Dataset

We assume the data is downloaded in $DATA_ROOT

Store data in $DATA_ROOT/dcm by following command,

```
cd $DATA_ROOT
mkdir dcm
```

Data structure

```
$DATA_ROOT
└── dcm
    ├── 1
    │   ├── LCC.dcm
    │   ├── LMLO.dcm
    │   ├── RCC.dcm
    │   └── RMLO.dcm
    ├── 2
    │   ├── LCC.dcm
    │   ├── LMLO.dcm
    │   ├── RCC.dcm
    │   └── RMLO.dcm
...
 
$DATA_ROOT
└── annotation_result_1st
    ├── Various time stamps…(e.g., 20191015082002_KST)
    ├── \_\_results
    │   ├── dst_json
    │   |   ├── Various time stamps...
    │   |   ├── 20190930172251_KST
    │   │   |   ├── Success     
    │   │   │   |   ├── Type_id.json (e.g. Cancer_00367.json)   
...         
```
 

## Data preprocessing
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

