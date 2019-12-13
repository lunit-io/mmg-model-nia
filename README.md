
# mmg-nia Repository for NIA project    
    
## How to use it 
1. `$ git clone https://github.com/lunit-io/mmg-nia`  
2. `$ pip install -r requirements.txt` 
3. `$ cd data_preprocessing` and do data-preprocessing [here](https://github.com/lunit-io/mmg-model-nia/tree/master/data_preprocessing)
4. To train and test 5-fold cross validation, use `$ sh test.sh $GPU_ID $PICKLE_PATH $DATA_ROOT` \
e.g. `$ sh test.sh 0 data_preprocessing/db/shuffled_db.pkl /data/mmg/mg_nia`  
    `- If you want to use many GPUs, input multiple numbers: sh test.sh 0,1,2,3, ...`
5. `$ cat resnet34-5fold-result`  
```  
   threshold : 0.1
         calculated accuracy is 0.8144577092389047
         calculated specificity is 0.8112627121478205
         calculated sensitivity is 0.825194007255318
   threshold : 0.15
         calculated accuracy is 0.8477143885489631
         calculated specificity is 0.8706655574469052
         calculated sensitivity is 0.768865680293087
   threshold : 0.2
         calculated accuracy is 0.8648696254049115
         calculated specificity is 0.9067113501876425
         calculated sensitivity is 0.7214681631914128
   calculated auc is 0.9070296037910798
```  
6. `$ cat densenet121-5fold-result`  
```  
   threshold : 0.1
         calculated accuracy is 0.8379339405852875
         calculated specificity is 0.8391845548624011
         calculated sensitivity is 0.8327769440438639
   threshold : 0.15
         calculated accuracy is 0.8601385974141034
         calculated specificity is 0.8777076140580005
         calculated sensitivity is 0.7991593433754576
   threshold : 0.2
         calculated accuracy is 0.872767241617985
         calculated specificity is 0.9012399299294076
         calculated sensitivity is 0.7745227785230062
   calculated auc is 0.9193286916314279
```
