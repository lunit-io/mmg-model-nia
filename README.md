
# mmg-nia Repository for NIA project    
    
## How to use it 
1. `$ git clone https://github.com/lunit-io/mmg-nia`  
2. `$ pip install -r requirements.txt` 
3. `$ cd data_preprocessing` and do data-preprocessing [here](https://github.com/lunit-io/mmg-model-nia/tree/master/data_preprocessing)
4. To train and test 5-fold cross validation, use `$ sh test.sh 0 data_preprocessing/db/shuffled_db.pkl /lunit/data/mmg/mg_nia`  
5. `$ cat resnet34-5fold-result`  
```  
compressed
   threshold : 0.1
         calculated accuracy is 0.8150886790885683
         calculated specificity is 0.8227048930437848
         calculated sensitivity is 0.7889675985264972
   threshold : 0.15
         calculated accuracy is 0.8451890694648247
         calculated specificity is 0.878533964063642
         calculated sensitivity is 0.7308764166673534
   threshold : 0.2
         calculated accuracy is 0.8608703452476536
         calculated specificity is 0.9128091136592129
         calculated sensitivity is 0.6827785378337696
   calculated auc is 0.8896701982655968
uncompressed
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
compressed
   threshold : 0.1
         calculated accuracy is 0.8405651319250256
         calculated specificity is 0.8526687249469763
         calculated sensitivity is 0.7984347172444817
   threshold : 0.15
         calculated accuracy is 0.8595072953293281
         calculated specificity is 0.888602910574434
         calculated sensitivity is 0.7593899157536472
   threshold : 0.2
         calculated accuracy is 0.8708727262659541
         calculated specificity is 0.9096908783863396
         calculated sensitivity is 0.7374108776754932
   calculated auc is 0.9031778432556026
uncompressed
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
