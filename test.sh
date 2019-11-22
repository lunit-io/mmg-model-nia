gpu=$1
pkldir=$2
imagedir=$3

python main.py -a resnet34 --gpu $gpu --fold-number 5 --current-fold-number 1 $pkldir --prefix-image-dir-path $imagedir --image-size 960 640
python main.py -a resnet34 --gpu $gpu --fold-number 5 --current-fold-number 2 $pkldir --prefix-image-dir-path $imagedir --image-size 960 640
python main.py -a resnet34 --gpu $gpu --fold-number 5 --current-fold-number 3 $pkldir --prefix-image-dir-path $imagedir --image-size 960 640
python main.py -a resnet34 --gpu $gpu --fold-number 5 --current-fold-number 4 $pkldir --prefix-image-dir-path $imagedir --image-size 960 640
python main.py -a resnet34 --gpu $gpu --fold-number 5 --current-fold-number 5 $pkldir --prefix-image-dir-path $imagedir --image-size 960 640
python main.py -a densenet121 --gpu $gpu --fold-number 5 --current-fold-number 1 $pkldir --prefix-image-dir-path $imagedir --image-size 960 640 --batch-size 20
python main.py -a densenet121 --gpu $gpu --fold-number 5 --current-fold-number 2 $pkldir --prefix-image-dir-path $imagedir --image-size 960 640 --batch-size 20
python main.py -a densenet121 --gpu $gpu --fold-number 5 --current-fold-number 3 $pkldir --prefix-image-dir-path $imagedir --image-size 960 640 --batch-size 20
python main.py -a densenet121 --gpu $gpu --fold-number 5 --current-fold-number 4 $pkldir --prefix-image-dir-path $imagedir --image-size 960 640 --batch-size 20
python main.py -a densenet121 --gpu $gpu --fold-number 5 --current-fold-number 5 $pkldir --prefix-image-dir-path $imagedir --image-size 960 640 --batch-size 20

python main.py -a resnet34 --gpu $gpu --fold-number 5 --current-fold-number 1 -e --resume resnet34-5-1-model_best.pth.tar --use-compressed $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > resnet34-5-1-compressed
python main.py -a resnet34 --gpu $gpu --fold-number 5 --current-fold-number 2 -e --resume resnet34-5-2-model_best.pth.tar --use-compressed $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > resnet34-5-2-compressed
python main.py -a resnet34 --gpu $gpu --fold-number 5 --current-fold-number 3 -e --resume resnet34-5-3-model_best.pth.tar --use-compressed $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > resnet34-5-3-compressed
python main.py -a resnet34 --gpu $gpu --fold-number 5 --current-fold-number 4 -e --resume resnet34-5-4-model_best.pth.tar --use-compressed $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > resnet34-5-4-compressed
python main.py -a resnet34 --gpu $gpu --fold-number 5 --current-fold-number 5 -e --resume resnet34-5-5-model_best.pth.tar --use-compressed $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > resnet34-5-5-compressed
python main.py -a resnet34 --gpu $gpu --fold-number 5 --current-fold-number 1 -e --resume resnet34-5-1-model_best.pth.tar $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > resnet34-5-1-uncompressed
python main.py -a resnet34 --gpu $gpu --fold-number 5 --current-fold-number 2 -e --resume resnet34-5-2-model_best.pth.tar $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > resnet34-5-2-uncompressed
python main.py -a resnet34 --gpu $gpu --fold-number 5 --current-fold-number 3 -e --resume resnet34-5-3-model_best.pth.tar $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > resnet34-5-3-uncompressed
python main.py -a resnet34 --gpu $gpu --fold-number 5 --current-fold-number 4 -e --resume resnet34-5-4-model_best.pth.tar $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > resnet34-5-4-uncompressed
python main.py -a resnet34 --gpu $gpu --fold-number 5 --current-fold-number 5 -e --resume resnet34-5-5-model_best.pth.tar $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > resnet34-5-5-uncompressed
python main.py -a densenet121 --gpu $gpu --fold-number 5 --current-fold-number 1 -e --resume densenet121-5-1-model_best.pth.tar --use-compressed $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > densenet121-5-1-compressed
python main.py -a densenet121 --gpu $gpu --fold-number 5 --current-fold-number 2 -e --resume densenet121-5-2-model_best.pth.tar --use-compressed $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > densenet121-5-2-compressed
python main.py -a densenet121 --gpu $gpu --fold-number 5 --current-fold-number 3 -e --resume densenet121-5-3-model_best.pth.tar --use-compressed $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > densenet121-5-3-compressed
python main.py -a densenet121 --gpu $gpu --fold-number 5 --current-fold-number 4 -e --resume densenet121-5-4-model_best.pth.tar --use-compressed $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > densenet121-5-4-compressed
python main.py -a densenet121 --gpu $gpu --fold-number 5 --current-fold-number 5 -e --resume densenet121-5-5-model_best.pth.tar --use-compressed $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > densenet121-5-5-compressed
python main.py -a densenet121 --gpu $gpu --fold-number 5 --current-fold-number 1 -e --resume densenet121-5-1-model_best.pth.tar $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > densenet121-5-1-uncompressed
python main.py -a densenet121 --gpu $gpu --fold-number 5 --current-fold-number 2 -e --resume densenet121-5-2-model_best.pth.tar $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > densenet121-5-2-uncompressed
python main.py -a densenet121 --gpu $gpu --fold-number 5 --current-fold-number 3 -e --resume densenet121-5-3-model_best.pth.tar $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > densenet121-5-3-uncompressed
python main.py -a densenet121 --gpu $gpu --fold-number 5 --current-fold-number 4 -e --resume densenet121-5-4-model_best.pth.tar $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > densenet121-5-4-uncompressed
python main.py -a densenet121 --gpu $gpu --fold-number 5 --current-fold-number 5 -e --resume densenet121-5-5-model_best.pth.tar $pkldir --prefix-image-dir-path $imagedir --image-size 960 640  > densenet121-5-5-uncompressed

python aggregate.py -a resnet34 --fold-number 5 # resnet34-5fold-result
python aggregate.py -a densenet121 --fold-number 5 # densenet121-5fold-result