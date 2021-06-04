# AI Programming Final Project: 
Object Detection with PASCAL VOC 2012

Student ID : 2016310526 

Student Name : KimChanho

## Download Dataset
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
and unzip tar file

```
tar -xvf VOCtrainval_11-May-2012.tar 
```

then you can see the VOCdevkit directory

## Preprocess Data file
run 2016310526_datasest on terminal
```
python3 2016310526_dataset.py
```

then you can see 
```
len of train=13313, valid=2140, test=1664
dataset class test start...

dataset class test done...
```

If this lines printed, then data preprocessing is done.

total train 13313 images, valid 2140 images, test 1664 images

and directory tree looks like this
```
├── 2016310526_VOC_TEST_DATA.txt
├── 2016310526_VOC_TRAIN_DATA.txt
├── 2016310526_VOC_VALID_DATA.txt
├── 2016310526_dataset.py
├── VOCdevkit
│   └── VOC2012
```
## Train
Let's train
run 2016310526_train.py on terminal
```
python3 train.py
```
because python file name cannot import with number... so If you want to check train then modified name like train.py, dataset.py, loss.py

and also I use
```
scaler = torch.cuda.amp.GradScaler()
```

in train, so If you want to train same condition. you have to run this train code with cuda.


## Test
run 2016310526_test.py on terminal
```
python3 test.py
```

test should be run with gpu.

I don't know why but if on only cpu device there is an error when load model.

If you want to test with custom Images, then you can change 

```
root_dataset_dirpath = './VOCdevkit/VOC2012/'
...
result = predict_gpu(model, img_path, root_path=os.path.join(root_dataset_dirpath, 'JPEGImages/'), device=device)
image = cv2.imread(os.path.join(root_dataset_dirpath, 'JPEGImages/') + img_path)
```

root_dataset_dirpath and 'JPEGImages/'

after test finished. you can see test_images directory.

you can see the bndbox and text(class) on the input images.


