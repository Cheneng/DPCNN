# Deep Pyramid Convolutional Neural Networks for Text Categorization

> This is a simple version of the paper *Deep Pyramid Convolutional Neural Networks for Text Categorization*.


!['model'](./pictures/figure1.png)


You should rewrite the Dataset class in the data/dataset.py  
and put your data in '/data/train' or any other directory.

run by

```
python main.py --lr=0.001 --epoch=20 --batch_size=64 --gpu=0 --seed=0 --label_num=2			
```

## Some Experiments  
> 	I personally run the model in one dataset about Ad identify.  
	And make a comparition to the vanilla TextCNN, LSTM and our DPCNN. 
	 
Here are the step loss in **TextCNN** and **LSTM**.  
<img src="./pictures/textcnn.png" width="350" height="250">  <img src="./pictures/lstm.png" width="350" height="250"> 


This is the loss of the **DPCNN**.  
<img src="./pictures/dpcnn.png" width="350" height="250">

