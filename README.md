# Deep Pyramid Convolutional Neural Networks for Text Categorization

> This is a simple version of the paper *Deep Pyramid Convolutional Neural Networks for Text Categorization*.


!['model'](./pictures/figure1.png)


You should rewrite the Dataset class in the data/dataset.py  
and put your data in '/data/train' or any other directory.

run by

```
python main.py --lr=0.001 --epoch=20 --batch_size=64 --gpu=0 --seed=0 --label_num=2			
```
Result:  
	I personally run the model in one dataset about Ad identify.  
	And Compare to the vanilla TextCNN, LSTM and our DPCNN. 
	 
#### TextCNN
!['TextCNN'](./pictures/textcnn.png)
	
#### LSTM  
!['LSTM'](./pictures/lstm.png)
	
### DPCNN
!['DPCNN](./pictures/dpcnn.png)