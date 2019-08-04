# Text-Classification-PyTorch
Text classification tutorial by PyTorch.
## Requirements
* python 3.6+
* torch==1.1.0
* pandas
* matplotlib
* nltk
* scikit_learn
## IMDB Dataset
This dataset contains movie reviews along with their associated binary
sentiment polarity labels. It is intended to serve as a benchmark for
sentiment classification. The core dataset contains 50,000 reviews split evenly into 25k train
and 25k test sets. The overall distribution of labels is balanced (25k
pos and 25k neg). We also include an additional 50,000 unlabeled
documents for unsupervised learning. 
## Get Start
1. **Install all the required package.**   
`$ cd Text-Classification-PyTorch`  
`$ pip install -r requirements.txt` 
2. **Download dataset**  
`$ wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz`   
`$ tar -zxvf aclImdb_v1.tar.gz`  
3. **Initialize data.**  
`$ python initialize.py`  
Or use `$ python initialize.py -h` for help.  
4. **Train.**  
`$ !rm -rf output`  
`$ python train.py`  
Or use `$ python train.py -h` for help.  
5. **Evaluation.**  
`$ python eval.py`  
Or use `$ python eval.py -h` for help.  
6. **Check evaluation results**  
Open `--name` file to view PR curve.
## Training in Google Colab
https://colab.research.google.com/drive/1VJmSx-vThBFlGZYJ9sKWDMINKWOzFNCD
## Training in Docker
1. Pull image  
`$ docker pull wisedoge/text_clf_pytorch`
2. Run  
`$ docker run -it wisedoge/text_clf_pytorch`
## Results
### Parameters
| #    | Param           | Mini | Base  | Large |
| ---- | --------------- | ---- | ----- | ----- |
| 1    | Vocab           | 6000 | 30000 | 30000 |
| 2    | Max len         | 128  | 256   | 512   |
| 3    | Embedding dim   | 64   | 256   | 256   |
| 4    | Hidden dim      | 64   | 128   | 512   |
| 5    | Attn dim        | 64   | 128   | 256   |
| 6    | Dropout         | 0.2  | 0.2   | 0.2   |
| 7    | Num LSTM layer  | 2    | 3     | 3     |
| 8    | Num DPCNN block | 5    | 7     | 8     |

| #          | Used Parameters |
| ---------- | --------------- |
| LSTM       | 1,2,3,4,6,7     |
| CNNAttn    | 1,2,3,4,5       |
| CNN        | 1,2,3,4         |
| BiLSTMAttn | 1,2,3,4,5       |
| TextCNN    | 1,2,3           |
| DPCNN      | 1,2,3,8         |
### Accuracy
| #          | Mini        | Base        | Large       |
| ---------- | ----------- | ----------- | ----------- |
| LSTM       | 0.75860     | 0.78552     | 0.60808     |
| CNNAttn    | 0.79340     | 0.82924     | 0.83192     |
| CNN        | 0.84788     | 0.86588     | 0.89580     |
| BiLSTMAttn | 0.84812     | 0.83052     | 0.87396     |
| TextCNN    | 0.85752     | **0.87776** | **0.90012** |
| DPCNN      | **0.85872** | 0.86100     | 0.87968     |
### PR-Curve
**Mini**  
![](docs/mini.svg)  
**Base**  
![](docs/base.svg)  
**Large**  
![](docs/large.svg)  

