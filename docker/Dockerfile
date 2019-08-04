FROM pytorch/pytorch:latest

RUN git clone https://github.com/WiseDoge/Text-Classification-PyTorch.git 
WORKDIR /workspace/Text-Classification-PyTorch
RUN pip install -r requirements.txt
RUN apt-get update 
RUN apt-get install wget
RUN wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
RUN tar -zxf aclImdb_v1.tar.gz

