#! /bin/bash

testFile=${1}
predictFile=${2}
### generate test.pkl
python3.6 getSeq2Tag.py ${testFile}
### predict predict.jsonl
python3.6 predictSeq2Tag.py \
    seq2tagCheckpoint/BCEloader1.pt seq2tagEmbedding/embedding.pkl \
     testSeq2Tag.pkl ${predictFile}