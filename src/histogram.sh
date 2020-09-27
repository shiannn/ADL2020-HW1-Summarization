#! /bin/bash

testFile=${1}
### generate test.pkl
python3.6 getSeq2Tag.py ${testFile}
### predict predict.jsonl
python3.6 plotingRelative.py \
    seq2tagCheckpoint/BCEloader1.pt seq2tagEmbedding/embedding.pkl \
     testSeq2Tag.pkl