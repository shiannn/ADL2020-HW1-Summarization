#! /bin/bash

testFile=${1}
predictFile=${2}
### generate test.pkl
python3.6 getSeq2Seq.py ${testFile}
### predict predict.jsonl
python3.6 predictSeq2Seq.py \
    seq2seqCheckpoint/encoder3.pt seq2seqCheckpoint/decoder3.pt seq2seqEmbedding/embedding.pkl \
     testSeq2Seq.pkl ${predictFile}