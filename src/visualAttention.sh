#! /bin/bash

testFile=${1}
### generate test.pkl
python3.6 getSeq2Seq.py ${testFile}
### predict predict.jsonl
python3.6 plotingAttention.py \
    attentionCheckpoint/encoder3.pt attentionCheckpoint/decoder3.pt seq2seqEmbedding/embedding.pkl \
     testSeq2Seq.pkl