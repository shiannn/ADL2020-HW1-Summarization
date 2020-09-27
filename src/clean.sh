#! /bin/bash

### 2 embeddings
file="seq2tagEmbedding/embedding.pkl"
if [ -e $file ] ; then
    rm $file
fi

file="seq2seqEmbedding/embedding.pkl"
if [ -e $file ] ; then
    rm $file
fi

### 3 models
file="seq2seqCheckpoint/encoder3.pt"
if [ -e $file ] ; then
    rm $file
fi

file="seq2seqCheckpoint/decoder3.pt"
if [ -e $file ] ; then
    rm $file
fi


file="attentionCheckpoint/encoder3.pt"
if [ -e $file ] ; then
    rm $file
fi

file="attentionCheckpoint/decoder3.pt"
if [ -e $file ] ; then
    rm $file
fi

file="seq2tagCheckpoint/BCEloader1.pt"
if [ -e $file ] ; then
    rm $file
fi

### generated pkl
file="testSeq2Seq.pkl"
if [ -e $file ] ; then
    rm $file
fi

file="testSeq2Tag.pkl"
if [ -e $file ] ; then
    rm $file
fi

### predicted jsonl

file="seq2tag.jsonl"
if [ -e $file ] ; then
    rm $file
fi

file="seq2seq.jsonl"
if [ -e $file ] ; then
    rm $file
fi

file="attention.jsonl"
if [ -e $file ] ; then
    rm $file
fi