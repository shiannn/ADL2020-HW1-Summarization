# ADL HW1 README
1.  training
    -   mkdir data/ and put all json in data
    -   generate data
        -   getSeq2Tag.py [training data jsonl path]
        -   getSeq2Tag.py [validating data jsonl path]
        -   getSeq2Tag.py [testing data jsonl path]
        -   getSeq2Seq.py [training data jsonl path]
        -   getSeq2Seq.py [validating data jsonl path]
        -   getSeq2Seq.py [testing data jsonl path]

    -   make folder to save checkpoint and chart
        -   mkdir figplot checkpoint
        -   mkdir figplot_seq2seq checkpoint_seq2seq
        -   mkdir figplot_attentionBi checkpoint_attentionBi
    -   seq2tag
        -   python3.6 trainSeq2Tag.py [training data path] [validating data path] [embedding path] [ground truth jsonl path]

    -   seq2seq
        -   python3.6 trainSeq2Seq.py [training data path] [validating data path] [embedding path] [load model ('none')]

    -   attention
        -   python3.6 trainAttention.py [training data path] [validating data path] [embedding path] [load model ('none')]

2.  predicting
    -   seq2tag
        -   ./extractive.sh test.jsonl predict.jsonl
    -   seq2seq
        -   ./seq2seq.sh test.jsonl predict.jsonl
    -   attention
        -   ./attention.sh test.jsonl predict.jsonl

3.  ploting
    -   visualize attention
        -   python3.6 plotingAttention.py [encoder path] [decoder path] [embedding path] [testdata path]

    -   plot relative position of summary
        -   python3.6 plotingRelative.py [seq2tag model path] [embedding path] [testdata path]