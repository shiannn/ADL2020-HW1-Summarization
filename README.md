# ADL HW1 Sample Code

## How TA Create Testing Environment
Note: We only allow python3.6 or 3.7.
```bash
bash install_packages.sh
```

## Preproccessing
* Modify the paths in `datasets/{seq2seq,seq_tag}/config.json`
* You can change the value in the configs.
```bash
python src/preprocess_seq2seq.py datasets/seq2seq/
python src/preprocess_seq_tag.py datasets/seq_tag/
```

> If you are new to deep learning, we suggest you check out [pytorch lightning](https://github.com/PyTorchLightning/pytorch-lightning).
