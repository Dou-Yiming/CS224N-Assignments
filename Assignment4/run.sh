#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 D:/coding/language/Anaconda3/envs/torch_env/python.exe run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --cuda
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 D:/coding/language/Anaconda3/envs/torch_env/python.exe run.py decode model_128.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs_128.txt --cuda
elif [ "$1" = "train_local" ]; then
	D:/coding/language/Anaconda3/envs/torch_env/python.exe run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json
elif [ "$1" = "test_local" ]; then
    D:/coding/language/Anaconda3/envs/torch_env/python.exe run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	D:/coding/language/Anaconda3/envs/torch_env/python.exe vocab.py --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en vocab.json
else
	echo "Invalid Option Selected"
fi
