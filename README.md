
# XNM experiments on VQA2.0

### Pipeline to preprocess data
1. Download [glove pretrained 300d word vectors](http://nlp.stanford.edu/data/glove.840B.300d.zip)

2. Preprocess glove file by running the command:
```
python preprocess_glove.py </path/to/downloaded/glove/file> </path/to/output/glove/pickle/file>```

3. Preprocess VQA2.0 train questions and obtain two output files: train_questions.pt and vocab.json.
```
python preprocess_questions.py --glove_pt </path/to/generated/glove/pickle/file> --input_questions_json </your/path/to/v2_OpenEnded_mscoco_train2014_questions.json> --input_annotations_json </your/path/to/v2_mscoco_train2014_annotations.json> --output_pt </your/output/path/train_questions.pt> --vocab_json </your/output/path/vocab.json> --mode train
```
> To combine the official train set and val set for training, just use : to join multiple json files. For example, `--input_questions_json train2014_questions.json:val2014_questions.json`

4. Preprocess VQA2.0 val questions. Note `--vocab_json` must be the one that is generated last step.
```
python preprocess_questions.py --input_questions_json </your/path/to/v2_OpenEnded_mscoco_val2014_questions.json> --input_annotations_json </your/path/to/v2_mscoco_val2014_annotations.json> --output_pt </your/output/path/val_questions.pt> --vocab_json </just/generated/vocab.json> --mode val
```

5. Download grounded features from the [repo](https://github.com/peteanderson80/bottom-up-attention) of paper [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)
```
wget https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
```

6. Unzip it and preprocess features
```
python preprocess_features.py --input_tsv_folder /your/path/to/trainval_36/ --output_h5 /your/output/path/trainval_feature.h5
```

Before training, make sure your have following files in the folder data/files:
- vocab.json
- train_questions.pt
- trainval_feature.h5
- val_questions.pt

### Train
```
python main.py 
```

### Visualization
Startup `visualize.ipynb` and follow the instructions.

### Acknowledgment
The code was taken partially from the original [XNM-net repo](https://github.com/shijx12/XNM-Net) to implement [Explainable and Explicit Visual Reasoning over Scene Graphs](https://arxiv.org/abs/1812.01855) paper

