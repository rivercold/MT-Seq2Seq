# MT-Seq2Seq
## Assignment 1
### Code Structure

+ -assign1/
    + Attention.py: implementation of bi-directional attention model
    + encdec.py: implementation of vanilla encoder-decoder model
    + util.py: util functions 
    + report.pdf: Assignment report
    + -output/
        + blind.predict: prediction of blind set
        + test.predict: prediction of test set
### Example command
Train:
```bash
python Attention.py --dynet-mem 4000 -batch True -save True -train_en ../data/train.en-de.low.en -train_de ../data/train.en-de.low.de
```

### Usage for Attention model
optional arguments:
  
  -batch: Whether or not use batch training (default: False)
  
  -layer: Number of LSTM layers (default: 2)
  
  -embed: Embedding size (default: 200)
  
  -hid: Hidden size (default: 128)
  
  -att: Attention size (default: 128)
  
  -load: Model path to load (default: None)
  
  -save: Whether or not save the model during training (default: True)
                        
  -se: Starting epoch, used for continue training from a certain node (default: 0)
                        
  -bs: Batch size (default: 20)
  
  -beam: Whether or not use Beam search in translation (default: False)
                        
  -beam-width: Beam width (default: 3)
                        
  -pred: Only prediction without training (default: False)
  
  -train_en: Target sentence file for training (default: None)
  
  -train_de: Source sentence file for training (default: None)
  
  -test_en: Target sentence file for testing (default: None)
  
  -test_de: Source sentence file for testing (default: None)
  
  -result: Result translation file for target testing sentences
                        (default: None)
