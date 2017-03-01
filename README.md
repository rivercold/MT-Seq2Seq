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

### Usage for Attention model
optional arguments:

  -h, --help            show this help message and exit
  
  -batch BATCH          Whether or not use batch training (default: False)
  
  -layer LAYER          Number of LSTM layers (default: 2)
  
  -embed EMBED          Embedding size (default: 200)
  
  -hid HID              Hidden size (default: 128)
  
  -att ATT              Attention size (default: 128)
  
  -load LOAD            Model path to load (default: None)
  
  -save SAVE            Whether or not save the model during training
                        (default: True)
                        
  -se SE                Starting epoch, used for continue training from a
                        certain node (default: 0)
                        
  -bs BS                Batch size (default: 20)
  
  --dynet-mem DYNET_MEM
  
  --dynet-gpu-ids DYNET_GPU_IDS
  
  -beam BEAM            Whether or not use Beam search in translation
                        (default: False)
                        
  -beam-width BEAM_WIDTH
                        Beam width (default: 3)
                        
  -pred PRED            Only prediction without training (default: False)
  
  -train_en TRAIN_EN    Target sentence file for training (default: None)
  
  -train_de TRAIN_DE    Source sentence file for training (default: None)
  
  -test_en TEST_EN      Target sentence file for testing (default: None)
  
  -test_de TEST_DE      Source sentence file for testing (default: None)
  
  -result RESULT        Result translation file for target testing sentences
                        (default: None)
### Example command
Train:
```bash
python Attention.py --dynet-mem 4000 -batch True -save True -train_en ../data/train.en-de.low.en -train_de ../data/train.en-de.low.de
```
