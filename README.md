# RNA secondary structure prediction via machine learning

## Requirements

This software requires numpy, pytorch, scipy, and tqdm. 

## Inference

To predict the secondary structure using the provided trained model, run:
```
python3 RNApred.py pred <input sequence>
```
An output secondary structure will then be generated. 
For instance, one can run the below command. 
```
python3 RNApred.py pred GGGGCCUUAGCUCAGCUGGGAGAGCGCCUGCUUUGCACGCAGGAGGUCAGCGGUUCGAUCCCGCUAGGCUCCA
```
This yields the secondary structure "(((((((..((((........))))((((((.......))))))....(((((.......)))))))))))).". 

## Training

To train a new model use the below command. 
```
python3 RNApred.py train <model file>
```
To perform inference using a new model, run the below command. 
```
python3 RNApred.py pred -m <model file> <input sequence>
```
