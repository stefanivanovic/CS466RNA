# RNA secondary structure prediction via machine learning

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
