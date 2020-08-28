# GenerateCuttingEdgeAndCNNReconstruction
Generate Cutting Edges And CNN Reconstruction

You need master data, so at least one measurement of a cutting edge. In the best case this should be an ideal cutting edge.
This can be used to generate further training data with the code. The training and validation data are designed to always provide ideal references.

But the test data are cutting edges with defects! These are then reconstructed with the help of an autoencoder, built up with CNNs, and parameters can be determined to validate this cutting edge. 

The main code is stored in the file 


```json
main_all.py
```

and also serves as a guide.
