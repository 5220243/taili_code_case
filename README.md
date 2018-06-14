# taili_code_case


This project contains part of the Python code for the SLIC-CNN method in the article "Intelligent Object-oriented Large Scale Geological Mapping Algorithm: A Case Study of Taili."

There are four main files, pic2CifarArr, CNN_input, CNN_train, CNN_slic.

The pic2CifarArr file can index the position of the picture list text and store the picture after serialization.
The CNN_input file can be used to serialize the reading of pictures. 
The CNN_train file can be used to construct the output of CNN networks and training results. 
The CNN_slic file outputs the SLIC map patch.

After obtaining the above results, the GIS tool is used to vectorize the slic classification map, and the results of the SLIC_CNN can be obtained by “spatial linking” with the CNN classification result. 
Subsequent Nibble and mode decision processes in the article are implemented in GIS tools and will not be repeated here.

The resolution of the picture in the code is 32*32 pixels.
