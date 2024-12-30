Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 160, 160, 32)      896       
 max_pooling2d (MaxPooling2D  (None, 80, 80, 32)       0         
 conv2d_1 (Conv2D)           (None, 80, 80, 64)        18496     
 max_pooling2d_1 (MaxPooling (None, 40, 40, 64)        0         
 dense (Dense)               (None, 512)               1049088   
 dropout (Dropout)           (None, 512)               0         
 dense_1 (Dense)             (None, 2)                 1026      
=================================================================
Total params: 1,069,506
Trainable params: 1,069,506
Non-trainable params: 0
_________________________________________________________________
