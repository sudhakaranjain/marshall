## Marshall: Modality-Agnostic Representation learning by SHAred pre-training of muLtiple modaLities

### Setup
Python 3.8 or above
#### Install Requirements:
Run `make install`

#### Dataset:
 Download MS COCO dataset and place these folders inside `datasets` folder:
 -  train2017
 -  val2017
 -  annotations_trainval2017

#### Preprocess:
Extract `Car` class subset:  
`python preprocess_dataset.py`

To analyze classes, run below command:  
`python analyse_dataset.py`  
This outputs details of 80 class names and number of samples tagged by their `category_id`
