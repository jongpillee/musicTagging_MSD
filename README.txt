# README
----------------------------------------------------------------------------
* Contact Info *

<Jongpil Lee>
Korea Advanced Institute of Science and Technoloty (KAIST)
Graduate School of Culture Techonology (GSCT)
richter@kaist.ac.kr

----------------------------------------------------------------------------
* Description *

This is slightly modifided versions from our submission to the 2017 MIREX audio classification (train/test) tasks.
Used model is based on our previously published paper [https://arxiv.org/abs/1706.06810].

There are total two functions in this repo.

1. predicting 50 tags using sampleCNN learned from MSD tagging dataset.

2. transfer last hidden layer of the sampleCNN to your new task.
	This function consists of two stage: feature extraction and train/classification.

----------------------------------------------------------------------------
* Platform and Requirements *

<Dependencies>
keras 1.1.0
theano 0.8.2
python 2.7.6

<Python Libraries>
librosa
numpy
sklearn

<input>
30s, 22.05kHz, wav file is expected input 

----------------------------------------------------------------------------
* Use *

1. 50 tag prediction

./ForwardProp.sh -m=prediction /path/to/save/folder /path/to/fileList.txt

Example fileList.txt
/media/bach1/dataset/gtzan/blues/blues.00035.wav	
/media/bach1/dataset/gtzan/blues/blues.00036.wav	
/media/bach1/dataset/gtzan/blues/blues.00037.wav	

{"file_name": "./path/to/save/folder/individual_file_List.json", "prediction_msd": {"beautiful": "0.0206099", "punk": "0.00465381", "indie": "0.0876653", "male vocalists": "0.0211934", "female vocalist": "0.00529418", "heavy metal": "0.00191998", "pop": "0.063148", "sad": "0.015539", "00s": "0.0115924", "ambient": "0.0148107", "alternative": "0.0425866", "hard rock": "0.00436063", "electronic": "0.016531", "blues": "0.143018", "folk": "0.315052", "classic rock": "0.0361686", "alternative rock": "0.00850769", "90s": "0.00585691", "60s": "0.0267258", "indie rock": "0.0129534", "electronica": "0.00600895", "female vocalists": "0.0476008", "easy listening": "0.0104203", "dance": "0.00346507", "funk": "0.00661781", "House": "0.00164513", "80s": "0.00953005", "party": "0.00136872", "Mellow": "0.0486049", "electro": "0.00234408", "chillout": "0.017821", "happy": "0.00424408", "oldies": "0.0182328", "rnb": "0.00878901", "jazz": "0.123137", "70s": "0.0187786", "instrumental": "0.0407893", "indie pop": "0.0125248", "sexy": "0.00269948", "Hip-Hop": "0.00374524", "chill": "0.0139084", "guitar": "0.0837907", "country": "0.0271717", "metal": "0.00198551", "soul": "0.0420783", "catchy": "0.00135911", "rock": "0.118368", "acoustic": "0.203366", "Progressive rock": "0.0103604", "experimental": "0.024019"}}

These json files of file list would be saved in the save folder.



2. get last hidden layer and train svm onto new label dataset

# get last hidden layer
./ForwardProp.sh -m=encoding /path/to/save/folder /path/to/fileList.txt

# train and classification
./TrainAndClassify.sh /path/to/save/folder /path/to/trainListFile.txt /path/to/testListFile.txt /path/to/output
	
Example trainListFile.txt	
/media/bach1/dataset/gtzan/blues/blues.00029.wav	blues
/media/bach1/dataset/gtzan/blues/blues.00030.wav	blues
/media/bach1/dataset/gtzan/blues/blues.00031.wav	blues
/media/bach1/dataset/gtzan/blues/blues.00032.wav	blues
...

Example testListFile.txt
/media/bach1/dataset/gtzan/blues/blues.00035.wav	
/media/bach1/dataset/gtzan/blues/blues.00036.wav	
/media/bach1/dataset/gtzan/blues/blues.00037.wav	
	
Expected output file
/media/bach1/dataset/gtzan/blues/blues.00035.wav	blues
/media/bach1/dataset/gtzan/blues/blues.00036.wav	blues
/media/bach1/dataset/gtzan/blues/blues.00037.wav	blues

---------------------------------------------------------------------------

