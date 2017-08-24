#!/bin/sh

# $1 scratch folder
# $2 feature extraction list
# $3 model selection



for i in "$@"
do
case $i in 
	-m=*|--mode=*)
	mode="${i#*=}"
	;;
esac
done

echo mode = ${mode}

# encoding feature
if [ mode="encoding" ]; then
python encoding_cnn.py "$2" "$3" 
elif [ mode="prediction" ]; then
python prediction_cnn.py "$2" "$3"
fi


