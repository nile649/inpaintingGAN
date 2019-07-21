DIR='./models'
URL='https://drive.google.com/uc?export=download&id=1gyQrN1M2o8E0uwTbR3Qtcj-Im3PZXeg7'

echo "Downloading pre-trained models..."
mkdir -p $DIR
FILE="$(curl -sc /tmp/gcokie "${URL}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')" 
curl -Lb /tmp/gcokie "${URL}&confirm=$(awk '/_warning_/ {print $NF}' /tmp/gcokie)" -o "$DIR/${FILE}" 

echo "Done Downloading trained model..."
