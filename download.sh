echo 'Downloading Stanford sentiment analysis dataset.'
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
echo 'Unpacking Stanford sentiment analysis dataset.'
tar -xvzf aclImdb_v1.tar.gz
echo 'Downloading GloVe... Might take a while.'
wget http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip
echo 'Unpacking GloVe... Please, be patient.'
unzip glove.42B.300d.zip
