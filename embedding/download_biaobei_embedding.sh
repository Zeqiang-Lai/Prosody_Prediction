BASEDIR=$(dirname "$0")
wget http://dvmvd-4602.kmlltpro.corp.kuaishou.com/prosody_prediction/biaobei_embedding.tar.gz -O "$BASEDIR"/biaobei_embedding.tar.gz
tar -zxvf "$BASEDIR"/biaobei_embedding.tar.gz -C "$BASEDIR"
rm "$BASEDIR"/biaobei_embedding.tar.gz