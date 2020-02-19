BASEDIR=$(dirname "$0")
wget http://dvmvd-4602.kmlltpro.corp.kuaishou.com/prosody_prediction/biaobei_pretrained.tar.gz -O "$BASEDIR"/biaobei_pretrained.tar.gz
tar -zxvf "$BASEDIR"/biaobei_pretrained.tar.gz -C "$BASEDIR"
rm "$BASEDIR"/biaobei_pretrained.tar.gz