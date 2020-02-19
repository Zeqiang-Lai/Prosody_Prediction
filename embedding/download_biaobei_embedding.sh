BASEDIR=$(dirname "$0")
wget http://dvmvd-4602.kmlltpro.corp.kuaishou.com/prosody_prediction_biaobei_embedding.tar.gz -O "$BASEDIR"/prosody_prediction_biaobei_embedding.tar.gz
tar -zxvf "$BASEDIR"/prosody_prediction_biaobei_embedding.tar.gz -C "$BASEDIR"
rm "$BASEDIR"/prosody_prediction_biaobei_embedding.tar.gz