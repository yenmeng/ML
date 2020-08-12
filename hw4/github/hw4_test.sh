#!/usr/bin/env bash
wget https://www.dropbox.com/s/o3y6rkavha6m5ax/ckpt_gru.model?dl=1
mv ckpt_gru.model\?dl\=1 ckpt_gru.model
wget https://www.dropbox.com/s/lk3093hx5g0huzy/w2v_all_gru.model?dl=1
mv w2v_all_gru.model\?dl\=1 w2v_all_gru.model
wget https://www.dropbox.com/s/8akmn9wci9uivy6/w2v_all_gru.model.wv.vectors.npy?dl=1
mv w2v_all_gru.model.wv.vectors.npy\?dl\=1 w2v_all_gru.model.wv.vectors.npy
wget https://www.dropbox.com/s/f4m4vsrqdvorsh3/w2v_all_gru.model.trainables.syn1neg.npy?dl=1
mv w2v_all_gru.model.trainables.syn1neg.npy\?dl\=1 w2v_all_gru.model.trainables.syn1neg.npy
python3 hw4_testing.py $1 $2 

