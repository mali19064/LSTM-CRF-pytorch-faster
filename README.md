# LSTM-CRF-pytorch-faster fd

This is a more than 1000X faster LSTM-CRF implementation modified from the slower version in the Pytorch official tutorial   (URL:https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html). 

I have modified the dynamic planning part, including Viterbi decoding and partition function calculation. In the experiment, it has achieved a speed increase of more than 50 times compared to the original version. Furthermore, the original version can only input one sample at a time when it is trained. 

In the most recent updated module 'LSTM_CRF_faster_parallel.py', I modified the model to support parallel computing for batch, so that the training time was greatly improved again. When the batchsize is large, parallel computing can bring you hundreds of times faster.

You only need to call the function 'neg_log_likelihood_parallel' in module 'lstm-crf-parallel.py', instead of the original 'neg_log_likelihood', to perform batch parallel training. 

The code defaults to training word embedding from scratch. If you need to use pre-training word embedding, or take another model's outputs as it's inputs, you need to make some changes to the code, directly take the embedding as inpus of method '_get_lstm_features_parallel'.

(in previous version, parallel version is not work, now the bug is rectified)
