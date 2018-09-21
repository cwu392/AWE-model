# AWE-model
## This is the code is of our AAAI2019 paper "AWE: Asymmetric Word Embedding for Textual Entailment". 
```
Authors: Tengfei Ma, Chiamin Wu, Cao Xiao, Jimeng Sun
It gets state-of-the-art performance on the SciTail dataset(textual entailment) -- 84.4%. We release both the code and the pretrained model.
```

## Reference: https://github.com/yinwenpeng/SciTail
```
In this repository, we reference the Theano code from ACL2018 paper "End-Task Oriented Textual Entailment via Deep Explorations of Inter-Sentence Interactions"
```

## Description:
```
By using our asymmetric word embedding method, we learn interactions with "premise" and "hypothesis" directly from training data.

By adding this relationships to different models(DEISTE and Decomposable Attention Vanilla), our proposed method can improve their performance on both SciTail and SNLI dataset.

In the end, we achieved the current SOTA accuracy on SciTail which is 84.2%. (+2.1%)
```

## Jupyter Notebook code demo:
```
http://htmlpreview.github.io/?https://github.com/cwu392/AWE-model/blob/master/AAAI_Final_0.8441860465116279.html
```