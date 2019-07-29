# TF Expr Framework
Personal framework for conducting experiments with tensorflow

This was designed and implemented before tensflow layer interface came out. 

a pytorch like interface framework, including:
* boilerpipe codes of logging, model save/load, weight dumping, partial freezing weight
* advanced ops like convolutional rnn cell, beam search
* abstract interface for supervised training, policy gradient training, structured learning training
* also support horovod as backend for paralleling training

## model
the asbtract pytorch like inteface
#### module.py
* AbstractModule: module as base component
* AbstractModel: subclass of module for supervised training, with additional interface of loss, build_trn_tst_graph, build_tst_graph and so on
* AbstractPGModel: subclass of AbstractModel for policy gradient training, with additional inteface of rollout (sampling) phase
* AbstractStructModel: subclass of AbstractModel for structured learning training, with addition interface of scoring (x,y) phase

#### data.py
* Reader: data loader interface

#### trntst.py
* TrnTst: boilerpipe codes for supervised training, such as building graph, running session, saving/loading model, validation
* PGTrnTst: boilerpipe codes for policy gradient training
* StructTrnTst: boilerpipe codes for structured learning

## impl
implementation of some common modules used in various experiments
* encoder/vlad.py: VLAD pooling
* encoder/conv_rnn.py: convolutional RNN
* encoder/birnn.py: bidirectional RNN
* gradient/poincare.py: get gradient for embedding on Poincare manifold
* gradient/lorentz.py: get gradient for embedding in Lorentz space

## util
common utility functions
