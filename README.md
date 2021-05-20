#### Rational Language Model

This repo contains code for a rational language model

Still needs to be cleaned up a bit. But currently what works is:

- `train_pl.py` (Used to pretrain the model )
- `train_prediction_rational_pl.py` (Train the model)


Currently the gumbell softmax is used for training. As this was easiest to implement.



#### Training

#### LSTM language model

The training of the Rational Extractor consists of different steps.
First we need to train the language model:

1) `python train_language_model.py configs/simple_lm_config.yml`

 