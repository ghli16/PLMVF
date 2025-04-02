## Accurate prediction of virulence factors using pre-train protein language model and ensemble learning

This is the repository related to our manuscript Accurate prediction of virulence factors using pre-train protein language model and ensemble learning, currently in submission at BMC Genomics.

## Code
### Environment Requirement

The code has been tested running under Python 3.8.10. The required packages are as follows:
- numpy == 1.24.4
- pytorch== 1.10.0
- seaborn==0.13.2
- tqdm==4.66.2
- pandas==2.2.2
- pykan==0.0.3
- xgboost==2.0.3
- fair-esm == 2.0.0
- matplotlib==3.6.2
- scikit-learn==1.1.3 

## Files

1.dataset: dataset.fasta store the protein sequence information of the training set, validation set, and test set.
2. src: a.Model.py：the PLMVF framework；b.main.py: training model saves the optimal parameters of the model; c.predict.py: Prediction of the model.
3. TM_Predictor:a.model.py: the TM-Predictor framework；b.main.py: training model saves the optimal parameters of the model; c.util.py: data processing functions.


## Train and Predict

First, obtain the predicted TM score features using TM_predictor and get the sequence features using ESM-2. Then, concatenate these two types of features and input them into train.py for training. Finally, use predict.py to make predictions.
