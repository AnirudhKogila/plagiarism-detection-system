# Plagiarism Detection System


## Steps to Run Source Code

1. **Install Dependencies**: Install all the required packages listed in the `requirements.txt` file.

2. **Data Preparation**: Run the `DataPrep.py` script and provide the file directories of the original `.tsv` files containing the train and test datasets from the ultrabalanced PatentMatch dataset.

3. **Token Generation**: Execute the `TokGen.py` script to clear the data and convert it to index tokens.

4. **Validation Set Generation**: Run the `ValidationSet.py` script to generate a validation dataset from the training dataset.

5. **Baseline Evaluation**: Run the `Baseline.py` script to evaluate the dataset and create the `score.pt` file, which will be used to log the model's performance.

6. **Model Training and Evaluation**: Choose and run any of the model scripts (`CNN.py`, `Transformer.py`)


