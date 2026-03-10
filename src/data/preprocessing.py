import pandas as pd 

class CheXpertPreprocessing():
    '''Class perfroms preprocessing on a dataset given the path to the CSV file of the dataset'''
    def __init__(self, path, splits_path):
        self.path = path
        self.splits_path = splits_path

    def read_data(self, split=None):
        '''Read data from path and filter the dataset'''
        # tmp = pd.read_csv(self.splits_path, nrows=0)  # read header only
        # cols = [c for c in tmp.columns if not c.startswith("X_feat_")]
        # splits = pd.read_csv(self.splits_path, usecols=cols)

        print(self.path)
        chexpert_plus = pd.read_csv(self.path)
        # chexpert_plus = chexpert_plus.drop("split", axis=1)
        # splits["path"] = splits["path"].str.replace("CheXpert-v1.0/", "", regex=False)
        # chexpert_plus = chexpert_plus.merge(splits, left_on="path_to_image", right_on="path", how="inner")

        # if split is not None:
        #     chexpert_plus = chexpert_plus[chexpert_plus['split'] == split].reset_index(drop=True)
            
        return chexpert_plus