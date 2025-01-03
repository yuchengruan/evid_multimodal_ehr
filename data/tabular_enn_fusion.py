import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np




class MIMIC_Structured_Notes(Dataset):
    def __init__(self, data_path, split, hparams = None) -> None:
        super().__init__()

        data_cont_path = os.path.join(data_path, split + "_cont.csv")
        data_cat_path = os.path.join(data_path, split + "_cat.csv")

        if hparams.pretrained_model == "emilyalsentzer/Bio_ClinicalBERT":
            data_note_path = os.path.join(data_path, split + "_onenote_Bio_ClinicalBERT.npy")
        elif hparams.pretrained_model == "dmis-lab/biobert-v1.1":
            data_note_path = os.path.join(data_path, split + "_onenote_biobert-v1.1.npy")
        elif hparams.pretrained_model == "google-bert/bert-base-uncased":
            data_note_path = os.path.join(data_path, split + "_onenote_bert-base-uncased.npy")
            
        data_label_path = os.path.join(data_path, split + "_label.csv")

        self.cont = pd.read_csv(data_cont_path)
        self.cat = pd.read_csv(data_cat_path)
        self.notes = np.load(data_note_path).squeeze()

        labels = pd.read_csv(data_label_path)
        if "mortality" in hparams.dataset:
            self.label = labels["icu_death"]
        elif "los" in hparams.dataset:
            self.label = labels["long_icu"]

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        
        cont_data = self.cont.iloc[index].values.astype("float32")
        cat_data = self.cat.iloc[index].values.astype("float32")
        notes_data = self.notes[index]
        label_data = self.label.iloc[index]


        return (cont_data, cat_data, notes_data), label_data


