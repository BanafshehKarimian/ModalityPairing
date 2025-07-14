from datasets import load_dataset 
from torch.utils.data import Dataset, DataLoader
import torch
class TCGAText(Dataset):
    def __init__(self, tissue_type = None):
        clinical_dataset = load_dataset("Lab-Rasool/TCGA", "clinical", split="gatortron")
        self.tissue = clinical_dataset['tissue_or_organ_of_origin']
        self.clinical_ID = clinical_dataset['case_submitter_id']
        pathology_report_dataset = load_dataset("Lab-Rasool/TCGA", "pathology_report", split="gatortron")
        self.ID = pathology_report_dataset['PatientID']
        if tissue_type:
            path_ids = self.get_ids(tissue_type.lower())
            pathology_report_dataset = pathology_report_dataset.select(path_ids)
        self.report_text = pathology_report_dataset['report_text']
        self.embedding = pathology_report_dataset['embedding']
        #self.report_text.append("A histology image")

    def get_ids(self, tissue_type):
        ids = []
        for i in range(len(self.tissue)):
            if str(self.tissue[i]).lower().find(tissue_type) != -1:
                ids.append(self.clinical_ID[i])
        path_ids = []
        for i in range(len(self.ID)):
            if self.ID[i] in ids:
                path_ids.append(i)
        return path_ids
            
    def __getitem__(self, idx, embed = False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if embed:
            return self.embedding[idx]
        return self.report_text[idx]