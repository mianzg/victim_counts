import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

class RawDataset:
    def __init__(self, direc):
        self.header = 0
        self.direc = direc
        self.fnames = []
        self.columns = []
        self.data = None

    def _load_one_table(self, fname):
        """
        Read one table file, supporting csv, tsv, excel sheets

        Param:
        ======
        fname: str, path to table file
        Return:
        df: pandas.DataFrame
        """
        if fname is None:
            return "Empty Data"
        _, ext = os.path.splitext(fname)
        if ext == ".csv":
            df = pd.read_csv(fname, header = self.header)
        elif ext == ".tsv":
            df = pd.read_csv(fname, header = self.header, sep="\t")
        elif ext == ".xls" or ext == ".xlsx":
            df = pd.read_excel(fname, header=self.header)
        else:
            return ValueError("Unsupported file type, has to be csv, xls or xlsx")
        return df

    def load(self):
        """
        Load all associated files
        """
        if len(self.fnames) == 0:
            return None
        else:
            f = os.path.join(self.direc, self.fnames[0])
            final_df = self._load_one_table(f)
            if len(self.fnames) > 1:
                for name in self.fnames[1:]:
                    df = self._load_one_table(os.path.join(self.direc, name))
                    final_df = pd.concat([final_df, df], ignore_index=True)
        if len(self.columns) >0:
            self.data = final_df[self.columns]
        else:
            self.data = final_df

    def get_death_data(self):
        """
        Extract event descriptions and fatality counts
        """
        pass

    def get_injured_data(self):
        """
        Extract event descriptions and injury counts
        """
        pass

    def _get_text(self, textcol, is_available):
        if self.data is None:
            return ValueError("Data is not loaded, please use load() method")
        description = self.data[textcol][is_available]
        return description

    def describe(self, include_zero = False, is_plot=True):
        """
        Display basic statistics

        Param:
        ======
        include_zero: bool, whether to include entries with ground truth zero count
        is_plot: bool, whether to plot distribution
        """
        tot_samples = "Total Number of samples: {}\n".format(self.__len__())
        if include_zero:
            print("The samples include zero counts")
            d_text, deaths = self.get_death_data()
            i_text, injuries = self.get_injured_data()
        else:
            print("The samples do not include zero counts")
            d_text, deaths = self.get_nonzero_death_data()
            i_text, injuries = self.get_nonzero_injured_data()
        deaths_samples = "Total Number of available death samples: {}\n".format(len(deaths))
        deaths_stat = [min(deaths)] + list(np.percentile(deaths, q=[25, 50, 75, 90]))+ [max(deaths), np.mean(deaths)]
        inj_samples = "Total Number of available injury samples: {}\n".format(len(injuries))
        inj_stat = [min(injuries)] + list(np.percentile(injuries, q=[25, 50, 75, 90])) + [max(injuries), np.mean(injuries)]
        print(tot_samples+deaths_samples+inj_samples)
        print("Basic Statistics:\n")
        print("Deaths: min {}  25% {}  median {}  75% {}  90% {} max {}  mean {}\n".format(deaths_stat[0], deaths_stat[1], deaths_stat[2], deaths_stat[3], deaths_stat[4], deaths_stat[5], deaths_stat[6]))
        print("Injuries: min {}  25% {}  median {}  75% {}  90% {} max {}  mean {}\n".format(inj_stat[0], inj_stat[1], inj_stat[2], inj_stat[3], inj_stat[4], inj_stat[5], inj_stat[6]))
        # Info on text data #TODO
        if is_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            sns.histplot(deaths, ax=ax1)
            sns.histplot(injuries, ax=ax2)
            ax1.set_xscale("log")
            ax2.set_xscale("log")

    def get_nonzero_death_data(self):
        descriptions, labels = self.get_death_data()
        is_nonzero = labels > 0
        return descriptions[is_nonzero], labels[is_nonzero]

    def get_nonzero_injured_data(self):
        descriptions, labels = self.get_injured_data()
        is_nonzero = labels > 0
        return descriptions[is_nonzero], labels[is_nonzero]

    def __len__(self):
        return self.data.shape[0]

class WAD(RawDataset):
    def __init__(self, direc) -> None:
        super().__init__(direc)
        self.name = "WAD" # Dataset name to be used in experiments
        self.header = 2 # Row index of the header row
        self.fnames = ["pitf.world.19950101-20121231.xls", \
                       "pitf.world.20130101-20151231.xls", \
                       "pitf.world.20160101-20200229.xlsx"] # List of raw data files
        # Columns of interest to keep
        ## At least you need to have a column of event desciptions (here is "Description")
        ## For training, you also need a column of victim counts truth labels (here are "Deaths Number" and "Injured Number")
        self.columns = ['Event Type', 'Campaign Identifier', 'Start Year', 'Deaths Number', 'Injured Number', 'Description']

    def get_death_data(self):
        return self._get_descrpition_and_label("Deaths Number")

    def get_injured_data(self):
        return self._get_descrpition_and_label("Injured Number")

    def _get_descrpition_and_label(self, colname):
        """
        Extract event descriptions and victim count labels (by colname)

        Return:
        =======
        description: pandas.Series
        labels: numpy array
        """
        if self.data is  None:
            self.load()
        f1 = self.data[colname].apply(lambda x: type(x) is int)
        f2 = self.data[colname].notna()
        filter = f1 & f2
        labels = self.data[colname][filter].astype(int).to_numpy()
        description = self._get_text("Description", filter)
        return description, labels

# The following data classes pre-process the event descriptions into Question-Answering data
# used in different task formulations

class QADataset(Dataset):
    def __init__(self, name, queries, labels, tokenizer=None):
        self.name = name
        self.queries = queries
        self.labels = labels
        self.str_labels = [str(l) for l in self.labels]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx]

class ClfDataset(QADataset):
    """
    QA Dataset where the original labels are converted into ordinal classes
    """
    def __init__(self, name, queries, labels, num_classes):
        super().__init__(name, queries, labels)
        self.num_classes = num_classes
        self._map_label_to_class()
        self.classes = [i for i in range(num_classes)]

    def _map_label_to_class(self):
        # zero, some, many
        new_labels = []
        if self.num_classes == 11:
            for label in self.labels:
                l = int(label)
                if 0 <= l and l < 10:
                    new_labels.append(l)
                else:
                    new_labels.append(10)
        elif self.num_classes == 3:
            for label in self.labels:
                l = int(label)
                if l <=3:
                    new_labels.append(0)
                elif l > 3 and l <= 10:
                    new_labels.append(1)
                else:
                    new_labels.append(2)
        # elif self.num_classes == 5
        self.labels = new_labels

class QACollator():
    def __init__(self, tokenizer, device, is_str_label=False, is_int_label=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = 512
        self.is_str_label = is_str_label
        self.is_int_label = is_int_label

    def __call__(self, batch):
        tokenized_queries = self.tokenizer([example[0] for example in batch], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = tokenized_queries['input_ids']
        attention_mask = tokenized_queries['attention_mask']
        if self.is_str_label:
            labels = self.tokenizer([example[1] for example in batch], padding=True, truncation=True, max_length=10, return_tensors="pt")["input_ids"]
            labels[labels[:,:]==0] = -100
        elif self.is_int_label:
            labels = torch.LongTensor([example[1] for example in batch])
        else:
            labels = torch.FloatTensor([example[1] for example in batch])
        return {
            'input_ids': input_ids.to(self.device),
            'attention_mask': attention_mask.to(self.device), #Genbert does not use
            'labels': labels.to(self.device),
        }

def split_data(data, ratios=[0.7,0.1,0.2]):
    """
    Split data in train, validation and test datasets
    """
    assert sum(ratios) == 1
    tot_len = len(data)
    train_length = int(tot_len * ratios[0])
    dev_length = int(tot_len * ratios[1])
    test_length = tot_len - train_length - dev_length
    train_set, dev_set, test_set = random_split(data, [train_length, dev_length, test_length],
                                                generator=torch.Generator().manual_seed(42))
    return train_set, dev_set, test_set
