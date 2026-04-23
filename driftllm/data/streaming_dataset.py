from torch.utils.data import IterableDataset


class StreamingHFDataset(IterableDataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __iter__(self):
        for row in self.dataset:
            yield row
