from typing import List
from torch.utils.data import Dataset
from collator import PretrainDataCollatorWithPadding


class ClickDataset(Dataset):
    def __init__(
        self, dataset: List[List[str]], collator: PretrainDataCollatorWithPadding
    ):
        super().__init__()

        self.dataset = dataset
        self.collator = collator

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> List[str]:

        return self.dataset[index]

    def collate_fn(self, data: List[List[str]]):

        return self.collator([{"items": line} for line in data])
