import json
import os
import pathlib
from typing import List, Tuple
import random
from tqdm import tqdm


class AudioCap:

    @staticmethod
    def train_test_split(dataset: List[str], test_split: float = 0.2) -> Tuple[List[str], List[str]]:
        train_data = random.sample(dataset, k=int(len(dataset) * (1 - test_split)))
        test_data = list(set(dataset) - set(train_data))
        return train_data, test_data

    @staticmethod
    def from_zalo(data_path: str, output_path: str, test_split: float = 0.2):
        pathlib.Path(os.path.join(output_path)).mkdir(parents=True, exist_ok=True)

        dataset = json.load(open(f"{data_path}/train.json", 'r'))
        train_ids, test_ids = AudioCap.train_test_split(list(dataset.keys()), test_split)
        for split in ["train", "valid"]:
            with open(f'{output_path}/{split}.json', 'w') as outfile:
                for sound_file_id in tqdm(train_ids if split == "train" else test_ids):
                    json.dump(
                        {
                            "dataset": "zalo",
                            "location": os.path.join(data_path, "audio", sound_file_id),
                            "captions": dataset[sound_file_id]
                        },
                        outfile
                    )
                    outfile.write('\n')


AudioCap.from_zalo("data/zalo", "data/zalo_converted", 0.2)
