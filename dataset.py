import os
from torch.utils.data import Dataset, DataLoader

class AVA(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.audio_paths = []
        self.labels = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        label_mapping = {
            'NO_SPEECH': 0,
            'CLEAN_SPEECH': 1,
            'SPEECH_WITH_MUSIC': 1,
            'SPEECH_WITH_NOISE': 1
        }

        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                label = label_mapping.get(folder_name, -1)
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.wav'):
                        self.audio_paths.append(os.path.join(folder_path, file_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        return audio_path, label

if __name__ == "__main__":
    test_dataset = AVA('/share/nas165/aaronelyu/Datasets/AVA-speech/')
    # audio_path, label = test_dataset[0]
    # print(f'Test dataset size: {len(test_dataset)}')
    # print(f'Audio path: {audio_path}, Label: {label}')

    test_loader = DataLoader(test_dataset, batch_size=1)
    for batch in test_loader:
        print(f'Batch: {batch}')
        audio_path, label = batch
        print(f"Audio Path: {audio_path[0]}")
        print(f"Label: {label.item()}")
        break
