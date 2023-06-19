from PIL import Image
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here.
        """

        self.data = []
        label_path = dataset_path + '/labels.csv'

        with open(label_path, mode='r')as file:
            # reading the CSV file
            csv_file = csv.reader(file) # hi2

            i = 0
            for lines in csv_file:

                if i == 0:
                    i = 1
                    continue
                i = i + 1

                img_path = dataset_path + '/' + lines[0]
                img = Image.open(img_path)
                im_tensor = transforms.ToTensor()
                tens = im_tensor(img)

                tup = (tens, label_to_int(lines[1]))
                self.data.append(tup)

            self.length = i - 1

        # raise NotImplementedError('SuperTuxDataset.__init__')

    def __len__(self):
        """
        Your code here
        """
        return self.length

        # raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        return self.data[idx]
        # raise NotImplementedError('SuperTuxDataset.__getitem__')


def label_to_int(label):
    if label == 'background':
        return 0
    elif label == 'kart':
        return 1
    elif label == 'pickup':
        return 2
    elif label == 'nitro':
        return 3
    elif label == 'bomb':
        return 4
    elif label == 'projectile':
        return 5


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
