import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch
import numpy as np
from collections import Counter

# --- CLASSE DATASET ---
class RAFCEDataset(Dataset):
    def __init__(self, img_dir, label_file, partition_file, split='train', transform=None, return_dict=True):
        self.img_dir = img_dir
        self.transform = transform
        self.return_dict = return_dict

        # Lecture des fichiers
        df_partition = pd.read_csv(partition_file, sep=' ', header=None, names=['image_name', 'partition_label'])
        df_labels = pd.read_csv(label_file, sep=' ', header=None, names=['image_name', 'emotion_label'])
        
        # Fusion
        df_merged = pd.merge(df_partition, df_labels, on='image_name')
        split_map = {'train': 0, 'test': 1, 'val': 2}
        target = split_map.get(split, 0)
        self.data_info = df_merged[df_merged['partition_label'] == target].reset_index(drop=True)

        # Mapping ID -> Texte
        self.classes = {
            0: "happily surprised", 1: "happily disgusted", 2: "sadly fearful",
            3: "sadly angry", 4: "sadly surprised", 5: "sadly disgusted",
            6: "fearfully angry", 7: "fearfully surprised", 8: "fearfully disgusted",
            9: "angrily surprised", 10: "angrily disgusted", 11: "disgustedly surprised",
            12: "happily fearful", 13: "happily sad"
        }

    def __len__(self):
        return len(self.data_info)

    def get_labels(self):
        return self.data_info['emotion_label'].values

    def __getitem__(self, idx):
        img_name = self.data_info.iloc[idx]['image_name']
        label = int(self.data_info.iloc[idx]['emotion_label'])
        label_text = self.classes.get(label, "unknown")

        possible_names = [img_name, img_name + ".jpg", img_name.replace(".jpg", "") + "_aligned.jpg"]
        img_path = None
        for name in possible_names:
            temp_path = os.path.join(self.img_dir, name)
            if os.path.exists(temp_path):
                img_path = temp_path
                break

        image = Image.open(img_path).convert('RGB') if img_path else Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)

        if self.return_dict:
            return {"image": image, "label": label, "label_text": label_text}
        else:
            return image, label

def get_dataloaders(img_dir, label_file, partition_file, batch_size=32, use_sampler=True):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_ds = RAFCEDataset(img_dir, label_file, partition_file, 'train', train_transform)
    val_ds = RAFCEDataset(img_dir, label_file, partition_file, 'val', eval_transform)
    test_ds = RAFCEDataset(img_dir, label_file, partition_file, 'test', eval_transform)

    sampler = None
    shuffle = True
    if use_sampler:
        labels = train_ds.get_labels()
        counts = Counter(labels)
        weights = torch.tensor([1.0/counts[l] for l in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(weights))
        shuffle = False

    return (DataLoader(train_ds, batch_size, shuffle=shuffle, sampler=sampler),
            DataLoader(val_ds, batch_size, shuffle=False),
            DataLoader(test_ds, batch_size, shuffle=False))
