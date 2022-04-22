# import os
# import torch
# import torch.utils.data as data
# import nltk
#
# from PIL import Image
# from pycocotools.coco import COCO


# class COCODataset(data.Dataset):
#     def __init__(self, cocoroot, json, vocab, transform=None):
#         self.cocoroot = cocoroot
#         self.coco = COCO(json)
#         self.ids = list(self.coco.anns.keys())
#         self.vocab = vocab
#         self.transform =transform
#
#     def __getitem__(self, item):
#         ann_id = self.ids[item]
#         image_id = self.coco.anns[ann_id]['image_id']
#         caption = self.coco.anns[ann_id]['caption']
#         path = self.coco.loadImgs(image_id)[0]['file_name']
#
#         image = Image.open(os.path.join(self.cocoroot, path)).convert('RGB')
#         if self.transform is not None:
#             image = self.transform(image)
#
#         tokens = nltk.word_tokenize(str(caption).lower())
#         caption = []
#         caption.append(self.vocab['START'])
#         caption.extend([self.vocab[token] for token in tokens])
#         caption.append(self.vocab['END'])
#         caption = torch.Tensor(caption)
#         return image, caption
#
#     def __len__(self):
#         return len(self.ids)
#
#
# def collate_fn(data):                       # ([batch_size, 3, 224, 224], [batch_size, seqlen])
#     data.sort(key=lambda x: len(x[1]), reverse=True)
#     images, captions = zip(*data)           # images: [torch[2, 224, 224], torch[], ..., torch[]]   captions: [torch[len], ..., torch[]]
#
#     images = torch.stack(images, 0)         # images: torch[batch_size, 2, 224, 224]
#     length = [len(caption) for caption in captions]
#     padding_result = torch.zeros(len(captions), max(length)).long()     # padding_result: torch[batch_size, max_seq_len]
#     for i, caption in enumerate(captions):
#         end = length[i]
#         padding_result[i, :end] = caption[:end]
#     return images, padding_result, length
#
#
# def get_loader(cocoroot, json, vocab, transform, batch_size, shuffle, num_workers):
#     coco = COCODataset(cocoroot, json, vocab, transform)
#     COCODataloader = data.DataLoader(dataset=coco, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn())
#
#     return COCODataloader


import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


def get_loader(data_folder, data_name, split, transform, batch_size, shuffle, num_workers):
    data_loader = DataLoader(dataset=CaptionDataset(data_folder, data_name, split, transform), batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers)

    return data_loader
