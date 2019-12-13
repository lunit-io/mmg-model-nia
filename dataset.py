from torch.utils.data import Dataset
import numpy as np
import cv2


def _read_image(image_path):
    # Read image (mmg : 16 bit image, dbt : 8 bit image)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    image = image.astype(np.float32) / np.iinfo(image.dtype).max

    return image


def _read_mask(image_size, lesions):
    mask = np.zeros(image_size, np.uint8)
    for lesion in lesions:
        if lesion['label'] != 2:
            continue
        contour = np.asarray(lesion['contour'])
        mask = cv2.drawContours(mask, [contour], 0, color=255, thickness=-1)
    mask = mask > 0
    return mask


def _input_normalization(image, stats):
    # Normalization
    image -= stats['mean']
    image /= stats['std']
    return image


class ImageDataset(Dataset):
    def __init__(self, data, args, use_compressed=False):
        self._data = data
        self._image_size = tuple(args.image_size)
        self._map_size = args.map_size
        self._use_compressed = use_compressed
        self.stats = {'mean': 0., 'std': 1.}

        assert len(self._data) > 0, "Data read from pickle failed or too few data"
        assert (not self._use_compressed) or 'compressed_image_path' in self._data[0].keys(), \
                "Dataset requires compressed images, but pickle doesn't have them."
        assert (self._use_compressed) or 'uncompressed_image_path' in self._data[0].keys(), \
                "Dataset requires uncompressed images, but pickle doesn't have them."

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        image = _read_image(
            self._data[index]['compressed_image_path'] if self._use_compressed
            else self._data[index]['uncompressed_image_path']
        )
        image = _input_normalization(image, self.stats)
        mask_cancer = _read_mask(self._data[index]['image_size'], self._data[index]['lesions'])

        image = cv2.resize(image, self._image_size)

        mask_cancer = mask_cancer.astype(np.uint8)
        mask_cancer = cv2.resize(mask_cancer, self._map_size, interpolation=cv2.INTER_NEAREST)
        mask_cancer = mask_cancer.astype(np.float32)

        image = np.expand_dims(image, 0)

        return image, mask_cancer


class EvalImageDataset(ImageDataset):
    def __getitem__(self, index):
        image, mask_cancer = super(EvalImageDataset, self).__getitem__(index)
        return image, mask_cancer, self._data[index]['case_id'], self._data[index]['case_label']
