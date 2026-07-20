from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision as tv
from skimage.color import gray2rgb
from skimage.io import imread
from torch.utils.data import Dataset

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    """Dataset for the solar-cell multi-label classification task."""

    def __init__(self, data: Any, mode: str):
        if mode not in {"train", "val"}:
            raise ValueError(
                f"mode must be either 'train' or 'val', but received {mode!r}"
            )

        if data is None:
            raise ValueError("data must be a pandas DataFrame")

        self.data = data.reset_index(drop=True).copy()
        self.mode = mode
        self.image_column = self._find_image_column()
        self.label_columns = self._find_label_columns()

        if mode == "train":
            self.transform = tv.transforms.Compose(
                [
                    tv.transforms.ToPILImage(),
                    tv.transforms.RandomHorizontalFlip(p=0.5),
                    tv.transforms.RandomVerticalFlip(p=0.5),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(
                        mean=train_mean,
                        std=train_std,
                    ),
                ]
            )
        else:
            self.transform = tv.transforms.Compose(
                [
                    tv.transforms.ToPILImage(),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(
                        mean=train_mean,
                        std=train_std,
                    ),
                ]
            )

    def _find_image_column(self):
        """Determine which DataFrame column contains image paths."""
        preferred_names = (
            "filename",
            "file_name",
            "filepath",
            "file_path",
            "path",
            "image",
            "image_path",
            "img",
        )

        lower_to_original = {
            str(column).lower(): column for column in self.data.columns
        }

        for name in preferred_names:
            if name in lower_to_original:
                return lower_to_original[name]

        for column in self.data.columns:
            if str(column).lower().startswith("unnamed"):
                continue

            values = self.data[column].dropna()
            if values.empty:
                continue

            sample = str(values.iloc[0]).lower()
            if sample.endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            ):
                return column

        non_index_columns = [
            column
            for column in self.data.columns
            if not str(column).lower().startswith("unnamed")
        ]

        if not non_index_columns:
            raise ValueError("Could not find an image-path column")

        return non_index_columns[0]

    def _find_label_columns(self):
        """Determine the two columns containing crack/inactive labels."""
        lower_to_original = {
            str(column).lower(): column for column in self.data.columns
        }

        crack_column = None
        inactive_column = None

        for name, original_name in lower_to_original.items():
            if "crack" in name:
                crack_column = original_name
            if "inactive" in name:
                inactive_column = original_name

        if crack_column is not None and inactive_column is not None:
            return [crack_column, inactive_column]

        candidate_columns = [
            column
            for column in self.data.columns
            if column != self.image_column
            and not str(column).lower().startswith("unnamed")
        ]

        if len(candidate_columns) < 2:
            raise ValueError(
                "The DataFrame must contain an image path and two label "
                "columns"
            )

        return candidate_columns[-2:]

    @staticmethod
    def _resolve_image_path(raw_path):
        path = Path(str(raw_path)).expanduser()

        candidates = [
            path,
            Path.cwd() / path,
            Path(__file__).resolve().parent / path,
        ]

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(
            f"Could not find image {raw_path!r}. Tried: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    @staticmethod
    def _prepare_image(image):
        """Convert an image into an RGB array suitable for ToPILImage."""
        image = np.asarray(image)

        if image.ndim == 2:
            image = gray2rgb(image)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = gray2rgb(image[:, :, 0])
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            elif image.shape[2] != 3:
                raise ValueError(
                    f"Unsupported image shape: {image.shape}"
                )
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if image.dtype == np.bool_:
            image = image.astype(np.uint8) * 255
        elif np.issubdtype(image.dtype, np.integer):
            if image.dtype != np.uint8:
                max_value = np.iinfo(image.dtype).max
                image = (
                    image.astype(np.float32) / float(max_value) * 255.0
                )
                image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        elif np.issubdtype(image.dtype, np.floating):
            image = image.astype(np.float32)
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

            if image.max(initial=0.0) > 1.0:
                image = image / 255.0

            image = np.clip(image, 0.0, 1.0)
        else:
            raise TypeError(f"Unsupported image dtype: {image.dtype}")

        return np.ascontiguousarray(image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()

        row = self.data.iloc[int(index)]

        image_path = self._resolve_image_path(row[self.image_column])
        image = imread(str(image_path))
        image = self._prepare_image(image)
        image = self.transform(image)

        labels = row[self.label_columns].to_numpy(dtype=np.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)

        return image, labels