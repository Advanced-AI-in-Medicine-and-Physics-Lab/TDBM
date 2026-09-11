"""Datasets for TDBM.

Three loaders are provided:

``PairedDataset``
    Synthetic training pairs produced by the physics-based data engine.  Every
    sample contains two contrast-filled images that share the *same* background
    but carry *different* vessel trees, the shared contrast-free background, and
    both vessel masks.  This background-invariant pairing is what makes the
    vessel-aware contrastive loss possible.

``ImageSequenceDataset``
    Real XCA sequences used for the annealed real-data term.  The first frame of
    a sequence (before contrast arrives) acts as the contrast-free target and the
    later frames as contrast-filled inputs.

``ImageDataset``
    Inference-time loader that walks a directory tree of frames and returns a
    temporal window of ``2n + 1`` neighbouring frames per sample.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import Dataset
from torchvision.transforms import RandomErasing
from torchvision.transforms import functional as F

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class ImageDataset(Dataset):
    """Return each frame of a directory tree together with its temporal neighbours.

    Sub-directories are treated as independent sequences: neighbours are only
    taken from within the same folder, and sequences are padded by repeating the
    first/last frame at the boundaries.

    Args:
        image_dir: Root directory; each sub-directory is one sequence.
        n: Half-width of the temporal window, so each sample has ``2n + 1`` frames.
        transform: Torchvision transform applied to every frame.
    """

    def __init__(self, image_dir, n=1, transform=None):
        self.image_dir = image_dir
        self.n = n
        self.transform = transform
        self.image_paths = []
        self.folder_to_images = {}

        for root, _, files in os.walk(image_dir):
            files = sorted(f for f in files if f.lower().endswith(IMAGE_EXTENSIONS))
            full_paths = [os.path.join(root, f) for f in files]
            if full_paths:
                self.folder_to_images[root] = full_paths
                self.image_paths.extend(full_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        target_path = self.image_paths[idx]
        folder = os.path.dirname(target_path)
        image_list = self.folder_to_images[folder]
        pos_in_folder = image_list.index(target_path)

        start = max(0, pos_in_folder - self.n)
        end = min(len(image_list), pos_in_folder + self.n + 1)
        selected = image_list[start:end]

        # Pad at the sequence boundaries by repeating the edge frame.
        while len(selected) < 2 * self.n + 1:
            if start == 0:
                selected.insert(0, selected[0])
            else:
                selected.append(selected[-1])

        images = [Image.open(p).convert("L") for p in selected]
        if self.transform:
            images = [self.transform(img) for img in images]

        return torch.cat(images, dim=0), target_path


class PairedDataset(Dataset):
    """Background-invariant synthetic pairs from the physics-based data engine.

    The dataset directory is expected to hold one sub-directory per image role::

        <root>/image1        contrast-filled image, vessel tree A
        <root>/image2        contrast-filled image, vessel tree B (same background)
        <root>/label         vessel mask of image1
        <root>/label2        vessel mask of image2
        <root>/image1_ori    shared contrast-free background
        <root>/image2_ori    (optional) second background
        <root>/label_ori     (optional) unaugmented mask of image1
        <root>/label2_ori    (optional) unaugmented mask of image2

    File names must match across directories.  Missing optional roles are
    returned as ``None`` and skipped by the collate step at training time.

    When ``channel > 1`` the two contrast-filled images are expanded into a
    temporal window of ``channel`` frames.  Each frame receives random erasing
    (the temporal augmentation of the paper, which forces the model to recover
    occluded content from neighbouring frames) and every non-centre frame is
    additionally warped elastically to mimic cardiac and respiratory motion.

    Args:
        path: Dataset root directory.
        size: Target ``(height, width)`` after cropping.
        transform: If truthy, apply the random train-time augmentation pipeline;
            otherwise centre-crop only.
        channel: Number of frames in the temporal window (``2*delta + 1``).
        overall_aug: If True, augment the whole frame; if False, restrict the
            per-frame augmentation to the vessel region.
    """

    ROLES = (
        ("image1", "image1"),
        ("image2", "image2"),
        ("label1", "label"),
        ("label2", "label2"),
        ("image1_ori", "image1_ori"),
        ("image2_ori", "image2_ori"),
        ("label1_ori", "label_ori"),
        ("label2_ori", "label2_ori"),
    )

    REQUIRED_ROLES = ("image1", "image2", "label1", "label2", "image1_ori")

    def __init__(self, path, size=(512, 512), transform=None, channel=1, overall_aug=False):
        self.root = path
        self.dirs = {key: os.path.join(path, sub) for key, sub in self.ROLES}

        missing = [role for role in self.REQUIRED_ROLES if not os.path.isdir(self.dirs[role])]
        if missing:
            raise FileNotFoundError(
                f"{path} is missing required sub-directories for roles {missing}. "
                "See docs/DATA.md for the expected layout."
            )

        self.image_names = sorted(os.listdir(self.dirs["image1"]))
        self.transform = transform
        self.size = tuple(size)
        self.channel = channel
        self.overall_aug = overall_aug

    def __len__(self):
        return len(self.image_names)

    # ------------------------------------------------------------- helpers

    @staticmethod
    def apply_padding(img, pad):
        return None if img is None else F.pad(img, padding=pad, fill=0)

    def resize_if_needed(self, img):
        """Upsample images that are smaller than the crop size."""
        if img is None:
            return None
        w, h = img.size
        target_h, target_w = self.size
        if h < target_h or w < target_w:
            return F.resize(img, self.size)
        return img

    @staticmethod
    def generate_elastic_field(image_shape, alpha=1.0, sigma=10.0):
        """Return a smooth random displacement field ``(dx, dy)``.

        Args:
            image_shape: ``(height, width)`` of the field.
            alpha: Displacement magnitude.
            sigma: Standard deviation of the Gaussian smoothing kernel.
        """
        random_state = np.random.RandomState(None)
        dx = random_state.uniform(-1, 1, image_shape) * alpha
        dy = random_state.uniform(-1, 1, image_shape) * alpha
        dx = gaussian_filter(dx, sigma, mode="constant", cval=0)
        dy = gaussian_filter(dy, sigma, mode="constant", cval=0)
        return dx, dy

    def apply_elastic_transform(self, image, alpha=1.0, sigma=10.0):
        """Warp a 2-D array with a smooth random displacement field."""
        shape = image.shape
        dx, dy = self.generate_elastic_field(shape, alpha, sigma)
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        distorted_x = np.clip(x + dx, 0, shape[1] - 1)
        distorted_y = np.clip(y + dy, 0, shape[0] - 1)
        return map_coordinates(image, [distorted_y, distorted_x], order=1, mode="reflect")

    # ------------------------------------------------------------ sampling

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        data = {}
        for key, _ in self.ROLES:
            path = os.path.join(self.dirs[key], image_name)
            data[key] = Image.open(path).convert("L") if os.path.exists(path) else None

        data = {k: self.resize_if_needed(v) for k, v in data.items()}
        present = [k for k, v in data.items() if v is not None]

        if self.transform:
            # The same padding / crop / flip is applied to every role so that the
            # images and masks stay pixel-aligned.
            pad = random.randint(0, 20)
            data = {k: self.apply_padding(v, pad) for k, v in data.items()}

            if all(data[k] is not None for k in ("image1", "image2", "label1")):
                i, j, h, w = transforms.RandomCrop.get_params(data["image1"], output_size=self.size)
                for key in present:
                    data[key] = F.crop(data[key], i, j, h, w)

                if random.random() > 0.5:
                    for key in present:
                        data[key] = F.hflip(data[key])

            for key in present:
                data[key] = F.to_tensor(data[key])
        else:
            for key in present:
                data[key] = F.to_tensor(F.center_crop(data[key], self.size))

        if self.channel > 1 and self.transform:
            data["image1"], data["image2"] = self._build_temporal_window(data)

        return {k: v for k, v in data.items() if v is not None}

    def _build_temporal_window(self, data):
        """Expand ``image1``/``image2`` into a ``self.channel``-frame temporal window."""
        mask1 = (data["label1"] > 0.1).float()
        mask2 = (data["label2"] > 0.1).float()
        mid_ch = self.channel // 2

        def random_erase(x):
            # Value 0.48 is roughly the mean background intensity of the synthetic
            # frames, so erased regions stay photometrically plausible.
            return RandomErasing(p=1, scale=(0.05, 0.15), ratio=(0.3, 3.3), value=0.48)(x)

        identity = lambda x: x  # noqa: E731

        channels1, channels2 = [], []
        for ch in range(self.channel):
            # 80% of the frames get random erasing; the rest are left untouched.
            func = random_erase if random.random() < 0.8 else identity

            erased1 = func(data["image1"])
            erased2 = func(data["image2"])

            if self.overall_aug:
                frame1, frame2 = erased1, erased2
            else:
                frame1 = data["image1"] * (1 - mask1) + erased1 * mask1
                frame2 = data["image2"] * (1 - mask2) + erased2 * mask2

            # Non-centre frames are warped elastically to simulate inter-frame motion;
            # the centre frame stays aligned with the target background.
            if ch != mid_ch:
                alpha = np.random.uniform(80.0, 200.0)
                sigma = np.random.uniform(4.0, 12.0)
                frame1 = torch.from_numpy(
                    self.apply_elastic_transform(frame1.squeeze(0).numpy(), alpha=alpha, sigma=sigma)
                ).unsqueeze(0).float()
                frame2 = torch.from_numpy(
                    self.apply_elastic_transform(frame2.squeeze(0).numpy(), alpha=alpha, sigma=sigma)
                ).unsqueeze(0).float()

            channels1.append(frame1)
            channels2.append(frame2)

        return torch.cat(channels1, dim=0), torch.cat(channels2, dim=0)


class ImageSequenceDataset(Dataset):
    """Real XCA sequences paired with a contrast-free reference frame.

    Expects the layout::

        <data_root>/image/<sequence>/*.png    contrast-filled frames
        <data_root>/label/<sequence>/*.png    contrast-free reference (single file)

    Args:
        data_root: Directory containing the ``image`` and ``label`` folders.
        num_images: Size of the temporal window, taken from the centre of each
            sequence.  Must be odd.
        image_size: Target ``(height, width)``.
        mode: ``"train"`` enables padding/cropping/flipping augmentation.
    """

    def __init__(self, data_root, num_images=7, image_size=(256, 256), mode="train"):
        if num_images % 2 == 0:
            raise ValueError(f"num_images must be odd, got {num_images}")

        self.data_root = Path(data_root)
        self.num_images = num_images
        self.image_size = image_size
        self.mode = mode

        self.data_pairs = self._scan_data()
        if not self.data_pairs:
            raise ValueError(f"No valid data found in {data_root}")
        print(f"Found {len(self.data_pairs)} data pairs in {mode} mode")

    # ------------------------------------------------------------- helpers

    @staticmethod
    def apply_padding(img, pad):
        return None if img is None else F.pad(img, padding=pad, fill=0)

    def _scan_data(self):
        """Match each image sub-folder with its contrast-free reference frame."""
        image_root = self.data_root / "image"
        label_root = self.data_root / "label"
        if not image_root.exists() or not label_root.exists():
            raise ValueError(f"'image' or 'label' folder not found in {self.data_root}")

        data_pairs = []
        for image_folder in sorted(image_root.iterdir()):
            if not image_folder.is_dir():
                continue

            label_folder = label_root / image_folder.name
            if not label_folder.exists():
                print(f"Warning: no corresponding label folder for {image_folder.name}")
                continue

            image_files = self._get_image_files(image_folder)
            if len(image_files) < self.num_images:
                print(
                    f"Warning: {image_folder.name} has only {len(image_files)} frames, "
                    f"need {self.num_images}"
                )
                continue

            label_files = self._get_image_files(label_folder)
            if not label_files:
                print(f"Warning: no label file found in {label_folder}")
                continue

            data_pairs.append(
                {
                    "image_folder": image_folder,
                    "image_files": image_files,
                    "label_file": label_files[0],
                    "folder_name": image_folder.name,
                }
            )
        return data_pairs

    @staticmethod
    def _get_image_files(folder):
        files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        return sorted(files)

    def _select_center_images(self, image_files):
        """Take ``num_images`` frames centred on the middle of the sequence."""
        total = len(image_files)
        center = total // 2
        half = self.num_images // 2
        indices = [max(0, min(center + offset, total - 1)) for offset in range(-half, half + 1)]
        return [image_files[i] for i in indices]

    @staticmethod
    def _load_image(image_path):
        try:
            image = Image.open(image_path)
            return image if image.mode == "L" else image.convert("L")
        except (OSError, ValueError) as exc:
            print(f"Error loading image {image_path}: {exc}")
            return Image.new("L", (256, 256), 0)

    def _apply_transforms(self, images, label, flip_horizontal=False):
        """Resize, then optionally pad/crop/flip images and label identically."""
        resize = transforms.Resize(self.image_size)
        images = [resize(img) for img in images]
        label = resize(label)

        if self.mode == "train":
            pad = random.randint(0, 20)
            images = [self.apply_padding(img, pad) for img in images]
            label = self.apply_padding(label, pad)

            output_size = (
                (self.image_size, self.image_size)
                if isinstance(self.image_size, int)
                else self.image_size
            )
            i, j, h, w = transforms.RandomCrop.get_params(images[0], output_size=output_size)
            images = [F.crop(img, i, j, h, w) for img in images]
            label = F.crop(label, i, j, h, w)

            if flip_horizontal:
                images = [F.hflip(img) for img in images]
                label = F.hflip(label)

        return [F.to_tensor(img) for img in images], F.to_tensor(label)

    # ------------------------------------------------------------ sampling

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        data_pair = self.data_pairs[idx]
        selected = self._select_center_images(data_pair["image_files"])

        images = [self._load_image(f) for f in selected]
        label = self._load_image(data_pair["label_file"])

        flip_horizontal = self.mode == "train" and random.random() > 0.5
        image_tensors, label_tensor = self._apply_transforms(images, label, flip_horizontal)

        return {
            "image": torch.cat(image_tensors, dim=0),  # (num_images, H, W)
            "label": label_tensor,  # (1, H, W)
            "folder_name": data_pair["folder_name"],
            "num_images": self.num_images,
        }
