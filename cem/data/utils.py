import numpy as np
import torch
import torchvision.transforms as v2


################################################################################
## DATASET MANIPULATION
################################################################################


def gauss_noise_tensor(img, sigma):
    assert isinstance(img, torch.Tensor), type(img).__name__
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)/255.0

    out = img + sigma * torch.randn_like(img)

    if out.dtype != dtype:
        out = out.to(dtype)
    return out

def salt_and_pepper_noise_tensor(
    img,
    s_vs_p=0.5,
    amount=0.05,
    all_channels=False,
):
    assert isinstance(img, torch.Tensor), type(img).__name__
    dtype = img.dtype
    max_elem, min_elem = 1.0, 0.0
    if not img.is_floating_point():
        img = img.to(torch.float32)/255.0

    out = img + 0.0

    # Salt mode
    shape_to_use = img.shape[:-1] if all_channels else img.shape
    num_salt = np.ceil(amount * np.prod(tuple(shape_to_use)) * s_vs_p)
    coords = [
        np.random.randint(0, i - 1, int(num_salt))
        for i in tuple(shape_to_use)
    ]
    if all_channels:
        out[coords, :] = max_elem
    else:
        out[coords] = max_elem


    # Pepper mode
    num_pepper = np.ceil(amount* np.prod(tuple(shape_to_use)) * (1. - s_vs_p))
    coords = [
        np.random.randint(0, i - 1, int(num_pepper))
        for i in tuple(shape_to_use)
    ]
    if all_channels:
        out[coords, :] = min_elem
    else:
        out[coords] = min_elem

    return out

def harder_salt_and_pepper_noise_tensor(
    img,
    s_vs_p=0.5,
    amount=0.05,
    all_channels=False,
):
    if len(img.shape) == 4:
        # Then there is a batch dimension to consider! Let's apply it to all
        # elements independently and them put them back together
        return torch.concat(
            [
                salt_and_pepper_noise_tensor(
                    img[idx, :, :, :],
                    s_vs_p=s_vs_p,
                    amount=amount,
                    all_channels=all_channels,
                ).unsqueeze(0)
                for idx in range(img.shape[0])
            ]
        )
    assert isinstance(img, torch.Tensor), type(img).__name__
    dtype = img.dtype
    max_elem, min_elem = 1.0, 0.0
    if not img.is_floating_point():
        img = img.to(torch.float32)/255.0

    out = img + 0.0

    # Salt mode
    shape_to_use = tuple(img.shape[:-1]) if all_channels else tuple(img.shape)
    num_corrupted = int(np.ceil(amount * np.prod(tuple(shape_to_use))))
    total_pixels = np.prod(tuple(shape_to_use))
    selected_indices = np.random.choice(
        total_pixels,
        size=num_corrupted,
        replace=False,
    )

    new_values = np.random.choice(
        [max_elem, min_elem],
        size=num_corrupted,
        p=[s_vs_p, 1 - s_vs_p],
    )
    if num_corrupted:
        if all_channels:
            h_idxs, w_idxs = np.unravel_index(selected_indices, shape_to_use)
            out[h_idxs, w_idxs, :] = torch.FloatTensor(new_values)
        else:
            h_idxs, w_idxs, c_idxs = np.unravel_index(selected_indices, shape_to_use)
            out[h_idxs, w_idxs, c_idxs] = torch.FloatTensor(new_values)
    return out

class LambdaDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        outputs = self.subset[index]
        if isinstance(outputs, (list, tuple)):
            x = outputs[0]
        else:
            x = outputs
        if self.transform:
            x = self.transform(x)
        if isinstance(outputs, (list, tuple)):
            return (x, *outputs[1:])
        return x

    def __len__(self):
        return len(self.subset)


class IdentityTransform:
    def __call__(self, x):
        return x


class GaussianNoiseTransform:
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, x):
        return gauss_noise_tensor(x, sigma=self.sigma)


class SaltAndPepperTransform:
    def __init__(self, s_vs_p=0.5, amount=0.05, all_channels=False):
        self.s_vs_p = s_vs_p
        self.amount = amount
        self.all_channels = all_channels

    def __call__(self, x):
        return salt_and_pepper_noise_tensor(
            x,
            s_vs_p=self.s_vs_p,
            amount=self.amount,
            all_channels=self.all_channels,
        )


class HarderSaltAndPepperTransform:
    def __init__(self, s_vs_p=0.5, amount=0.05, all_channels=False):
        self.s_vs_p = s_vs_p
        self.amount = amount
        self.all_channels = all_channels

    def __call__(self, x):
        return harder_salt_and_pepper_noise_tensor(
            x,
            s_vs_p=self.s_vs_p,
            amount=self.amount,
            all_channels=self.all_channels,
        )


class RandomNoiseTransform:
    def __init__(self, noise_level=0.0, low_noise_level=0.0):
        self.noise_level = noise_level
        self.low_noise_level = low_noise_level

    def __call__(self, x):
        if self.noise_level <= 0.0:
            return x

        mask = np.random.choice(
            [0, 1],
            size=x.shape,
            p=[1 - self.noise_level, self.noise_level],
        )
        mask = torch.as_tensor(mask, device=x.device, dtype=x.dtype)

        substitutes = np.random.uniform(
            low=0,
            high=self.low_noise_level,
            size=x.shape,
        )
        substitutes = torch.as_tensor(
            substitutes, device=x.device, dtype=x.dtype
        )

        return mask * substitutes + (1 - mask) * x


def transform_from_config(transform):
    if isinstance(transform, list):
        return v2.Compose([
            transform_from_config(t) for t in transform
        ])

    if transform is None or transform == {}:
        return IdentityTransform()

    transform_name = transform["name"].lower().strip()

    if transform_name == "identity":
        return IdentityTransform()

    if transform_name in ["gaussian_noise", "gaussiannoise"]:
        return GaussianNoiseTransform(
            sigma=transform.get("sigma", 1)
        )

    if transform_name in ["salt_and_pepper", "s&p", "saltandpepper"]:
        return SaltAndPepperTransform(
            s_vs_p=transform.get("s_vs_p", 0.5),
            amount=transform.get("amount", 0.05),
            all_channels=transform.get("all_channels", False),
        )

    if transform_name in [
        "harder_salt_and_pepper",
        "harder_s&p",
        "harder_saltandpepper",
    ]:
        return HarderSaltAndPepperTransform(
            s_vs_p=transform.get("s_vs_p", 0.5),
            amount=transform.get("amount", 0.05),
            all_channels=transform.get("all_channels", False),
        )

    if transform_name == "random_noise":
        return RandomNoiseTransform(
            noise_level=transform.get("noise_level", 0.0),
            low_noise_level=transform.get("low_noise_level", 0.0),
        )

    if transform_name == "randomapply":
        return v2.RandomApply(
            transforms=[
                transform_from_config(t)
                for t in transform["transforms"]
            ],
            p=transform["p"],
        )

    if transform_name == "normalize":
        return v2.Normalize(
            mean=transform["mean"],
            std=transform["std"],
        )

    raise ValueError(f"Unsupported transformation {transform_name}")


