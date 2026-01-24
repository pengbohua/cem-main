"""
Dataloader the Cifar10 Dataset
"""
import numpy as np
import os
import torch
import torchvision
from torchvision.datasets import CIFAR10
from scipy.special import softmax
from pytorch_lightning import seed_everything
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
import csv

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

N_CLASSES = 10
DATASET_DIR = os.environ.get("DATASET_DIR", 'cem/data/cifar10/')

CONCEPT_SEMANTICS = [
    "Hunter",
    "barn",
    "beak",
    "bed",
    "bed for carrying cargo",
    "beetle",
    "bird feeder",
    "birdfeeder",
    "birdhouse",
    "bit",
    "boat",
    "branch",
    "bridle",
    "cab for the driver",
    "cage",
    "captain",
    "cat bed",
    "cat food dish",
    "cat toy",
    "collar",
    "copilot",
    "crew",
    "dashboard",
    "deck",
    "deer stand",
    "dock",
    "dog bowl",
    "driver",
    "field",
    "flat back end",
    "flat front and back",
    "flight attendant",
    "fly",
    "food bowl",
    "forest",
    "four-legged mammal",
    "gear shift",
    "green or brown body",
    "grille",
    "halter",
    "hitch",
    "large body",
    "large size",
    "large, metal body",
    "large, rectangular body",
    "lead rope",
    "leaf",
    "leash",
    "lily pad",
    "litter box",
    "long neck",
    "mane and tail",
    "mast",
    "mosquito",
    "nest",
    "net",
    "nose",
    "passenger",
    "pasture",
    "pedal",
    "person",
    "pilot",
    "pointed front end",
    "pond",
    "port",
    "reddish-brown coat",
    "reins",
    "rider",
    "rifle",
    "road",
    "rudder",
    "runway",
    "saddle",
    "scratching post",
    "seat",
    "seatbelt",
    "short, stocky build",
    "slender body",
    "small, lithe body",
    "spider",
    "steering wheel",
    "suitcase",
    "tail",
    "tire",
    "tough, durable frame",
    "toy",
    "trailer",
    "tree",
    "wet nose",
    "wheel",
    "wide mouth",
    "windshield",
    "worm",
    "amphibian",
    "an engine",
    "animal",
    "antlers",
    "cargo",
    "engines",
    "engines on the wings",
    "feathers",
    "food",
    "four legs",
    "four round, black tires",
    "four wheels",
    "fur",
    "grass",
    "hooves",
    "insects",
    "landing gear",
    "large sails or engines",
    "large, brown eyes",
    "large, bulging eyes",
    "lily pads",
    "living thing",
    "long ears",
    "long hind legs for jumping",
    "long, thin legs",
    "machine",
    "mammal",
    "multiple decks",
    "multiple sails",
    "nature",
    "object",
    "organism",
    "passengers",
    "pointed ears",
    "quadruped",
    "side windows",
    "taillights",
    "the ability to fly",
    "the ocean",
    "transportation",
    "trees",
    "two headlights",
    "vertebrate",
    "vessel",
    "watercraft",
    "webbed feet",
    "whiskers",
    "white spots on the coat",
    "wings",
    "woods",
]


########################################################
## Dataset Loader
########################################################
def identity(x):
    return x

class Cifar10Dataset(Dataset):
    """
    Cifar10 dataset
    """

    def __init__(
        self,
        root_dir,
        split='train',
        concept_transform=None,
        additional_sample_transform=None,
        binarization_mode='clustering',
        zero_shot_scale=1000,
        threshold=None,
        regenerate=False,
        template="",
    ):
        self.root_dir = root_dir
        self.split = split
        assert split in ['train', 'test', 'val']
        if concept_transform is None:
            concept_transform = identity
        self.concept_transform = concept_transform
        self.binarization_mode = binarization_mode
        self.zero_shot_scale = zero_shot_scale
        self.threshold = threshold

        self.transform = additional_sample_transform

        if not os.path.exists(self.root_dir):
            raise ValueError(
                f'{self.root_dir} does not exist yet. Please generate the '
                f'dataset first.'
            )

        self._download_data_and_splits()
        self._concepts = self._process_vit_concepts(
            split=split,
            regenerate=True,
            template=template,
        )

    def _download_data_and_splits(self):
        self._inner_dataset = torchvision.datasets.CIFAR10(
            root=os.path.expanduser(self.root_dir),
            download=True,
            train=self.split in ['train', 'val'],
            transform=transforms.Compose(
                (
                    [
                        transforms.ToTensor(),
                        transforms.ConvertImageDtype(torch.float32),
                    ] +
                    ([self.transform] if self.transform is not None else []) +
                    [
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
            ),
        )
        split_path = os.path.join(self.root_dir, f'{self.split}_idxs.npy')
        self.idx_remap = list(range(len(self._inner_dataset)))
        if self.split in ['train', 'val']:
            if os.path.exists(split_path):
                self.idx_remap = np.load(split_path)
            else:
                rng = np.random.default_rng(42)
                val_idxs = sorted(
                    rng.choice(
                        len(self._inner_dataset),
                        size=int(np.ceil(len(self._inner_dataset) * 0.2)),
                        replace=False,
                    )
                )
                if self.split == 'val':
                    self.idx_remap = val_idxs
                else:
                    exclude_set = set(val_idxs)
                    self.idx_remap = [
                        i for i in range(len(self._inner_dataset))
                        if i not in exclude_set
                    ]
                np.save(split_path, self.idx_remap)

    def _process_vit_concepts(
        self,
        split='train',
        regenerate=False,
        template="",
    ):
        template = template + "_" if template else ""
        used_split = 'train' if split == 'val' else split
        if regenerate or not os.path.exists(
            os.path.join(
                self.root_dir,
                f'{template}c_{used_split}_{self.binarization_mode}_bool.pt',
            )
        ):

            if self.binarization_mode == 'zero_shot':
                # Then we will use the zero-shot classifier to make concept
                # labels
                concept_scores = torch.tensor(torch.load(
                    os.path.join(self.root_dir, f'cifar10_concepts_{used_split}.pt')
                )).float()
                concept_scores = concept_scores.numpy()
                n_concepts = concept_scores.shape[1]//2
                cluster_assignments = np.zeros(
                    (concept_scores.shape[0], n_concepts)
                )
                # n_samples, n_concept
                for concept_idx in range(n_concepts):
                    pos_scores = concept_scores[:, 2*concept_idx:2*concept_idx+1]
                    neg_scores = concept_scores[:, 2*concept_idx + 1:2*concept_idx + 2]
                    cluster_assignments[:, concept_idx] = np.argmax(
                        np.concatenate([neg_scores, pos_scores], axis=-1),
                        axis=-1
                    )
                c = torch.tensor(cluster_assignments).float()
                if self.threshold is not None:
                    c = (c >= self.threshold).float()

            elif self.binarization_mode == 'percentile_threshold':
                # Let's find the normalization factors
                all_scores = np.concatenate(
                    [
                        torch.tensor(torch.load(
                            os.path.join(
                                self.root_dir,
                                f'{template}c_{s}.pt'
                            )
                        )).float().detach().cpu().numpy()
                        for s in ['train', 'test']
                    ],
                    axis=0,
                )
                thresh = self.threshold or 50
                split_values = torch.tensor(np.expand_dims(
                    np.percentile(all_scores, thresh, axis=0),
                    axis=0,
                ))
                c = torch.tensor(torch.load(
                    os.path.join(self.root_dir, f'{template}c_{used_split}.pt')
                ) >= split_values).float()
            else:
                raise ValueError(
                    f'Unsupported binarization mode {self.binarization_mode}'
                )

            torch.save(
                c,
                os.path.join(
                    self.root_dir,
                    f'{template}c_{used_split}_{self.binarization_mode}_bool.pt',
                ),
            )
        else:
            c = torch.tensor(torch.load(
                os.path.join(
                    self.root_dir,
                    f'{template}c_{used_split}_{self.binarization_mode}_bool.pt',
                ),
            ))
        print(f"[Split {split}] Concept distribution is: {list(np.mean(c.detach().cpu().numpy(), axis=0))}")
        return c

    def __len__(self):
        return len(self.idx_remap)

    def __getitem__(self, idx):
        x, y = self._inner_dataset[self.idx_remap[idx]]
        c = self._concepts[self.idx_remap[idx]]
        if self.concept_transform is not None:
            c = self.concept_transform(c)
        return x, y, c

def load_data(
    split,
    batch_size,
    root_dir='data/cifar10/',
    num_workers=1,
    dataset_transform=identity,
    dataset_size=None,
    concept_transform=None,
    additional_sample_transform=None,
    binarization_mode='zero_shot',
    zero_shot_scale=1000,
    threshold=None,
    regenerate=False,
    template="",
):
    """
    Loader for Cifar10 dataset as described in MixCEM's paper.
    """
    dataset = Cifar10Dataset(
        split=split,
        root_dir=root_dir,
        concept_transform=concept_transform,
        additional_sample_transform=additional_sample_transform,
        binarization_mode=binarization_mode,
        zero_shot_scale=zero_shot_scale,
        threshold=threshold,
        regenerate=regenerate,
        template=template,
    )
    if dataset_size is not None:
        # Then we will subsample this training set so that after splitting
        # into a training, test, and validation set we end up with
        # `dataset_size` samples in the training dataset
        train_idxs = np.random.permutation(len(dataset))
        if dataset_size > 0 and dataset_size < 1:
            # Then this is a fraction (function of the total size of the set!)
            dataset_size = int(
                np.ceil(len(dataset) * dataset_size)
            )
        train_idxs = train_idxs[:int(np.ceil(dataset_size))]
        dataset = Subset(
            dataset,
            train_idxs,
        )
    if split == 'train':
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    loader = DataLoader(
        dataset_transform(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return loader


########################
## Data Module
########################

def get_num_labels(*args, **kwargs):
    """
    This call is needed to satisfy the loader API used in this codebase.

    Returns:
        int: the number of class labels used in the waterbirds dataset.
    """
    return N_CLASSES

def get_num_attributes(*args, **kwargs):
    """
    This call is needed to satisfy the loader API used in this codebase.

    Returns:
        int: the number of attributes in the waterbirds dataset.
    """
    return len(kwargs.get('selected_attributes', CONCEPT_SEMANTICS))

def generate_data(
    config,
    root_dir=DATASET_DIR,
    seed=42,
    output_dataset_vars=False,
    rerun=False,
    dataset_transform=identity,
    training_transform=None,

    train_sample_transform=None,
    test_sample_transform=None,
    val_sample_transform=None,
    binarization_mode='clustering',
    zero_shot_scale=1000,
    threshold=None,
    regenerate=False,
    selected_concepts=None,
    template='',
):
    if root_dir is None:
        root_dir = DATASET_DIR
    seed_everything(seed)

    training_transform = (
        training_transform if training_transform is not None
        else dataset_transform
    )


    sampling_percent = config.get("sampling_percent", 1)
    num_workers = config.get('num_workers', 8)
    dataset_size = config.get('dataset_size', None)
    batch_size = config.get('batch_size', 32)
    selected_concepts = config.get('selected_concepts', selected_concepts)
    binarization_mode = config.get(
        'binarization_mode',
        binarization_mode,
    )
    zero_shot_scale = config.get('zero_shot_scale', zero_shot_scale)
    regenerate = config.get('regenerate', regenerate)
    template = config.get('template', template)
    threshold = config.get('threshold', threshold)

    n_concepts = get_num_attributes(**config)
    concept_group_map = None

    if selected_concepts is not None:
        for idx, concept_name in enumerate(selected_concepts):
            if isinstance(concept_name, str):
                selected_concepts[idx] = CONCEPT_SEMANTICS.index(concept_name)

        def concept_transform(sample):
            if isinstance(sample, list):
                sample = np.array(sample)
            return sample[selected_concepts]
        n_concepts = len(selected_concepts)
    elif sampling_percent != 1:
        # Do the subsampling
        new_n_concepts = int(np.ceil(n_concepts * sampling_percent))
        selected_concepts_file = os.path.join(
            root_dir,
            f"selected_concepts_sampling_{sampling_percent}.npy",
        )
        if (not rerun) and os.path.exists(selected_concepts_file):
            selected_concepts = np.load(selected_concepts_file)
        else:
            selected_concepts = sorted(
                np.random.permutation(n_concepts)[:new_n_concepts]
            )
            np.save(selected_concepts_file, selected_concepts)

        def concept_transform(sample):
            if isinstance(sample, list):
                sample = np.array(sample)
            return sample[selected_concepts]

        n_concepts = len(selected_concepts)
    else:
        concept_transform = None

    train_dl = load_data(
        split='train',
        batch_size=batch_size,
        root_dir=root_dir,
        num_workers=num_workers,
        dataset_transform=training_transform,
        dataset_size=dataset_size,
        concept_transform=concept_transform,
        additional_sample_transform=train_sample_transform,
        binarization_mode=binarization_mode,
        zero_shot_scale=zero_shot_scale,
        threshold=threshold,
        regenerate=regenerate,
        template=template,
    )

    if config.get('weight_loss', False):
        attribute_count = np.zeros((n_concepts,))
        samples_seen = 0
        for i, (_, y, c) in enumerate(train_dl):
            c = c.cpu().detach().numpy()
            attribute_count += np.sum(c, axis=0)
            samples_seen += c.shape[0]
        imbalance = samples_seen / attribute_count - 1
    else:
        imbalance = None

    val_dl = load_data(
        split='val',
        batch_size=batch_size,
        root_dir=root_dir,
        num_workers=num_workers,
        dataset_transform=dataset_transform,
        dataset_size=dataset_size,
        concept_transform=concept_transform,
        additional_sample_transform=val_sample_transform,
        binarization_mode=binarization_mode,
        zero_shot_scale=zero_shot_scale,
        threshold=threshold,
        regenerate=regenerate,
        template=template,
    )

    test_dl = load_data(
        split='test',
        batch_size=batch_size,
        root_dir=root_dir,
        num_workers=num_workers,
        dataset_transform=dataset_transform,
        dataset_size=dataset_size,
        concept_transform=concept_transform,
        additional_sample_transform=test_sample_transform,
        binarization_mode=binarization_mode,
        zero_shot_scale=zero_shot_scale,
        threshold=threshold,
        regenerate=regenerate,
        template=template,
    )

    if not output_dataset_vars:
        return train_dl, val_dl, test_dl
    return (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        (n_concepts, get_num_labels(**config), concept_group_map),
    )


########################################################
## Preprocessing: top-k frequent concepts per class
########################################################

def compute_topk_concepts_of_classes(
    *,
    root_dir=DATASET_DIR,
    split="train",
    binarization_mode="zero_shot",
    threshold=None,
    template="",
    topk_list=(4, 8, 16, 32),
    output_dir="cem/data/cocnepts_of_classes/cifar10/",
):
    """Compute per-class top-k most frequent concepts for CIFAR10.

    Builds a count matrix M of shape (C, K) by iterating over the dataset with
    batch_size=1. For each sample (x_i, y_i, c_i), updates:

        M[y_i, :] += int(c_i)
        T[y_i] += 1

    Then for each k in topk_list, computes top-k over the concept dimension,
    and saves:
        - indices CSV: f"{output_dir}/top{k}_indices.csv"
        - freqs CSV:   f"{output_dir}/top{k}_freqs.csv"

    Frequencies are computed as counts / T[class].

    Returns:
        (M, T): torch tensors with shapes (C,K) and (C,)
    """
    dataset = Cifar10Dataset(
        split=split,
        root_dir=root_dir,
        concept_transform=None,
        additional_sample_transform=None,
        binarization_mode=binarization_mode,
        threshold=threshold,
        regenerate=False,
        template=template,
    )
    with open("cem/data/cifar10/label_names.txt", "r") as f:
        label_names = f.read().splitlines()

    # Iterate with batch_size=1 exactly as requested
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    num_classes = N_CLASSES
    # Infer K from the concept vector
    _, y0, c0 = dataset[0]
    K = int(torch.as_tensor(c0).numel())

    M = torch.zeros((num_classes, K), dtype=torch.long)
    T = torch.zeros((num_classes,), dtype=torch.long)

    for _, y, c in loader:
        # y: (1,) ; c: (1, K)
        y_idx = int(y.reshape(-1)[0].item())
        c_vec = c.long().squeeze(0)
        M[y_idx] += c_vec
        T[y_idx] += 1

    os.makedirs(output_dir, exist_ok=True)
    normalized_M = M/T.unsqueeze(1)
    # Avoid division by zero if a class is missing (shouldn't happen on CIFAR10)
    T_safe = torch.clamp(T, min=1).to(dtype=torch.float32)

    for k in topk_list:
        k = int(k)
        if k <= 0:
            continue
        k_eff = min(k, K)

        # topk per class
        top_counts, top_indices = torch.topk(normalized_M.to(dtype=torch.float32), k_eff, dim=1, largest=True)
        top_freqs = top_counts / T_safe.unsqueeze(1)

        indices_path = os.path.join(output_dir, f"top{k}_indices.csv")
        freqs_path = os.path.join(output_dir, f"top{k}_freqs.csv")


        concept_names = list(CONCEPT_SEMANTICS[:K])

        header = ["class"] + concept_names

        # indices CSV: store the rank (1..k) of each concept if it is in top-k; else empty.
        # freqs CSV: store frequency for each concept if in top-k; else 0.
        rank_matrix = [["" for _ in range(K)] for _ in range(num_classes)]
        freq_matrix = [[0.0 for _ in range(K)] for _ in range(num_classes)]
        for cls in range(num_classes):
            for rank in range(k_eff):
                idx = int(top_indices[cls, rank].item())
                rank_matrix[cls][idx] = int(rank + 1)
                freq_matrix[cls][idx] = float(top_freqs[cls, rank].item())

        with open(indices_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for cls in range(num_classes):
                writer.writerow([label_names[cls]] + rank_matrix[cls])

        with open(freqs_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for cls in range(num_classes):
                writer.writerow([label_names[cls]] + freq_matrix[cls])
        torch.save(top_indices.to(dtype=torch.long), os.path.join(output_dir, f"top{k}_indices.pt"))
    return M, T

# if __name__ == "__main__":
#     m, t = compute_topk_concepts_of_classes(split='train', binarization_mode='zero_shot', output_dir="cem\data\cocnepts_of_classes\cifar10")