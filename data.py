"""Fixed SUIM splits, paired labels, and paper-defined saliency."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from saliency import rgb_to_saliency


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def file_hash(path):
    """Calculate the SHA-256 hash of a file."""
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def pairs_from_folders(images, masks):
    """Match images and masks using case-insensitive filename stems."""

    def index(directory):
        directory = Path(directory)

        if not directory.is_dir():
            raise ValueError(
                f"Missing directory: {directory}. "
                f"Set the corresponding path in experiment.json."
            )

        found = {}

        for path in sorted(directory.iterdir()):
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                key = path.stem.casefold()

                if key in found:
                    raise ValueError(
                        f"Duplicate filename stem: "
                        f"{key} in {directory}"
                    )

                found[key] = path

        return found

    image_index = index(images)
    mask_index = index(masks)

    if not image_index:
        raise ValueError(
            f"No supported image files found in {images}"
        )

    missing_masks = sorted(
        image_index.keys() - mask_index.keys()
    )
    masks_without_images = sorted(
        mask_index.keys() - image_index.keys()
    )

    if missing_masks or masks_without_images:
        raise ValueError(
            "Image/mask filename stems do not match. "
            f"Missing masks: {missing_masks[:8]}; "
            f"masks without images: "
            f"{masks_without_images[:8]}"
        )

    return [
        (image_index[key], mask_index[key])
        for key in sorted(image_index)
    ]


def read_pair(image_path, mask_path):
    """
    Read one SUIM image and its combined RGB segmentation mask.

    The official SUIM collection includes some image/mask pairs
    stored at different resolutions. This is accepted because the
    Dataset resizes both to the same configured training size.

    Images use bilinear resizing. Binary masks use nearest-neighbor
    resizing.
    """
    image_path = Path(image_path)
    mask_path = Path(mask_path)

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.load()
    except Exception as error:
        raise ValueError(
            f"Could not read image {image_path}: {error}"
        ) from error

    try:
        with Image.open(mask_path) as source:
            if source.mode not in ("RGB", "RGBA", "P"):
                raise ValueError(
                    f"{mask_path}: expected an original combined "
                    f"RGB mask, but found mode {source.mode}."
                )

            if "transparency" in source.info:
                raise ValueError(
                    f"{mask_path}: transparent palette labels "
                    f"are unsupported."
                )

            if (
                source.mode == "RGBA"
                and source.getextrema()[-1] != (255, 255)
            ):
                raise ValueError(
                    f"{mask_path}: transparent labels "
                    f"are unsupported."
                )

            mask_rgb = np.asarray(
                source.convert("RGB"),
                dtype=np.uint8,
            )

    except Exception as error:
        if isinstance(error, ValueError):
            raise

        raise ValueError(
            f"Could not read mask {mask_path}: {error}"
        ) from error

    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError(
            f"{mask_path}: expected an RGB mask with shape "
            f"[height, width, 3], found {mask_rgb.shape}."
        )

    target = rgb_to_saliency(mask_rgb)

    if target.ndim != 2:
        raise ValueError(
            f"{mask_path}: saliency conversion returned "
            f"unexpected shape {target.shape}."
        )

    if not np.isfinite(target).all():
        raise ValueError(
            f"{mask_path}: saliency mask contains "
            f"non-finite values."
        )

    unique_values = set(
        np.unique(target).astype(float).tolist()
    )

    if not unique_values.issubset({0.0, 1.0}):
        raise ValueError(
            f"{mask_path}: expected binary saliency values, "
            f"found {sorted(unique_values)}."
        )

    target_image = Image.fromarray(
        target.astype(np.uint8),
        mode="L",
    )

    return image, target_image


class SUIM(Dataset):
    """SUIM dataset for single-channel semantic saliency."""

    def __init__(self, records, size, augment=False):
        self.records = records
        self.size = int(size)
        self.augment = bool(augment)

        if self.size < 1:
            raise ValueError(
                "Dataset image size must be positive."
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image, target = read_pair(
            record["image"],
            record["mask"],
        )

        image = image.resize(
            (self.size, self.size),
            Image.Resampling.BILINEAR,
        )

        target = target.resize(
            (self.size, self.size),
            Image.Resampling.NEAREST,
        )

        if self.augment and random.random() < 0.5:
            image = image.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )
            target = target.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

        image_array = (
            np.asarray(image, dtype=np.float32)
            .transpose(2, 0, 1)
            / 255.0
        )

        target_array = np.asarray(
            target,
            dtype=np.float32,
        )[None]

        # Nearest-neighbor resizing should preserve binary labels.
        target_array = (
            target_array >= 0.5
        ).astype(np.float32)

        return (
            torch.from_numpy(image_array.copy()),
            torch.from_numpy(target_array.copy()),
        )


def seed_worker(_):
    """Give each DataLoader worker deterministic random seeds."""
    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loader(
    records,
    cfg,
    seed,
    train=False,
    batch_size=None,
    workers=None,
):
    """Construct a deterministic DataLoader."""
    selected_batch_size = (
        batch_size
        if batch_size is not None
        else cfg["batch_size"]
    )

    selected_workers = (
        workers
        if workers is not None
        else cfg["workers"]
    )

    return DataLoader(
        SUIM(
            records,
            cfg["image_size"],
            augment=train,
        ),
        batch_size=selected_batch_size,
        shuffle=train,
        num_workers=selected_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=seed_worker,
    )


def prepare_data(cfg, directory):
    """
    Validate SUIM and create fixed group-aware experiment splits.

    Exact duplicate images inside train_val are kept together so
    identical inputs cannot appear in both training and validation.

    Exact duplicates crossing train_val and TEST are rejected because
    they would compromise held-out evaluation.
    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"

    if manifest_path.exists():
        raise ValueError(
            f"{manifest_path} already exists. Reuse it or "
            f"choose a new run_root; do not silently resplit."
        )

    root = Path(cfg["dataset_root"])

    train_pairs = pairs_from_folders(
        root / cfg["train_images"],
        root / cfg["train_masks"],
    )

    test_pairs = pairs_from_folders(
        root / cfg["test_images"],
        root / cfg["test_masks"],
    )

    train_records = []
    test_records = []

    # Tracks where each exact image first appeared.
    seen_images = {}

    # Exact images appearing in both train_val and TEST must
    # be excluded from train_val. TEST remains untouched.
    cross_split_duplicates = []

    split_sources = [
        (
            "train_val",
            train_pairs,
            train_records,
            True,
        ),
        (
            "test",
            test_pairs,
            test_records,
            False,
        ),
    ]

    for (
        source_split,
        pairs,
        destination,
        inspect_labels,
    ) in split_sources:
        for image_path, mask_path in pairs:
            image_sha256 = file_hash(image_path)
            mask_sha256 = file_hash(mask_path)

            previous = seen_images.get(image_sha256)

            if (
                previous is not None
                and previous["split"] != source_split
            ):
                cross_split_duplicates.append(
                    {
                        "image_sha256": image_sha256,
                        "train_val_image": (
                            previous["path"]
                            if previous["split"] == "train_val"
                            else str(image_path)
                        ),
                        "test_image": (
                            str(image_path)
                            if source_split == "test"
                            else previous["path"]
                        ),
                        "action": (
                            "Excluded all matching train_val "
                            "records; retained TEST record."
                        ),
                    }
                )

            if previous is None:
                seen_images[image_sha256] = {
                    "split": source_split,
                    "path": str(image_path),
                }

            record = {
                "id": image_path.stem,
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
                "image_sha256": image_sha256,
                "mask_sha256": mask_sha256,

                # All records with this value must stay together.
                "duplicate_group_sha256": image_sha256,
            }

            if inspect_labels:
                image, target = read_pair(
                    image_path,
                    mask_path,
                )

                target_array = np.asarray(
                    target,
                    dtype=np.uint8,
                )

                record["has_saliency"] = bool(
                    target_array.any()
                )

                record["original_image_size"] = list(
                    image.size
                )

                record["original_mask_size"] = list(
                    target.size
                )

                record["stored_dimensions_match"] = (
                    image.size == target.size
                )

            destination.append(record)

    # Preserve TEST as the held-out benchmark. Any byte-identical
    # train_val image is excluded from model development.
    test_image_hashes = {
        record["image_sha256"]
        for record in test_records
    }

    excluded_train_records = [
        record
        for record in train_records
        if record["image_sha256"] in test_image_hashes
    ]

    train_records = [
        record
        for record in train_records
        if record["image_sha256"] not in test_image_hashes
    ]

    if not train_records:
        raise ValueError(
            "All train_val images overlap TEST after exact "
            "duplicate filtering."
        )

    # Confirm that filtering removed every exact cross-split
    # image duplicate.
    remaining_train_hashes = {
        record["image_sha256"]
        for record in train_records
    }

    remaining_cross_split_overlap = (
        remaining_train_hashes
        & test_image_hashes
    )

    if remaining_cross_split_overlap:
        raise RuntimeError(
            "Internal filtering error: exact image hashes "
            "still cross train_val and TEST."
        )

    # Build groups of byte-identical training images.
    groups_by_hash = {}

    for record in train_records:
        group_hash = record[
            "duplicate_group_sha256"
        ]

        groups_by_hash.setdefault(
            group_hash,
            [],
        ).append(record)

    duplicate_groups = {
        group_hash: records
        for group_hash, records
        in groups_by_hash.items()
        if len(records) > 1
    }

    duplicate_record_count = sum(
        len(records)
        for records in duplicate_groups.values()
    )

    # Verify whether byte-identical inputs have equivalent
    # saliency labels. Conflicting groups are unusable for
    # supervised learning and are excluded in full.
    comparison_size = cfg["image_size"]

    conflicting_duplicate_groups = []
    conflicting_duplicate_hashes = set()

    for group_hash, records in duplicate_groups.items():
        reference_record = records[0]

        _, reference_mask = read_pair(
            reference_record["image"],
            reference_record["mask"],
        )

        reference_mask = reference_mask.resize(
            (comparison_size, comparison_size),
            Image.Resampling.NEAREST,
        )

        reference_array = (
            np.asarray(
                reference_mask,
                dtype=np.uint8,
            )
            >= 1
        )

        conflicting_record = None

        for record in records[1:]:
            _, candidate_mask = read_pair(
                record["image"],
                record["mask"],
            )

            candidate_mask = candidate_mask.resize(
                (comparison_size, comparison_size),
                Image.Resampling.NEAREST,
            )

            candidate_array = (
                np.asarray(
                    candidate_mask,
                    dtype=np.uint8,
                )
                >= 1
            )

            if not np.array_equal(
                reference_array,
                candidate_array,
            ):
                conflicting_record = record
                break

        if conflicting_record is not None:
            conflicting_duplicate_hashes.add(
                group_hash
            )

            conflicting_duplicate_groups.append(
                {
                    "image_sha256": group_hash,
                    "record_count": len(records),
                    "images": [
                        record["image"]
                        for record in records
                    ],
                    "masks": [
                        record["mask"]
                        for record in records
                    ],
                    "first_detected_conflict": {
                        "reference_mask":
                            reference_record["mask"],
                        "conflicting_mask":
                            conflicting_record["mask"],
                    },
                    "action": (
                        "Excluded the complete byte-identical "
                        "image group from training, validation "
                        "and calibration because its saliency "
                        "annotations conflict."
                    ),
                }
            )

    conflicting_duplicate_records = [
        record
        for record in train_records
        if record["image_sha256"]
        in conflicting_duplicate_hashes
    ]

    if conflicting_duplicate_hashes:
        train_records = [
            record
            for record in train_records
            if record["image_sha256"]
            not in conflicting_duplicate_hashes
        ]

        for group_hash in conflicting_duplicate_hashes:
            groups_by_hash.pop(
                group_hash,
                None,
            )

    if not train_records:
        raise ValueError(
            "No train_val records remain after filtering "
            "conflicting duplicate-label groups."
        )

    # Recalculate retained duplicate statistics after filtering.
    duplicate_groups = {
        group_hash: records
        for group_hash, records
        in groups_by_hash.items()
        if len(records) > 1
    }

    duplicate_record_count = sum(
        len(records)
        for records in duplicate_groups.values()
    )

    # Split whole duplicate groups rather than individual records.
    grouped_records = list(
        groups_by_hash.values()
    )

    rng = random.Random(
        cfg["split_seed"]
    )

    train = []
    validation = []

    for positive in (False, True):
        category_groups = [
            records
            for records in grouped_records
            if records[0]["has_saliency"] == positive
        ]

        if positive and len(category_groups) < 2:
            raise ValueError(
                "At least two independent image groups "
                "containing salient objects are required."
            )

        rng.shuffle(category_groups)

        if len(category_groups) > 1:
            validation_group_count = round(
                len(category_groups)
                * cfg["validation_fraction"]
            )

            validation_group_count = max(
                1,
                min(
                    len(category_groups) - 1,
                    validation_group_count,
                ),
            )
        else:
            validation_group_count = 0

        validation_groups = category_groups[
            :validation_group_count
        ]

        training_groups = category_groups[
            validation_group_count:
        ]

        for records in validation_groups:
            validation.extend(records)

        for records in training_groups:
            train.extend(records)

    rng.shuffle(train)
    rng.shuffle(validation)

    train_hashes = {
        record["image_sha256"]
        for record in train
    }

    validation_hashes = {
        record["image_sha256"]
        for record in validation
    }

    overlap = train_hashes & validation_hashes

    if overlap:
        raise RuntimeError(
            "Internal split error: exact duplicate image "
            "groups cross training and validation."
        )

    # Calibration uses one record from each unique training
    # image group so duplicates do not waste calibration slots.
    training_groups_by_hash = {}

    for record in train:
        training_groups_by_hash.setdefault(
            record["image_sha256"],
            [],
        ).append(record)

    unique_training_groups = list(
        training_groups_by_hash.values()
    )

    calibration_count = min(
        cfg["calibration_images"],
        len(unique_training_groups),
    )

    if calibration_count < 1:
        raise ValueError(
            "The training split is empty."
        )

    calibration_rng = random.Random(
        cfg["split_seed"] + 1
    )

    selected_calibration_groups = (
        calibration_rng.sample(
            unique_training_groups,
            calibration_count,
        )
    )

    # Select exactly one representative from each image group.
    calibration = [
        records[0]
        for records in selected_calibration_groups
    ]

    # Ensure calibration contains at least one positive image.
    if not any(
        record["has_saliency"]
        for record in calibration
    ):
        positive_record = next(
            (
                records[0]
                for records
                in unique_training_groups
                if records[0]["has_saliency"]
            ),
            None,
        )

        if positive_record is None:
            raise ValueError(
                "The training split contains no salient pixels."
            )

        calibration[0] = positive_record

    manifest = {
        "schema": 3,
        "split_seed": cfg["split_seed"],

        "saliency_mapping": {
            "BW": 0,
            "HD": 1,
            "PF": 0,
            "WR": 1,
            "RO": 1,
            "RI": 0,
            "FV": 1,
            "SR": 0,
        },

        "train": train,
        "validation": validation,
        "calibration": calibration,
        "test": test_records,

        "duplicate_summary": {
            "unique_train_val_images":
                len(groups_by_hash),

            "duplicate_groups":
                len(duplicate_groups),

            "records_in_duplicate_groups":
                duplicate_record_count,

            "train_validation_hash_overlap":
                len(overlap),

            "cross_split_hash_overlap_after_filtering":
                len(remaining_cross_split_overlap),

            "train_val_records_excluded_for_test_overlap":
                len(excluded_train_records),

            "conflicting_duplicate_groups_excluded":
                len(conflicting_duplicate_groups),

            "records_excluded_for_conflicting_annotations":
                len(conflicting_duplicate_records),
        },

        "cross_split_exclusions":
            cross_split_duplicates,

        "excluded_train_val_records":
            excluded_train_records,

        "conflicting_duplicate_groups_excluded":
            conflicting_duplicate_groups,

        "records_excluded_for_conflicting_annotations":
            conflicting_duplicate_records,

        "note": (
            "TEST remains untouched. Byte-identical images "
            "found in train_val and TEST are excluded from "
            "all model development. "
            "Image-level stratification by saliency presence. "
            "Byte-identical images are grouped so they cannot "
            "cross training and validation. Exact duplicates "
            "crossing train_val and TEST are rejected. "
            "Near-duplicate scenes are not inferred. Stored "
            "image/mask resolutions may differ; both are "
            "resized independently to the configured size."
        ),
    }

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
    )

    preview_records = train[:8]

    if not preview_records:
        raise ValueError(
            "No training records are available for preview."
        )

    preview_dataset = SUIM(
        preview_records,
        cfg["image_size"],
        augment=False,
    )

    panel_size = cfg["image_size"]
    row_height = panel_size + 24

    canvas = Image.new(
        "RGB",
        (
            panel_size * 2,
            row_height * len(preview_dataset),
        ),
    )

    draw = ImageDraw.Draw(canvas)

    for index, (
        image_tensor,
        target_tensor,
    ) in enumerate(preview_dataset):
        top = index * row_height

        draw.text(
            (3, top + 3),
            (
                preview_records[index]["id"]
                + " | paper saliency"
            ),
            fill="white",
        )

        image_array = (
            image_tensor
            .permute(1, 2, 0)
            .numpy()
            * 255.0
        ).clip(
            0,
            255,
        ).astype(np.uint8)

        target_array = (
            target_tensor[0].numpy()
            * 255.0
        ).astype(np.uint8)

        canvas.paste(
            Image.fromarray(
                image_array,
                mode="RGB",
            ),
            (0, top + 24),
        )

        canvas.paste(
            Image.fromarray(
                target_array,
                mode="L",
            ).convert("RGB"),
            (panel_size, top + 24),
        )

    preview_path = (
        directory / "data_preview.png"
    )

    canvas.save(preview_path)

    dimensions_differ = sum(
        not record["stored_dimensions_match"]
        for record in train_records
    )

    summary = {
        "train": len(train),
        "validation": len(validation),
        "calibration": len(calibration),
        "test": len(test_records),

        "unique_train_val_images":
            len(groups_by_hash),

        "duplicate_groups":
            len(duplicate_groups),

        "records_in_duplicate_groups":
            duplicate_record_count,

        "train_validation_duplicate_overlap":
            len(overlap),

        "cross_split_overlap_after_filtering":
            len(remaining_cross_split_overlap),

        "train_val_records_excluded_for_test_overlap":
            len(excluded_train_records),

        "cross_split_duplicate_occurrences":
            len(cross_split_duplicates),

        "conflicting_duplicate_groups_excluded":
            len(conflicting_duplicate_groups),

        "records_excluded_for_conflicting_annotations":
            len(conflicting_duplicate_records),

        "train_val_pairs_with_different_stored_dimensions":
            dimensions_differ,

        "manifest": str(manifest_path),
        "preview": str(preview_path),
    }

    print(
        json.dumps(
            summary,
            indent=2,
        ),
        flush=True,
    )

    return manifest


def read_manifest(cfg, verify=True):
    """Read and optionally verify the fixed experiment manifest."""
    path = Path(cfg["run_root"]) / "manifest.json"

    if not path.is_file():
        raise ValueError(
            "Run the prepare command first."
        )

    manifest = json.loads(
        path.read_text()
    )

    if verify:
        for split in (
            "train",
            "validation",
            "test",
        ):
            for record in manifest[split]:
                for kind in ("image", "mask"):
                    current_hash = file_hash(
                        record[kind]
                    )

                    expected_hash = record[
                        kind + "_sha256"
                    ]

                    if current_hash != expected_hash:
                        raise ValueError(
                            "Dataset changed since preparation: "
                            f"{record[kind]}"
                        )

    return manifest, file_hash(path)
