from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'data/processed/orand_car'
IMG_HEIGHT, IMG_WIDTH = 64, 512

MAX_LENGTH = 8
SEQ_LENGTH = MAX_LENGTH + 1
AUG_PER_IMAGE = 1  # 0 disables offline augmentation
rng = np.random.default_rng()

# Image preprocessing
def preprocess_image(path, augment=False):
    with Image.open(path) as img:
        if 'A' in img.getbands() or 'transparency' in img.info:
            img = Image.alpha_composite(
                Image.new('RGBA', img.size, 'white'), img.convert('RGBA'))
        img = img.convert('L')

    if augment:
        img = img.rotate(rng.uniform(-3, 3),
                         resample=Image.Resampling.BILINEAR,
                         expand=True, fillcolor=255)

    w, h = img.size
    scale = min(IMG_WIDTH / w, IMG_HEIGHT / h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                     Image.Resampling.BILINEAR)

    # Change brightness and add noise before padding.
    if augment:
        arr = np.asarray(img, dtype=np.float32)
        arr *= rng.uniform(0.9, 1.1)
        arr += rng.normal(0, 2.0, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    x, y = 0, (IMG_HEIGHT - img.height) // 2
    if augment:
        x = int(rng.integers(0, min(8, IMG_WIDTH - img.width) + 1))
        y = int(np.clip(y + rng.integers(-2, 3), 0, IMG_HEIGHT - img.height))

    canvas = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), 255)
    canvas.paste(img, (x, y))
    return np.asarray(canvas, dtype=np.uint8)[..., None]

# Vocabulary and label encoding
char_to_idx = {c: i for i, c in enumerate(['<PAD>', '<SOS>', '<EOS>'] + list('0123456789'))}
PAD, SOS, EOS = (char_to_idx[t] for t in ['<PAD>', '<SOS>', '<EOS>'])
VOCAB_SIZE = len(char_to_idx)

def encode_label(label):
    tokens = [char_to_idx[c] for c in label]
    padding = [PAD] * (MAX_LENGTH - len(tokens))
    return [SOS] + tokens + padding, tokens + [EOS] + padding

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        ROOT / 'data/processed/orand_car_metadata.csv',
        dtype={'label': str},
        keep_default_na=False
    )

    df['image_path'] = (
        df['image_path']
        .str.replace('\\', '/', regex=False)
        .str.removeprefix('../')
    )

    if not df['label'].str.fullmatch(r'[0-9]{2,8}').all():
        raise ValueError('Invalid labels: expected 2–8 digits.')

    if not df['image_path'].is_unique:
        raise ValueError('Duplicate image paths in metadata.')

    missing = ~df['image_path'].map(lambda p: (ROOT / p).is_file())

    if missing.any():
        print(df.loc[missing, 'image_path'].head())
        raise FileNotFoundError(f'{missing.sum()} image files are missing.')

    df = df.drop(columns=['split'], errors='ignore')
    df['label_length'] = df['label'].str.len()

    # Stratified random split (70/15/15); regenerated on every run
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df['label_length'])
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label_length'])

    split_df = pd.concat([
        train_df.assign(split='train'),
        val_df.assign(split='val'),
        test_df.assign(split='test')
    ], ignore_index=True)

    split_df.to_csv(OUTPUT_DIR / 'orand_car_split.csv', index=False)

    # Export arrays; augment training samples only
    for split in ['train', 'val', 'test']:
        part = split_df[split_df['split'] == split].reset_index(drop=True)
        copies = AUG_PER_IMAGE if split == 'train' else 0
        part = pd.concat([part.assign(is_augmented=k > 0, augmentation_id=k)
                          for k in range(copies + 1)], ignore_index=True)
        folder = OUTPUT_DIR / split
        folder.mkdir(exist_ok=True)
        X = np.empty((len(part), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
        inputs = np.empty((len(part), SEQ_LENGTH), dtype=np.int32)
        targets = np.empty_like(inputs)
        for i, row in enumerate(part.itertuples(index=False)):
            X[i] = preprocess_image(ROOT / row.image_path, augment=row.is_augmented)
            inputs[i], targets[i] = encode_label(row.label)
        np.save(folder / 'images.npy', X)
        np.save(folder / 'decoder_inputs.npy', inputs)
        np.save(folder / 'decoder_targets.npy', targets)
        part.to_csv(folder / 'manifest.csv', index=False)
        print(f'{split}: {X.shape}, {inputs.shape}, {targets.shape}')
        del X, inputs, targets

    # Save configuration
    config = {'char_to_idx': char_to_idx, 'PAD': PAD, 'SOS': SOS, 'EOS': EOS,
              'vocab_size': VOCAB_SIZE, 'max_label_length': MAX_LENGTH,
              'sequence_length': SEQ_LENGTH, 'image_shape': [IMG_HEIGHT, IMG_WIDTH, 1],
              'image_dtype': 'uint8', 'normalization': 'float32 / 255.0',
              'augmentation_mode': 'offline_train_only' if AUG_PER_IMAGE else 'none',
              'augmentation_per_train_image': AUG_PER_IMAGE}
    (OUTPUT_DIR / 'config.json').write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
