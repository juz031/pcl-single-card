import os
import shutil
from tqdm import tqdm
from random import sample


def all_img():
    train_dir = "./train"
    val_dir = "./val"
    img_dir = "./img"

    os.makedirs(img_dir, exist_ok=True)

    cats = sorted(os.listdir(train_dir))
    for cat in tqdm(cats):
        os.makedirs(os.path.join(img_dir, cat), exist_ok=True)
        old_train_dir = os.path.join(train_dir, cat)
        old_val_dir = os.path.join(val_dir, cat)
        new_dir = os.path.join(img_dir, cat)
        for img in os.listdir(old_train_dir):
            shutil.copy2(os.path.join(old_train_dir, img), os.path.join(new_dir, img))
        for img in os.listdir(old_val_dir):
            shutil.copy2(os.path.join(old_val_dir, img), os.path.join(new_dir, img))
        new_img = os.listdir(new_dir)
        print(len(new_img))

    for cat in cats:
        n_imgs = len(os.listdir(os.path.join(img_dir, cat)))
        print(n_imgs)


def split(data_dir, save_dir, ratio):
    img_dir = os.path.join(data_dir, 'img')
    shape_dir = os.path.join(data_dir, 'shape')

    save_train_dir = os.path.join(save_dir, 'train')
    save_val_dir = os.path.join(save_dir, 'val')
    save_shape_dir = os.path.join(save_dir, 'shape')
    
    cats = sorted(os.listdir(img_dir))
    for cat in tqdm(cats):
        os.makedirs(os.path.join(save_train_dir, cat), exist_ok=True)
        os.makedirs(os.path.join(save_val_dir, cat), exist_ok=True)
        os.makedirs(os.path.join(save_shape_dir, cat), exist_ok=True)
        img_list = os.listdir(os.path.join(img_dir, cat))
        print(f'Total imgs: {len(img_list)}')
        n_imgs = len(img_list)
        n_train = int(ratio * n_imgs)
        train_list = sample(img_list, n_train)
        val_list = list(set(img_list) - set(train_list))
        for train_img in train_list:
            shutil.copy2(os.path.join(img_dir, cat, train_img), os.path.join(save_train_dir, cat, train_img))
            shutil.copy2(os.path.join(shape_dir, cat, train_img), os.path.join(save_shape_dir, cat, train_img))
        for val_img in val_list:
            shutil.copy2(os.path.join(img_dir, cat, val_img), os.path.join(save_val_dir, cat, val_img))
    for cat in cats:
        print(cat)
        train_img_list = os.listdir(os.path.join(save_train_dir, cat))
        val_img_list = os.listdir(os.path.join(save_val_dir, cat))
        shape_img_list = os.listdir(os.path.join(save_shape_dir, cat))
        print(len(train_img_list))
        print(len(val_img_list))
        print(len(shape_img_list))

def split_original(data_dir, save_dir, ratio):
    train_dir = os.path.join(data_dir, 'train')
    # train_shape_dir = os.path.join(data_dir, 'shape')
    val_dir = os.path.join(data_dir, 'val')
    # val_shape_dir = os.path.join(data_dir, 'val_shape')

    save_train_dir = os.path.join(save_dir, 'train')
    save_val_dir = os.path.join(save_dir, 'val')
    save_shape_dir = os.path.join(save_dir, 'shape')
    
    cats = sorted(os.listdir(train_dir))
    for cat in tqdm(cat):
        os.makedirs(os.path.join(save_train_dir, cat), exist_ok=True)
        os.makedirs(os.path.join(save_val_dir, cat), exist_ok=True)
        os.makedirs(os.path.join(save_shape_dir, cat), exist_ok=True)
        img_list = []
        for root, dirs, files in os.walk(os.path.join(train_dir, cat)):
            for file in files:
                img_list.append(os.path.join(root, file))
        for root, dirs, files in os.walk(os.path.join(val_dir, cat)):
            for file in files:
                img_list.append(os.path.join(root, file))
        n_imgs = len(img_list)
        print(f'Total imgs: {n_imgs}')
        n_train = int(ratio * n_imgs)
        train_list = sample(img_list, n_train)
        val_list = list(set(img_list) - set(train_list))
        for train_img in train_list:
            img_name = train_img.split('/')[-1]
            shutil.copy2(train_img, os.path.join(save_train_dir, cat, img_name))
            shape_img = train_img.replace('train', 'shape', 1)
            shape_img = shape_img.replace('val', 'val_shape', 1)
            shape_img = shape_img.replace('JPEG', 'jpg', 1)
            shutil.copy2(shape_img, os.path.join(save_shape_dir, cat, img_name))
        for val_img in val_list:
            img_name = val_img.split('/')[-1]
            shutil.copy2(val_img, os.path.join(save_val_dir, cat, img_name))

    for cat in cats:
        print(cat)
        train_img_list = os.listdir(os.path.join(save_train_dir, cat))
        val_img_list = os.listdir(os.path.join(save_val_dir, cat))
        shape_img_list = os.listdir(os.path.join(save_shape_dir, cat))
        print(len(train_img_list))
        print(len(val_img_list))
        print(len(shape_img_list))


if __name__ == "__main__":
    # all_img()

    data_dir = "/user_data/junruz/imagenet"
    save_dir = "/user_data/junruz/IN100_73/3"
    ratio = 0.7

    split_original(data_dir, save_dir, ratio)