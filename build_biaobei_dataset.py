"""Read, split and save the biaobei dataset for our model"""

import csv
import os
import sys


def load_dataset(path_txt):
    """Loads dataset into memory from txt file"""
    with open(path_txt, 'r') as f:
        lines = f.readlines()
        dataset = []
        for line in lines:
            sent = []
            tag = []
            pos = []
            pairs = line.strip().split(' ')
            for pair in pairs:
                items = pair.split('_')
                sent.append(items[0])
                tag.append(items[1])
                pos.append(items[2])
            dataset.append((sent, tag, pos))
    return dataset


def save_dataset(dataset, save_dir):
    """Writes sentences.txt and labels.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences:
        with open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
            with open(os.path.join(save_dir, 'pos.txt'), 'w') as file_pos:
                for words, tags, pos in dataset:
                    file_sentences.write("{}\n".format(" ".join(words)))
                    file_labels.write("{}\n".format(" ".join(tags)))
                    file_pos.write("{}\n".format(" ".join(pos)))
    print("- done.")


if __name__ == "__main__":

    path_dataset = 'data/biaobei/final_tag_2_pos.txt'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg

    # Load the dataset into memory
    print("Loading biaobei dataset into memory...")
    dataset = load_dataset(path_dataset)
    print("- done.")

    # Split the dataset into train, val and split (dummy split with no shuffle)
    train_dataset = dataset[:int(0.7*len(dataset))]
    val_dataset = dataset[int(0.7*len(dataset)) : int(0.85*len(dataset))]
    test_dataset = dataset[int(0.85*len(dataset)):]

    # Save the datasets to files
    save_dataset(train_dataset, 'data/biaobei2/train')
    save_dataset(val_dataset, 'data/biaobei2/val')
    save_dataset(test_dataset, 'data/biaobei2/test')