from datasets import load_dataset

from constants import dataset_name


def get_datasets():
  print("downloading data")
  dataset = load_dataset(dataset_name, 'main')

  train_data = dataset['train']
  test_data = dataset['test'].select(range(500))
  return train_data, test_data