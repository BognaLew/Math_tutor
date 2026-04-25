from datasets import load_dataset

from constants import cache_dir, dataset_name


def get_datasets():
  print("downloading data")
  dataset = load_dataset(dataset_name, 'main', cache_dir=cache_dir)

  train_data = dataset['train'].select(range(1500))
  test_data = dataset['test'].select(range(500))
  return train_data, test_data