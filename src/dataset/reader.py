import os
from typing import Union

from .tfrecord.torch.dataset import TFRecordDataset
from .dataset_spec import BiLevelDatasetSpecification as BDS
from .dataset_spec import DatasetSpecification as DS
from .utils import Split


class Reader(object):
    """Class reading data from one source and assembling examples.

    Specifically, it holds part of a tf.data pipeline (the source-specific part),
    that reads data from TFRecords and assembles examples from them.
    """

    def __init__(self,
                 dataset_spec: Union[BDS, DS],
                 split: Split,
                 shuffle: bool,
                 offset: int):
        """Initializes a Reader from a source.

        The source is identified by dataset_spec and split.

        Args:
          dataset_spec: DatasetSpecification, dataset specification.
          split: A learning_spec.Split object identifying the source split.
        """
        self.dataset_spec = dataset_spec
        self.offset = offset
        self.shuffle = shuffle

        self.base_path = self.dataset_spec.path
        self.class_set = self.dataset_spec.get_classes(split)
        self.num_classes = len(self.class_set)

    def construct_class_datasets(self):
        """Constructs the list of class datasets.

        Returns:
          class_datasets: list of tf.data.Dataset, one for each class.
        """
        file_pattern = self.dataset_spec.file_pattern
        # We construct one dataset object per class. Each dataset outputs a stream
        # of `(example_string, dataset_id)` tuples.
        class_datasets = []
        for dataset_id in range(self.num_classes):
            class_id = self.class_set[dataset_id]  # noqa: E111
            if file_pattern.startswith('{}_{}'):
                # TODO(lamblinp): Add support for sharded files if needed.
                raise NotImplementedError('Sharded files are not supported yet. '  # noqa: E111
                                          'The code expects one dataset per class.')
            elif file_pattern.startswith('{}'):
                data_path = os.path.join(self.base_path, file_pattern.format(class_id))  # noqa: E111
                index_path = os.path.join(self.base_path, '{}.index'.format(class_id))  # noqa: E111
            else:
                raise ValueError('Unsupported file_pattern in DatasetSpec: %s. '  # noqa: E111
                                 'Expected something starting with "{}" or "{}_{}".' %
                                 file_pattern)
            description = {"image": "byte", "label": "int"}

            dataset = TFRecordDataset(data_path=data_path,
                                      index_path=index_path,
                                      description=description,
                                      shuffle=self.shuffle)

            class_datasets.append(dataset)

        assert len(class_datasets) == self.num_classes
        return class_datasets