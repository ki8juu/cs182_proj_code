from torch.utils.data import Dataset
import os
from datasets import load_dataset
import random
import torch

# TODO: create a base dataset class!
# TODO: make the input sequential
# TODO: make it work with input output strings
class LanguageDataset(Dataset):
  r"""PyTorch Dataset class for loading data.

  This is where the data parsing happens.

  Arguments:

    path (:obj:`str`):
        Path to the data partition.

  """

  def __init__(self, dataset_name, text_key, num_incontext):

    # Check if path exists.
    # if not os.path.isdir(path):
    #   # Raise error if path is invalid.
    #   raise ValueError('Invalid `path` variable! Needs to be a directory')


    # TODO: this is hardcoded, eventually, don't do this!
    self.dataset = load_dataset(dataset_name)

    # TODO: for now, lets just use train
    self.dataset['train'] = self.dataset['train'].shuffle(seed=42)
    self.dataset['test'] = self.dataset['test'].shuffle(seed=42)

    # randomly shuffle and sample from the dataset

    # TODO: sometimes its 'sentence' or 'text'

    self.texts = []
    self.labels = []

    print(len(self.dataset['train']))
    print(len(self.dataset['test']))

    print(num_incontext, "number of in context examples")

    max_examples = len(self.dataset['train']) // 11

    i = 0
    while i < max_examples:
        context = []
        for j in range(10):
            context.append("\n".join([self.dataset['train'][i + j][text_key], 'pos' if self.dataset['train'][i + j]['label'] else 'neg']))
        context.append(self.dataset['train'][i + 10][text_key])
        self.texts.append("\n\n".join(context))
        self.labels.append('pos' if self.dataset['train'][i + j]['label'] else 'neg')
        i += 1

    # Number of exmaples.
    self.n_examples = len(self.labels)

    print("initializing the dataset, number of examples:")
    print(self.n_examples)
    return

  def __len__(self):
    r"""When used `len` return the number of examples.

    """
    
    return self.n_examples

  def __getitem__(self, item):
    r"""Given an index return an example from the position.
    
    Arguments:

      item (:obj:`int`):
          Index position to pick an example to return.

    Returns:
      :obj:`Dict[str, str]`: Dictionary of inputs that contain text and 
      asociated labels.

    """
    # print("what the item looks like")
    # print(self.texts[item])
    return {'text':self.texts[item],
            'label':self.labels[item]}



class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask. 
    
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with 
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels':torch.tensor(labels)})

        return inputs
