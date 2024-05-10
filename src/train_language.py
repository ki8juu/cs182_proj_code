import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
# TODO: not sure if i need to use AutoTokenizer instead
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from classification_dataset import LanguageDataset, Gpt2ClassificationCollator
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score

import wandb

torch.backends.cudnn.benchmark = True
print_debug_steps = 50

def validation(model, dataloader, device):
  predictions_labels = []
  true_labels = []
  total_loss = 0

  model.eval()

  for batch in tqdm(dataloader, total=len(dataloader)):

    true_labels += batch['labels'].numpy().flatten().tolist()

    batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}

    with torch.no_grad():        

        outputs = model(**batch)
        loss, logits = outputs[:2]
        
        logits = logits.detach().cpu().numpy()

        total_loss += loss.item()
        
        predict_content = logits.argmax(axis=-1).flatten().tolist()

        predictions_labels += predict_content

  avg_epoch_loss = total_loss / len(dataloader)

  return true_labels, predictions_labels, avg_epoch_loss

# TODO: don't do curriculum training yet
def train_step(model, dataloader, optimizer, scheduler, device, epoch):
    predictions_labels = []
    true_labels = []

    total_loss = 0

    for batch_i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        true_labels += batch['labels'].numpy().flatten().tolist()

        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}

        model.zero_grad()

        outputs = model(**batch)

        loss, logits = outputs[:2]

        print("model loss", loss)

        total_loss += loss.item()

        loss.backward()

        # prevent exploading gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Update learning rate
        scheduler.step()

        logits = logits.detach().cpu().numpy()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

        # hardcoded 100, in args.wandb.log_every_steps or something
        # also no option for test run
        i = epoch * len(dataloader) + batch_i
        print(i, "train step")
        if (i) % 10 == 0:
            wandb.log(
                {
                    "overall_loss": loss,
                },
                step=i,
            )

    avg_epoch_loss = total_loss / len(dataloader)

    return true_labels, predictions_labels, avg_epoch_loss


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args, device, tokenizer):
    # DATA STUFF
    # TODO: set all this as something configurable
    # TODO: fix labels_ids and max_length potentially
    labels_ids = {'neg': 0, 'pos': 1}
    max_length = 60

    bsize = args.training.batch_size
    epochs = 6


    gpt2_classification_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, 
                                                          labels_encoder=labels_ids, 
                                                          max_sequence_len=max_length)

    train_dataset = LanguageDataset(args.data.dataset_name, args.data.text_key, args.model.n_positions)
    train_dataloader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, collate_fn=gpt2_classification_collator)

    valid_dataset =  LanguageDataset(args.data.dataset_name, args.data.text_key, args.model.n_positions)
    valid_dataloader = DataLoader(valid_dataset, batch_size=bsize, shuffle=False, collate_fn=gpt2_classification_collator)


    # EVERYTHING ELSE NEEDED FOR TRAINING
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate)
    # curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        # for i in range(state["train_step"] + 1):
        #     curriculum.update()

   
    print("length of the train_dataloader", len(train_dataloader))
    total_steps = len(train_dataloader) * epochs


    # TODO: not sure if we need this
    num_training_examples = args.training.num_training_examples

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps -  starting_step)
    # TODO: potentially get rid of this to get wandb stuff working
    all_loss = {'train_loss':[], 'val_loss':[]}
    all_acc = {'train_acc':[], 'val_acc':[]}

    for epoch in tqdm(range(epochs)):
        # Perform one full pass over the training set.
        train_labels, train_predict, train_loss = train_step(model, train_dataloader, optimizer, scheduler, device, epoch)
        train_acc = accuracy_score(train_labels, train_predict)

        valid_labels, valid_predict, val_loss = validation(model, valid_dataloader, device)
        val_acc = accuracy_score(valid_labels, valid_predict)

        all_loss['train_loss'].append(train_loss)
        all_loss['val_loss'].append(val_loss)
        all_acc['train_acc'].append(train_acc)
        all_acc['val_acc'].append(val_acc)

        # TODO: not sure if i can do epoch instead of step
        # wandb.log(
        #         all_loss,
        #         step=epoch,
        #     )

    # TODO: log epoch loss and accuracy in wandb

    # TODO: evaluate 
    

def main(args):
    if args.test_run:
        # curriculum_args = args.training.curriculum
        # curriculum_args.points.start = curriculum_args.points.end
        # curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model = build_model(args.model, "seq" in args.training.task)

    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=args.model.name)
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    # TODO: potentially remove
    model._backbone.resize_token_embeddings(len(tokenizer))

    # set model padding token id
    model._backbone.config.pad_token_id = model._backbone.config.eos_token_id

    model.to(device)
    model.train()

    train(model, args, device, tokenizer)

    # TODO: not sure what is done here
    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)