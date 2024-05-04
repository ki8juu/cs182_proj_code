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
from classification_dataset import MovieReviewsDataset, Gpt2ClassificationCollator
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score



import wandb

torch.backends.cudnn.benchmark = True
print_debug_steps = 50

# when we load in the dataset, load it with the InputExample format

# def tokenize(prompt, labels):
#     # TODO: take in the model string
#     tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
     
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "left"

#     #TODO: set a max tokens length
#     tokens = tokenizer(
#         prompt,                                            
#         return_tensors='pt',
#     )

#     newline_token_id = tokenizer.encode("\n")[-1]
#     label_ids = [tokenizer.encode(" " + label)[-1] for label in labels]

#     return tokens, newline_token_id, label_ids

    

# examples -> ["x", "y", "x", "y", "x", "y", "x"]
# TODO: potentially have to shorten the input prompt, if needed
# def get_prompt(xs, ys):
#     prompt = ""


#     for i in range(len(ys)):
#         prompt += ("\n\n" + xs[i])
#         prompt +=("\n" + ys[i])
    
#     prompt += ("\n\n" + xs[len(xs) - 1])
#     return prompt.strip()

def validation(model, dataloader, device):

  # Tracking variables
  predictions_labels = []
  true_labels = []
  #total loss for this epoch.
  total_loss = 0

  # Put the model in evaluation mode--the dropout layers behave differently
  # during evaluation.
  model.eval()

  # Evaluate data for one epoch
  for batch in tqdm(dataloader, total=len(dataloader)):

    # add original labels
    true_labels += batch['labels'].numpy().flatten().tolist()

    # move batch to device
    batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up validation
    with torch.no_grad():        

        # Forward pass, calculate logit predictions.
        # This will return the logits rather than the loss because we have
        # not provided labels.
        # token_type_ids is the same as the "segment ids", which 
        # differentiates sentence 1 and 2 in 2-sentence tasks.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(**batch)

        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple along with the logits. We will use logits
        # later to to calculate training accuracy.
        loss, logits = outputs[:2]
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.

        total_loss += loss.item()
        
        # get predicitons to list
        predict_content = logits.argmax(axis=-1).flatten().tolist()

        # update list
        predictions_labels += predict_content

  # Calculate the average loss over the training data.
  avg_epoch_loss = total_loss / len(dataloader)

  # Return all true labels and prediciton for future evaluations.
  return true_labels, predictions_labels, avg_epoch_loss

# TODO: don't do curriculum training yet

def train_step(model, dataloader, optimizer, scheduler, device):
    # Single pass through the dataloader (technically not just one train step lol)

    # Tracking variables.
    predictions_labels = []
    true_labels = []

    total_loss = 0

    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()

        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        # batch.pop('labels', None)

        # TODO: model.zero_grad or optimizer.zero_grad?
        model.zero_grad()

        outputs = model(**batch)

        # TODO: what is the loss function being used for pretrained GPT2
        loss, logits = outputs[:2]

        print("model loss", loss)

        total_loss += loss.item()

        loss.backward()

        # prevent exploading gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Update learning rate
        scheduler.step()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # Convert these logits to list of predicted labels values.
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

        # TODO: i'm not sure exactly how many batches it takes to go through dataloader
        # if i % args.wandb.log_every_steps == 0 and not args.test_run:
        #     wandb.log(
        #         {
        #             "overall_loss": loss,
        #             # "excess_loss": loss / baseline_loss,
        #             # "pointwise/loss": dict(
        #             #     zip(point_wise_tags, point_wise_loss.cpu().numpy())
        #             # ),
        #             # "n_points": curriculum.n_points,
        #             # "n_dims": curriculum.n_dims_truncated,
        #         },
        #         step=i,
        #     )
        # pbar.set_description(f"loss {loss}") ----- not sure what to do here

    avg_epoch_loss = total_loss / len(dataloader)

    return true_labels, predictions_labels, avg_epoch_loss


    # prompt = get_prompt(xs, ys)
    # # check evaluate_icl_causal_llm or what is passed in during training
    # # tokenize

    # tokens, newline_id, label_ids = tokenize(prompt, ys)

    # def prefix_allowed_tokens_fn(batch_id, input_ids):
    #     return label_ids

    # optimizer.zero_grad()


    # use the .generate() function to generate text using gpt2, research what it outputs
    # output = model(xs, ys)
    # loss = loss_func(output, ys)
    # loss.backward()
    # optimizer.step()
    # return loss.detach().item(), output.detach()


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
    epochs = 4


    gpt2_classification_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, 
                                                          labels_encoder=labels_ids, 
                                                          max_sequence_len=max_length)

    train_dataset = MovieReviewsDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, collate_fn=gpt2_classification_collator)

    valid_dataset =  MovieReviewsDataset()
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

    # n_dims = model.n_dims

    # NOT EVEN SURE IF WE NEED THIS
    # TODO(emma apr 24) change this, probably make a new data sampler for the new dataset
    # args.training.data -- dataset name: "gaussian" or "language"
    # not sure what n_dims is -- is it the dimensions of the data? im not sure what the dimensions of the language data is
    # we should also take in the dataset name as an argument, can probably add an args.dataset_name to config
    # data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, dataset_name=args.training.dataset_name)
    
    # TODO(emma apr 29) change this
    # task_sampler = get_task_sampler(
    #     args.training.task,
    #     n_dims,
    #     bsize,
    #     num_tasks=args.training.num_tasks,
    #     **args.training.task_kwargs,
    # )
    print("length of the train_dataloader", len(train_dataloader))
    total_steps = len(train_dataloader) * epochs

    # pbar = tqdm(range(starting_step, total_steps))

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
        # TODO modify the train function
        train_labels, train_predict, train_loss = train_step(model, train_dataloader, optimizer, scheduler, device)
        train_acc = accuracy_score(train_labels, train_predict)

        valid_labels, valid_predict, val_loss = validation(model, valid_dataloader, device)
        val_acc = accuracy_score(valid_labels, valid_predict)

        all_loss['train_loss'].append(train_loss)
        all_loss['val_loss'].append(val_loss)
        all_acc['train_acc'].append(train_acc)
        all_acc['val_acc'].append(val_acc)

    # TODO: log epoch loss and accuracy in wandb

    # TODO: evaluate 
    

    ### OLD CODE
    # TODO: make sure i use the some of this code 
    # for i in pbar:
    #     data_sampler_args = {}
    #     task_sampler_args = {}

    #     # if "sparse" in args.training.task:
    #     #     task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
    #     if num_training_examples is not None:
    #         assert num_training_examples >= bsize
    #         seeds = sample_seeds(num_training_examples, bsize)
    #         data_sampler_args["seeds"] = seeds
    #         task_sampler_args["seeds"] = [s + 1 for s in seeds]

    #     # TODO(emma apr 24) change this to sample from the specified dataset
    #     # n_points is the number of in context examples
    #     # it increases when you do curriculum training.

    #     # TODO(emma apr 29) do we want to have it as vectors or the prompt?
    #     xs = data_sampler.sample_xs(
    #         curriculum.n_points,
    #         bsize,
    #         curriculum.n_dims_truncated,
    #         **data_sampler_args,
    #     )

    #     # TODO(emma apr 30) in else
    #     # Task is retrieved after the xs are retrieved -- for language, we should retrieve task after. 
    #     # Or for simplicity we can probably remove the task sampler and directly use the language classification task
    #     task = task_sampler(**task_sampler_args)
    #     if "seq" in args.training.task:
    #         x0 = xs[:, 0, :]
    #         xs, ys = task.generate_sequence(x0, args.model.n_positions)
    #         # if i % print_debug_steps == 0:
    #         #   print("x0: ", x0[0])
    #         #   print("xs: ", xs[0])
    #         #   print("ys: ", ys[0])
    #     else:
    #         ys = task.evaluate(xs)

    #     #TODO(emma apr 30) use same loss as reference paper
    #     loss_func = task.get_training_metric()

    #     # TODO(emma apr 30) this function should also formulate the prompt, pass through model
    #     loss, output = train_step(model, xs.to(device), ys.to(device), optimizer, loss_func, i)

    #     # point_wise_tags = list(range(curriculum.n_points))
    #     point_wise_loss_func = task.get_metric()
    #     point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=0)

    #     # baseline_loss = (
    #     #     sum(
    #     #         max(curriculum.n_dims_truncated - ii, 0)
    #     #         for ii in range(curriculum.n_points)
    #     #     )
    #     #     / curriculum.n_points
    #     # )

    #     if i % args.wandb.log_every_steps == 0 and not args.test_run:
    #         wandb.log(
    #             {
    #                 "overall_loss": loss,
    #                 # "excess_loss": loss / baseline_loss,
    #                 # "pointwise/loss": dict(
    #                 #     zip(point_wise_tags, point_wise_loss.cpu().numpy())
    #                 # ),
    #                 # "n_points": curriculum.n_points,
    #                 # "n_dims": curriculum.n_dims_truncated,
    #             },
    #             step=i,
    #         )

    #     curriculum.update()

        # TODO: SAVE CHECKPOINTS PROPERLY -- also figure out when we know how many batches per epoch
        # pbar.set_description(f"loss {loss}")
        # if i % args.training.save_every_steps == 0 and not args.test_run:
        #     training_state = {
        #         "model_state_dict": model.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "train_step": i,
        #     }
        #     torch.save(training_state, state_path)

        # if (
        #     args.training.keep_every_steps > 0
        #     and i % args.training.keep_every_steps == 0
        #     and not args.test_run
        #     and i > 0
        # ):
        #     torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


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

    # TODO: do we need to do this -- right now it's erroring
    # resize model embedding to match new tokenizer
    # necessary when the vocab size of the tokenizer changes 
    # esp when you add new tokens to the tokenizer
    # for us this might not do anything
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