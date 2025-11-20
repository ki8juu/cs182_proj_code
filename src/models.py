import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb

from base_models import NeuralNetwork, ParallelNetworks

# def _count_parameters(module):
#     return sum(p.numel() for p in module.parameters() if p.requires_grad)


# def _align_hidden_dim(hidden_dim, alignment):
#     if hidden_dim < alignment:
#         return alignment
#     remainder = hidden_dim % alignment
#     if remainder == 0:
#         return hidden_dim
#     return hidden_dim + (alignment - remainder)


# def _match_lstm_like_dimensions(
#     conf, template_kwargs, builder, *, alignment, has_mlp=True
# ):
#     initial_hidden = _align_hidden_dim(
#         getattr(conf, "lstm_hidden_dim", conf.n_embd), alignment
#     )
#     target_model = TransformerModel(
#         n_dims=conf.n_dims,
#         n_positions=conf.n_positions,
#         n_embd=conf.n_embd,
#         n_layer=conf.n_layer,
#         n_head=conf.n_head,
#     )
#     target_params = _count_parameters(target_model)

#     visited = set()
#     candidates_checked = 0
#     max_candidates = 4096

#     def _candidate_iter():
#         yield initial_hidden
#         offset = 1
#         while True:
#             plus = initial_hidden + offset * alignment
#             minus = initial_hidden - offset * alignment
#             if plus >= alignment:
#                 yield plus
#             if minus >= alignment:
#                 yield minus
#             offset += 1

#     for hidden_dim in _candidate_iter():
#         if hidden_dim in visited:
#             continue
#         visited.add(hidden_dim)
#         candidates_checked += 1
#         if candidates_checked > max_candidates:
#             break
#         probe_kwargs = dict(template_kwargs, hidden_dim=hidden_dim)
#         if has_mlp:
#             probe_kwargs["mlp_hidden_dim"] = 1
#         probe_model = builder(**probe_kwargs)
#         base_count = _count_parameters(probe_model)
#         if base_count > target_params:
#             continue
#         if not has_mlp:
#             if base_count == target_params:
#                 return {"hidden_dim": hidden_dim}, target_params
#             continue
#         slope_kwargs = dict(probe_kwargs)
#         slope_kwargs["mlp_hidden_dim"] = 2
#         slope_model = builder(**slope_kwargs)
#         slope = _count_parameters(slope_model) - base_count
#         if slope <= 0:
#             continue
#         diff = target_params - base_count
#         if diff % slope != 0:
#             continue
#         mlp_hidden_dim = (diff // slope) + 1
#         return {"hidden_dim": hidden_dim, "mlp_hidden_dim": mlp_hidden_dim}, target_params

#     raise ValueError(
#         "Unable to find LSTM dimensions that match the transformer's parameter count."
#     )

def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == "lstm":
        hidden_dim = getattr(conf, "lstm_hidden_dim", conf.n_embd)
        model = LSTMModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            hidden_dim=hidden_dim,
            num_layers=getattr(conf, "lstm_num_layers", 2),
            dropout=getattr(conf, "lstm_dropout", 0.0),
            mlp_hidden_dim=getattr(conf, "lstm_mlp_hidden_dim", None),
            mlp_multiplier=getattr(conf, "lstm_mlp_multiplier", 4.0),
        )
        # match_params = getattr(conf, "lstm_match_transformer_params", True)
        # mlp_hidden_dim = getattr(conf, "lstm_mlp_hidden_dim", None)
        # if match_params:
        #     hidden_dim, mlp_hidden_dim, target_params = _match_lstm_like_dimensions(
        #         conf,
        #         template_kwargs,
        #         LSTMModel,
        #         alignment=1,
        #     )
        # else:
        #     hidden_dim = _align_hidden_dim(
        #         getattr(conf, "lstm_hidden_dim", conf.n_embd), 1
        #     )
        #     if mlp_hidden_dim is None:
        #         multiplier = getattr(conf, "lstm_mlp_multiplier", 4.0)
        #         mlp_hidden_dim = max(1, int(round(hidden_dim * multiplier)))

        # model = LSTMModel(
        #     hidden_dim=hidden_dim,
        #     mlp_hidden_dim=mlp_hidden_dim,
        #     **template_kwargs,
        # )

        # if match_params:
        #     lstm_params = _count_parameters(model)
        #     if lstm_params != target_params:
        #         raise ValueError(
        #             "LSTM model parameter count mismatch after matching routine."
        #         )
    elif conf.family == "lstm_attention":
        attn_heads = getattr(conf, "attn_num_heads", 4)
        if attn_heads <= 0:
            raise ValueError("attn_num_heads must be positive for the LSTM attention model")
        # if conf.n_layer <= 0 or conf.n_head <= 0:
        #     raise ValueError(
        #         "n_layer and n_head must be positive to define the transformer parameter budget"
        #     )
        # template_kwargs = dict(
        hidden_dim = getattr(conf, "lstm_hidden_dim", conf.n_embd)
        if hidden_dim % attn_heads != 0:
            raise ValueError("lstm_hidden_dim must be divisible by attn_num_heads")
        model = LSTMAttentionModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            hidden_dim=hidden_dim,
            num_layers=getattr(conf, "lstm_num_layers", 2),
            dropout=getattr(conf, "lstm_dropout", 0.0),
            attn_heads=attn_heads,
            attn_dropout=getattr(conf, "attn_dropout", 0.0),
            # attn_mlp_multiplier=getattr(conf, "attn_mlp_multiplier", 4.0),
        )
        # match_params = getattr(conf, "lstm_match_transformer_params", True)
        # mlp_hidden_dim = getattr(conf, "attn_mlp_hidden_dim", None)

        # if match_params:
        #     hidden_dim, mlp_hidden_dim, target_params = _match_lstm_like_dimensions(
        #         conf,
        #         template_kwargs,
        #         LSTMAttentionModel,
        #         alignment=attn_heads,
        #     )
        # else:
        #     hidden_dim = _align_hidden_dim(
        #         getattr(conf, "lstm_hidden_dim", conf.n_embd), attn_heads
        #     )
        #     if mlp_hidden_dim is None:
        #         multiplier = getattr(conf, "attn_mlp_multiplier", 4.0)
        #         mlp_hidden_dim = max(1, int(round(hidden_dim * multiplier)))

        # model = LSTMAttentionModel(
        #     hidden_dim=hidden_dim,
        #     mlp_hidden_dim=mlp_hidden_dim,
        #     **template_kwargs,
        # )

        # if match_params:
        #     lstm_params = _count_parameters(model)
        #     if lstm_params != target_params:
        #         raise ValueError(
        #             "LSTM attention model parameter count mismatch after matching routine."
        #         )
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs

class LSTMModel(nn.Module):
    def __init__(
        self,
        n_dims,
        n_positions,
        n_embd=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.0,
        mlp_hidden_dim=None,
        mlp_multiplier=4.0,
    ):
        super().__init__()
        self.name = f"lstm_embd={n_embd}_hidden={hidden_dim}_layers={num_layers}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._encoder = nn.LSTM(
            input_size=n_embd,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if mlp_hidden_dim is None:
            mlp_hidden_dim = max(1, int(round(hidden_dim * mlp_multiplier)))
        self.mlp_hidden_dim = mlp_hidden_dim
        self.name += f"_mlp={mlp_hidden_dim}"
        self._mlp_norm = nn.LayerNorm(hidden_dim)
        self._mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
        self._mlp_dropout = nn.Dropout(dropout)
        self._read_out = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1], device=ys.device, dtype=torch.long)
        else:
            inds = torch.tensor(inds, device=ys.device, dtype=torch.long)
            if max(inds).item() >= ys.shape[1] or min(inds).item() < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        encoded, _ = self._encoder(embeds)
        query = encoded[:, ::2, :]
        mlp_out = self._mlp(self._mlp_norm(query))
        query = query + self._mlp_dropout(mlp_out)
        prediction = self._read_out(query)[..., 0]
        return prediction[:, inds]

class LSTMAttentionModel(nn.Module):
    def __init__(
        self,
        n_dims,
        n_positions,
        n_embd=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.0,
        attn_heads=4,
        attn_dropout=0.0,
    ):
        super().__init__()
        self.name = (
            f"lstm_attention_embd={n_embd}_hidden={hidden_dim}_layers={num_layers}_heads={attn_heads}"
        )
        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._encoder = nn.LSTM(
            input_size=n_embd,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if hidden_dim % attn_heads != 0:
            raise ValueError("hidden_dim must be divisible by attn_heads")
        self._query_norm = nn.LayerNorm(hidden_dim)
        self._context_norm = nn.LayerNorm(hidden_dim)
        self._decoder_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self._decoder_dropout = nn.Dropout(attn_dropout)
        self._read_out = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1], device=ys.device, dtype=torch.long)
        else:
            inds = torch.tensor(inds, device=ys.device, dtype=torch.long)
            if max(inds).item() >= ys.shape[1] or min(inds).item() < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        encoded, _ = self._encoder(embeds)
        query = encoded[:, ::2, :]
        context = self._context_norm(encoded)

        num_queries = query.size(1)
        num_keys = context.size(1)

        query_positions = torch.arange(num_queries, device=query.device).unsqueeze(1)
        key_positions = torch.arange(num_keys, device=query.device).unsqueeze(0)
        attn_mask = key_positions > (2 * query_positions)

        attn_out, _ = self._decoder_attn(
            self._query_norm(query), context, context, attn_mask=attn_mask
        )
        attn_out = query + self._decoder_dropout(attn_out)
        prediction = self._read_out(attn_out)[..., 0]
        return prediction[:, inds]

class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
