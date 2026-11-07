import collections
import enum
import math
import pathlib
import typing
import warnings

import numpy as np
import torch
import torch.optim
import torch.utils.data
import tqdm
from matplotlib import pyplot as plt

from util import paint_reliability_diagram, compute_cost, seed_setup, calculate_calibration_curve

# Write reliability and confidence plots during validation.
ENABLE_EXTENDED_ANALYSIS = True
# Start SWAG from the supplied MAP checkpoint. Set to False to train MAP locally.
USE_PRETRAINED_WEIGHTS = True


def load_dataset(directory: pathlib.Path, split: str) -> torch.utils.data.TensorDataset:
    """Load images, snow flags, cloud flags and class labels in checker order."""
    with np.load(directory / f"{split}_xs.npz") as inputs, np.load(directory / f"{split}_ys.npz") as targets:
        return torch.utils.data.TensorDataset(
            torch.from_numpy(inputs[f"{split}_xs"]),
            torch.from_numpy(targets[f"{split}_is_snow"]),
            torch.from_numpy(targets[f"{split}_is_cloud"]),
            torch.from_numpy(targets[f"{split}_ys"]),
        )


def main():
    directory = pathlib.Path.cwd()
    training_dataset = load_dataset(directory, "train")
    validation_dataset = load_dataset(directory, "val")

    seed_setup()

    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    swag_inference = SWAInferenceHandler(
        train_xs=training_dataset.tensors[0],
        model_dir=directory,
    )
    swag_inference.train_model(training_loader)
    swag_inference.run_calibration(validation_dataset)

    # fork_rng ensures that the evaluation does not change the rng state.
    with torch.random.fork_rng():
        run_evaluation(swag_inference, validation_dataset, ENABLE_EXTENDED_ANALYSIS, directory)


class InferenceMode(enum.Enum):
    """
    Inference mode switch for your implementation.
    `MAP` simply predicts the most likely class using pretrained MAP weights.
    `SWAG_DIAGONAL` and `SWAG_FULL` correspond to SWAG-diagonal and the full SWAG method, respectively.
    """
    MAP = 0
    SWAG_DIAGONAL = 1
    SWAG_FULL = 2


class SWAInferenceHandler:
    """Fit and sample the SWAG approximation, then calibrate its predictions."""

    def __init__(
        self,
        train_xs: torch.Tensor,
        model_dir: pathlib.Path,
        inference_mode: InferenceMode = InferenceMode.SWAG_FULL,
        swag_training_epochs: int = 30,
        swag_lr: float = 0.043,
        swag_update_interval: int = 1,
        max_rank_deviation_matrix: int = 15,
        num_bma_samples: int = 30,
    ):
        """
        :param train_xs: Training images (for storage only)
        :param model_dir: Path to directory containing pretrained MAP weights
        :param inference_mode: Control which inference mode (MAP, SWAG-diagonal, full SWAG) to use
        :param swag_training_epochs: Total number of gradient descent epochs for SWAG
        :param swag_lr: Learning rate for SWAG gradient descent
        :param swag_update_interval: Frequency (in epochs) for updating SWAG statistics during gradient descent
        :param max_rank_deviation_matrix: Rank of deviation matrix for full SWAG
        :param num_bma_samples: Number of networks to sample for Bayesian model averaging during prediction
        """

        self.model_dir = model_dir
        self.inference_mode = inference_mode
        self.swag_training_epochs = swag_training_epochs
        self.swag_lr = swag_lr
        self.swag_update_interval = swag_update_interval
        self.max_rank_deviation_matrix = max_rank_deviation_matrix
        self.num_bma_samples = num_bma_samples

        self.network = CNN(in_channels=3, out_classes=6)
        self.training_dataset = torch.utils.data.TensorDataset(train_xs)

        # SWAG-diagonal
        self.mean_weights = self._create_weight_copy()
        self.squared_weights = self._create_weight_copy()
        self.num_updates = 0

        # Full SWAG
        self.deviation_vectors = collections.deque(maxlen=self.max_rank_deviation_matrix)

        self._calibration_threshold = 0.5
        self.temperature = 1.0

    def update_swag_statistics(self) -> None:
        """
        Update SWAG statistics with the current weights of self.network.
        """
        copied_params = {name: param.detach() for name, param in self.network.named_parameters()}
        self.num_updates += 1
        n = self.num_updates

        # SWAG-diagonal
        for name, param in copied_params.items():
            self.mean_weights[name] += (param - self.mean_weights[name]) / n
            self.squared_weights[name] += (param**2 - self.squared_weights[name]) / n

        # Full SWAG
        if self.inference_mode == InferenceMode.SWAG_FULL:
            deviation_dict = {
                name: param - self.mean_weights[name]
                for name, param in copied_params.items()
            }
            self.deviation_vectors.append(deviation_dict)

    def fit_swag_model(self, loader: torch.utils.data.DataLoader) -> None:
        """
        Fit SWAG on top of the pretrained network self.network.
        """
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.swag_lr,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )
        lr_scheduler = SWAGScheduler(
            optimizer,
            epochs=self.swag_training_epochs,
            steps_per_epoch=len(loader),
        )

        self.num_updates = 0

        self.network.train()
        with tqdm.trange(self.swag_training_epochs, desc="Running gradient descent for SWA") as pbar:
            progress_dict = {}
            for epoch in pbar:
                avg_loss = 0.0
                avg_accuracy = 0.0
                num_samples = 0
                for batch_images, _, _, batch_labels in loader:
                    optimizer.zero_grad()
                    predictions = self.network(batch_images)
                    batch_loss = loss_fn(input=predictions, target=batch_labels)
                    batch_loss.backward()
                    optimizer.step()
                    progress_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    lr_scheduler.step()

                    # Calculate cumulative average training loss and accuracy
                    avg_loss = (batch_images.size(0) * batch_loss.item() + num_samples * avg_loss) / (
                        num_samples + batch_images.size(0)
                    )
                    avg_accuracy = (
                        torch.sum(predictions.argmax(dim=-1) == batch_labels).item()
                        + num_samples * avg_accuracy
                    ) / (num_samples + batch_images.size(0))
                    num_samples += batch_images.size(0)
                    progress_dict["avg. epoch loss"] = avg_loss
                    progress_dict["avg. epoch accuracy"] = avg_accuracy
                    pbar.set_postfix(progress_dict)

                if (epoch + 1) % self.swag_update_interval == 0:
                    self.update_swag_statistics()

    def _predict_logits(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Collect logits from the current network in data-loader order.
        """
        all_predictions = []
        for (batch_images,) in loader:
            all_predictions.append(self.network(batch_images))

        return torch.cat(all_predictions)

    def run_calibration(self, validation_data: torch.utils.data.Dataset) -> None:
        """Fit the confidence cutoff using the policy from the passing public run."""
        if self.inference_mode == InferenceMode.MAP:
            self._calibration_threshold = 0.0
            self.temperature = 1.0
            return

        images, _, _, labels = validation_data.tensors
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images), batch_size=32)
        draws = []
        with torch.random.fork_rng(), torch.no_grad():
            for _ in tqdm.trange(self.num_bma_samples, desc="Sampling for calibration"):
                self.sample_parameters()
                draws.append(self._predict_logits(loader))

        # This cutoff is fitted on mean logits. Deployment still averages probabilities.
        mean_logits = torch.stack(draws).mean(dim=0)
        labeled = labels != -1
        log_temperature = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.01, max_iter=100)
        nll = torch.nn.CrossEntropyLoss()
        labeled_logits, targets = mean_logits[labeled], labels[labeled]

        def closure():
            optimizer.zero_grad()
            loss = nll(labeled_logits / log_temperature.exp(), targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = log_temperature.detach().exp().item()
        confidence, classes = (mean_logits / self.temperature).softmax(dim=-1).max(dim=-1)
        thresholds = torch.cat((torch.tensor([0.0]), confidence.unique(sorted=True), torch.tensor([1.01])))
        costs = [
            compute_cost(torch.where(confidence >= threshold, classes, -1), labels).item()
            for threshold in thresholds
        ]
        best = int(np.argmin(costs))
        self._calibration_threshold = thresholds[best].item()
        print(f"Temperature {self.temperature:.4f}, threshold {self._calibration_threshold:.6f}")

    @torch.no_grad()
    def predict_probabilities_swag(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Average class probabilities across posterior draws."""
        self.network.eval()
        model_predictions = []
        for _ in tqdm.trange(self.num_bma_samples, desc="Performing Bayesian model averaging"):
            self.sample_parameters()
            batches = [
                (self.network(images) / self.temperature).softmax(dim=-1)
                for (images,) in loader
            ]
            model_predictions.append(torch.cat(batches))

        probabilities = torch.stack(model_predictions).mean(dim=0)
        assert probabilities.ndim == 2 and probabilities.size(1) == 6
        return probabilities

    @torch.no_grad()
    def sample_parameters(self) -> None:
        """Draw weights with a diagonal term and an optional per-tensor low-rank term."""
        rank = len(self.deviation_vectors)
        for name, param in self.network.named_parameters():
            mean = self.mean_weights[name]
            variance = (self.squared_weights[name] - mean.square()).clamp_min(1e-30)
            diagonal = variance.sqrt() * torch.randn_like(param) / math.sqrt(2.0)
            sampled_weight = mean + diagonal

            if self.inference_mode == InferenceMode.SWAG_FULL and rank > 1:
                coefficients = torch.randn(rank, dtype=param.dtype) / math.sqrt(2.0 * (rank - 1))
                low_rank = torch.zeros_like(param)
                for coefficient, deviation in zip(coefficients, self.deviation_vectors):
                    low_rank += coefficient * deviation[name]
                sampled_weight += low_rank

            param.copy_(sampled_weight)

        self._update_batchnorm_statistics()

    def label_prediction(self, predicted_probabilities: torch.Tensor) -> torch.Tensor:
        """
        Predict labels in {0, 1, 2, 3, 4, 5} or "don't know" as -1
        based on your model's predicted probabilities.
        """

        label_probabilities, max_likelihood_labels = torch.max(predicted_probabilities, dim=-1)
        num_samples, num_classes = predicted_probabilities.size()
        assert label_probabilities.size() == (num_samples,) and max_likelihood_labels.size() == (num_samples,)

        # Use the optimized threshold found in run_calibration()
        return torch.where(
            label_probabilities >= self._calibration_threshold,
            max_likelihood_labels,
            torch.ones_like(max_likelihood_labels) * -1,
        )

    def _create_weight_copy(self) -> typing.Dict[str, torch.Tensor]:
        """Create an all-zero copy of the network weights as a dictionary that maps name -> weight"""
        return {
            name: torch.zeros_like(param, requires_grad=False)
            for name, param in self.network.named_parameters()
        }

    def train_model(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> None:
        """
        Perform full SWAG fitting procedure.
        """
        checkpoint_path = self.model_dir / "map_weights.pt"
        if USE_PRETRAINED_WEIGHTS:
            self.network.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            print("Loaded pretrained MAP weights from", checkpoint_path)
        else:
            self.fit_map_model(loader)

        # SWAG
        if self.inference_mode in (InferenceMode.SWAG_DIAGONAL, InferenceMode.SWAG_FULL):
            self.fit_swag_model(loader)

    def fit_map_model(self, loader: torch.utils.data.DataLoader) -> None:
        """
        MAP inference procedure to obtain initial weights of self.network.
        """
        map_training_epochs = 140
        initial_learning_rate = 0.01
        reduced_learning_rate = 0.0001
        start_decay_epoch = 50
        decay_factor = reduced_learning_rate / initial_learning_rate

        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=initial_learning_rate,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=decay_factor,
                    total_iters=(map_training_epochs - start_decay_epoch) * len(loader),
                ),
            ],
            milestones=[start_decay_epoch * len(loader)],
        )

        self.network.train()
        with tqdm.trange(map_training_epochs, desc="Fitting initial MAP weights") as pbar:
            progress_dict = {}
            for epoch in pbar:
                avg_loss = 0.0
                avg_accuracy = 0.0
                num_samples = 0
                for batch_images, _, _, batch_labels in loader:
                    optimizer.zero_grad()
                    predictions = self.network(batch_images)
                    batch_loss = loss_fn(input=predictions, target=batch_labels)
                    batch_loss.backward()
                    optimizer.step()

                    progress_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        lr_scheduler.step()

                    avg_loss = (batch_images.size(0) * batch_loss.item() + num_samples * avg_loss) / (
                        num_samples + batch_images.size(0)
                    )
                    avg_accuracy = (
                        torch.sum(predictions.argmax(dim=-1) == batch_labels).item()
                        + num_samples * avg_accuracy
                    ) / (num_samples + batch_images.size(0))
                    num_samples += batch_images.size(0)

                    progress_dict["avg. epoch loss"] = avg_loss
                    progress_dict["avg. epoch accuracy"] = avg_accuracy
                    pbar.set_postfix(progress_dict)

    def predict_probs(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for the given images xs.
        """
        self.network.eval()

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xs),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        with torch.no_grad():
            if self.inference_mode == InferenceMode.MAP:
                return self.predict_probabilities_map(loader)
            return self.predict_probabilities_swag(loader)

    def predict_probabilities_map(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Return class probabilities from the temperature-scaled MAP network."""
        logits = self._predict_logits(loader)
        return torch.softmax(logits / self.temperature, dim=-1)

    def _update_batchnorm_statistics(self) -> None:
        """
        Reset and fit batch normalization statistics using the training dataset self.training_dataset.
        """

        original_momentum_values = dict()
        for module in self.network.modules():
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue

            original_momentum_values[module] = module.momentum
            module.momentum = None
            module.reset_running_stats()

        loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        self.network.train()
        for (batch_images,) in loader:
            self.network(batch_images)
        self.network.eval()

        for module, momentum in original_momentum_values.items():
            module.momentum = momentum


class SWAGScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Custom learning rate scheduler.
    Default implementation keeps the original learning rate constant.
    """

    def calculate_lr(self, current_epoch: float, previous_lr: float) -> float:
        """
        Calculate the learning rate for the epoch given by current_epoch.
        """
        return previous_lr

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        steps_per_epoch: int,
        alpha_min = 0.042,
        alpha_max = 0.047,
        cycle_length = 10
    ):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.cycle_length = cycle_length
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )
        return [
            self.calculate_lr(self.last_epoch / self.steps_per_epoch, group["lr"])
            for group in self.optimizer.param_groups
        ]


def run_evaluation(
    swag_inference: SWAInferenceHandler,
    eval_dataset: torch.utils.data.Dataset,
    extended_evaluation: bool,
    output_location: pathlib.Path,
) -> None:
    """
    Run evaluation with your model.
    """

    print("Evaluating model on validation data")

    images, _, _, labels = eval_dataset.tensors

    all_pred_probabilities = swag_inference.predict_probs(images)
    max_pred_probabilities, argmax_pred_labels = torch.max(all_pred_probabilities, dim=-1)

    predicted_labels = swag_inference.label_prediction(all_pred_probabilities)

    print(f"Using optimized temperature: {swag_inference.temperature:.6f}")
    print(f"Using optimized calibration threshold: {swag_inference._calibration_threshold:.6f}")

    final_cost = compute_cost(predicted_labels, labels).item()
    print(f"Cost with this threshold: {final_cost:.6f}")

    non_ambiguous_mask = labels != -1

    overall_accuracy = torch.mean((predicted_labels == labels).float()).item()
    non_ambiguous_accuracy = torch.mean((predicted_labels[non_ambiguous_mask] == labels[non_ambiguous_mask]).float()).item()
    non_ambiguous_argmax_accuracy = torch.mean(
        (argmax_pred_labels[non_ambiguous_mask] == labels[non_ambiguous_mask]).float()
    ).item()
    print(f"Accuracy (raw): {overall_accuracy:.4f}")
    print(f"Accuracy (non-ambiguous only, your predictions): {non_ambiguous_accuracy:.4f}")
    print(f"Accuracy (non-ambiguous only, predicting most-likely class): {non_ambiguous_argmax_accuracy:.4f}")

    threshold_values = [0.0] + list(torch.unique(max_pred_probabilities, sorted=True))
    costs = []
    for threshold in threshold_values:
        thresholded_predictions = torch.where(max_pred_probabilities <= threshold, -1 * torch.ones_like(predicted_labels), argmax_pred_labels)
        costs.append(compute_cost(thresholded_predictions, labels).item())
    best_threshold_index = np.argmin(costs)
    print(f"Best cost (if argmax used) {costs[best_threshold_index]} at threshold {threshold_values[best_threshold_index]}")
    print("Note that this threshold does not necessarily generalize to the test set!")

    calibration_data = calculate_calibration_curve(all_pred_probabilities.numpy(), labels.numpy(), num_bins=20)
    print("Validation ECE:", calibration_data["ece"])

    if extended_evaluation:
        print("Plotting reliability diagram")
        fig = paint_reliability_diagram(calibration_data)
        fig.savefig(output_location / "reliability_diagram.pdf")

        sorted_confidence_indices = torch.argsort(max_pred_probabilities)

        # Plot samples your model is most confident about
        print("Plotting most confident validation set predictions")
        most_confident_indices = sorted_confidence_indices[-10:]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_index = most_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(images[sample_index].permute(1, 2, 0).numpy())
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f"pred. {predicted_labels[sample_index]}, true {labels[sample_index]}")
                bar_colors = ["C0"] * 6
                if labels[sample_index] >= 0:
                    bar_colors[labels[sample_index]] = "C1"
                ax[row + 1, col].bar(
                    np.arange(6), all_pred_probabilities[sample_index].numpy(), tick_label=np.arange(6), color=bar_colors
                )
        fig.suptitle("Most confident predictions", size=20)
        fig.savefig(output_location / "examples_most_confident.pdf")

        # Plot samples your model is least confident about
        print("Plotting least confident validation set predictions")
        least_confident_indices = sorted_confidence_indices[:10]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_index = least_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(images[sample_index].permute(1, 2, 0).numpy())
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f"pred. {predicted_labels[sample_index]}, true {labels[sample_index]}")
                bar_colors = ["C0"] * 6
                if labels[sample_index] >= 0:
                    bar_colors[labels[sample_index]] = "C1"
                ax[row + 1, col].bar(
                    np.arange(6), all_pred_probabilities[sample_index].numpy(), tick_label=np.arange(6), color=bar_colors
                )
        fig.suptitle("Least confident predictions", size=20)
        fig.savefig(output_location / "examples_least_confident.pdf")


class CNN(torch.nn.Module):
    """
    Small convolutional neural network used in this task.
    You should not modify this class before passing the baseline.
    """
    def __init__(
        self,
        in_channels: int,
        out_classes: int,
    ):
        super().__init__()

        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=5),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.pool1 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.pool2 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
        )

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.linear = torch.nn.Linear(64, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2(x)
        x = self.layer5(x)

        # Average features over both spatial dimensions
        x = self.global_pool(x).squeeze(-1).squeeze(-1)

        return self.linear(x)


if __name__ == "__main__":
    main()
