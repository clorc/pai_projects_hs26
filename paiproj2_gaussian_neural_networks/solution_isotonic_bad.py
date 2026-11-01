import abc
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

from sklearn.isotonic import IsotonicRegression

from util import paint_reliability_diagram, compute_cost, seed_setup, calculate_calibration_curve

ENABLE_EXTENDED_ANALYSIS = True
"""
Set `ENABLE_EXTENDED_ANALYSIS` to `True` in order to generate additional plots on validation data.
"""

USE_PRETRAINED_WEIGHTS = True
"""
If `USE_PRETRAINED_WEIGHTS` is `True`, then MAP inference uses provided pretrained weights.
You should not modify MAP training or the CNN architecture before passing the baseline.
If you set the constant to `False` (to further experiment),
this solution always performs MAP inference before running your SWAG implementation.
Note that MAP inference can take a long time.
"""


def main():

    data_location = pathlib.Path.cwd()
    model_location = pathlib.Path.cwd()
    output_location = pathlib.Path.cwd()

    # Load training data
    training_images = torch.from_numpy(np.load(data_location / "train_xs.npz")["train_xs"])
    training_metadata = np.load(data_location / "train_ys.npz")
    training_labels = torch.from_numpy(training_metadata["train_ys"])
    training_snow_labels = torch.from_numpy(training_metadata["train_is_snow"])
    training_cloud_labels = torch.from_numpy(training_metadata["train_is_cloud"])
    training_dataset = torch.utils.data.TensorDataset(training_images, training_snow_labels, training_cloud_labels, training_labels)

    # Load validation data
    validation_images = torch.from_numpy(np.load(data_location / "val_xs.npz")["val_xs"])
    validation_metadata = np.load(data_location / "val_ys.npz")
    validation_labels = torch.from_numpy(validation_metadata["val_ys"])
    validation_snow_labels = torch.from_numpy(validation_metadata["val_is_snow"])
    validation_cloud_labels = torch.from_numpy(validation_metadata["val_is_cloud"])
    validation_dataset = torch.utils.data.TensorDataset(validation_images, validation_snow_labels, validation_cloud_labels, validation_labels)

    # Fix all randomness
    seed_setup()

    # Build and run the actual solution
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    swag_inference = SWAInferenceHandler(
        train_xs=training_dataset.tensors[0],
        model_dir=model_location,
    )
    swag_inference.train_model(training_loader)
    swag_inference.run_calibration(validation_dataset)

    # fork_rng ensures that the evaluation does not change the rng state.
    with torch.random.fork_rng():
        run_evaluation(swag_inference, validation_dataset, ENABLE_EXTENDED_ANALYSIS, output_location)


class InferenceMode(enum.Enum):
    """
    Inference mode switch for your implementation.
    `MAP` simply predicts the most likely class using pretrained MAP weights.
    `SWAG_DIAGONAL` and `SWAG_FULL` correspond to SWAG-diagonal and the full SWAG method, respectively.
    """
    MAP = 0
    SWAG_DIAGONAL = 1
    SWAG_FULL = 2


class SWAInferenceHandler(object):
    """
    Your implementation of SWA-Gaussian.
    """

    def __init__(
        self,
        train_xs: torch.Tensor,
        model_dir: pathlib.Path,
        inference_mode: InferenceMode = InferenceMode.SWAG_FULL, # Set to Full SWAG
        # --- REVERTED ---
        # Back to the 30/15/30 setup from your 0.891 and 0.917 files
        swag_training_epochs: int = 30,
        swag_lr: float = 0.045,
        swag_update_interval: int = 1,
        max_rank_deviation_matrix: int = 25, # Reverted
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

        # Calibration
        self._calibration_threshold = 0.5  # Default, will be overwritten
        
        # --- NEW ---
        # Add temperature parameter for calibration
        self.temperature = 1.0 # Will be optimized in run_calibration

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
                for batch_images, batch_snow_labels, batch_cloud_labels, batch_labels in loader:
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

    # --- NEW HELPER FUNCTION ---
    # This is a copy of predict_probabilities_map, but returns LOGITS
    def _predict_logits_map(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Predict logits assuming that self.network is a MAP estimate.
        """
        all_predictions = []
        for (batch_images,) in loader:
            all_predictions.append(self.network(batch_images))

        all_predictions = torch.cat(all_predictions)
        return all_predictions # Return logits, not softmax

    def run_calibration(self, validation_data: torch.utils.data.Dataset) -> None:
        """
        Calibrate your predictions using a small validation set.
        
        --- CHANGED ---
        This function now performs two steps:
        1.  Isotonic Regression: Fits an isotonic regression model to calibrate
            the predicted probabilities using non-ambiguous validation samples.
        2.  Threshold Optimization: Uses the new calibrated probabilities
            to find the threshold that minimizes the cost function.
        """
        if self.inference_mode == InferenceMode.MAP:
            self._calibration_threshold = 0.0
            self.temperature = 1.0
            return

        print("Running calibration...")
        val_images, _, _, val_labels = validation_data.tensors
        
        # --- 1. Get BMA-averaged probabilities for the validation set ---
        print("Gathering BMA probabilities for calibration...")
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_images),
            batch_size=32, shuffle=False, num_workers=0, drop_last=False
        )
        
        model_probs = []
        with torch.random.fork_rng(), torch.no_grad():
            for _ in tqdm.tqdm(range(self.num_bma_samples), desc="Sampling for calibration", leave=False):
                self.sample_parameters()
                model_probs.append(self.predict_probabilities_map(loader))
        
        stacked_probs = torch.stack(model_probs)
        avg_probs = torch.mean(stacked_probs, dim=0)

        # Filter out ambiguous samples (label == -1) for calibration
        non_ambiguous_mask = val_labels != -1
        probs_for_cal = avg_probs[non_ambiguous_mask]
        labels_for_cal = val_labels[non_ambiguous_mask]

        # --- 2. Fit Isotonic Regression for each class ---
        print(f"Fitting isotonic regression on {len(labels_for_cal)} non-ambiguous samples...")
        
        # Convert to numpy for scikit-learn
        probs_np = probs_for_cal.numpy()
        labels_np = labels_for_cal.numpy()
        
        # Create one-vs-all labels for each class
        from sklearn.isotonic import IsotonicRegression
        
        self.isotonic_regressors = []
        calibrated_probs = np.zeros_like(probs_np)
        
        for class_idx in range(6):
            # Create binary labels for this class
            binary_labels = (labels_np == class_idx).astype(np.float32)
            class_probs = probs_np[:, class_idx]
            
            # Fit isotonic regression
            ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            ir.fit(class_probs, binary_labels)
            self.isotonic_regressors.append(ir)
            
            # Transform probabilities for this class
            calibrated_probs[:, class_idx] = ir.transform(class_probs)
        
        # Normalize the calibrated probabilities to sum to 1
        calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
        calibrated_probs = torch.from_numpy(calibrated_probs).float()

        # --- 3. Find Optimal Cost Threshold ---
        print("Finding optimal cost threshold with calibrated probabilities...")
        max_cal_probs, argmax_labels = torch.max(calibrated_probs, dim=-1)

        threshold_values = torch.cat(
            (torch.tensor([0.0]), torch.unique(max_cal_probs, sorted=True), torch.tensor([1.01]))
        )
        costs = []

        for threshold in threshold_values:
            thresholded_predictions = torch.where(
                max_cal_probs >= threshold,
                argmax_labels,
                -1 * torch.ones_like(argmax_labels)
            )
            costs.append(compute_cost(thresholded_predictions, labels_for_cal).item())

        best_cost = np.min(costs)
        best_threshold = threshold_values[np.argmin(costs)].item()

        self._calibration_threshold = best_threshold
        self.temperature = 1.0  # Keep temperature for compatibility, but it won't be used
        print(f"Found best threshold: {self._calibration_threshold:.6f} (Val Cost: {best_cost:.6f})")

    def predict_probabilities_swag(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Perform Bayesian model averaging using your SWAG statistics and predict
        probabilities for all samples in the loader.
        """
        self.network.eval()

        model_predictions = []
        for _ in tqdm.trange(self.num_bma_samples, desc="Performing Bayesian model averaging"):
            self.sample_parameters() 

            sample_predictions = []
            with torch.no_grad():
                for (batch_images,) in loader:
                    logits = self.network(batch_images)
                    probs = torch.softmax(logits, dim=-1)
                    sample_predictions.append(probs)
            
            model_predictions.append(torch.cat(sample_predictions, dim=0))

        stacked_predictions = torch.stack(model_predictions)
        bma_probabilities = torch.mean(stacked_predictions, dim=0)
        
        # Apply isotonic calibration if regressors are available
        if hasattr(self, 'isotonic_regressors'):
            bma_probabilities = self._apply_isotonic_calibration(bma_probabilities)
        
        return bma_probabilities

    def sample_parameters(self) -> None:
        """
        Sample a new network from the approximate SWAG posterior.

        --- CHANGED ---
        Re-applying the correct sampling formula from the SWAG paper.
        theta = mean + (1/sqrt(2)) * diag_sample + (1/sqrt(2(K-1))) * low_rank_sample
        """

        for name, param in self.network.named_parameters():
            z_diag = torch.randn(param.size())
            
            mean_weights = self.mean_weights[name]
            
            variance = torch.clamp(self.squared_weights[name] - mean_weights**2, min=1e-30)
            std_weights = torch.sqrt(variance)
            
            assert mean_weights.size() == param.size() and std_weights.size() == param.size()

            # --- BUG FIX ---
            # Diagonal part, scaled by 1/sqrt(2)
            diag_sample = std_weights * z_diag / math.sqrt(2.0)
            sampled_weight = mean_weights + diag_sample

            # Full SWAG part
            if self.inference_mode == InferenceMode.SWAG_FULL:
                K = len(self.deviation_vectors)
                low_rank_component = torch.zeros_like(param)
                
                if K > 1:
                    # This term is already scaled by 1/sqrt(2(K-1))
                    z_rank = torch.randn(K) / math.sqrt(2.0 * (K - 1))
                    
                    for i, deviation_dict in enumerate(self.deviation_vectors):
                        low_rank_component += z_rank[i] * deviation_dict[name]
                
                    sampled_weight += low_rank_component


            param.data = sampled_weight

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
        PRETRAINED_WEIGHTS_FILE = self.model_dir / "map_weights.pt"
        if USE_PRETRAINED_WEIGHTS:
            self.network.load_state_dict(torch.load(PRETRAINED_WEIGHTS_FILE))
            print("Loaded pretrained MAP weights from", PRETRAINED_WEIGHTS_FILE)
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
        start_decay_epoch = 30 # This was 30 in your 0.917 file
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
        self.network = self.network.eval()

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
            else:
                return self.predict_probabilities_swag(loader)

    def predict_probabilities_map(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Predict probabilities assuming that self.network is a MAP estimate.
        
        --- CHANGED ---
        Now applies self.temperature before the softmax.
        """
        all_predictions = []
        for (batch_images,) in loader:
            all_predictions.append(self.network(batch_images))

        all_predictions = torch.cat(all_predictions)
        
        # --- CHANGED ---
        # Apply temperature scaling
        return torch.softmax(all_predictions / self.temperature, dim=-1)

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

    def predict_probabilities_swag(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Perform Bayesian model averaging using your SWAG statistics and predict
        probabilities for all samples in the loader.
        """
        self.network.eval()

        model_predictions = []
        for _ in tqdm.trange(self.num_bma_samples, desc="Performing Bayesian model averaging"):
            self.sample_parameters() 

            sample_predictions = []
            with torch.no_grad():
                for (batch_images,) in loader:
                    logits = self.network(batch_images)
                    probs = torch.softmax(logits, dim=-1)
                    sample_predictions.append(probs)
            
            model_predictions.append(torch.cat(sample_predictions, dim=0))

        stacked_predictions = torch.stack(model_predictions)
        bma_probabilities = torch.mean(stacked_predictions, dim=0)
        
        # Apply isotonic calibration if regressors are available
        if hasattr(self, 'isotonic_regressors'):
            bma_probabilities = self._apply_isotonic_calibration(bma_probabilities)
        
        return bma_probabilities
    
    def _apply_isotonic_calibration(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Apply fitted isotonic regression calibration to probabilities.
        """
        if not hasattr(self, 'isotonic_regressors'):
            return probabilities
        
        probs_np = probabilities.numpy()
        calibrated_probs = np.zeros_like(probs_np)
        
        for class_idx, ir in enumerate(self.isotonic_regressors):
            calibrated_probs[:, class_idx] = ir.transform(probs_np[:, class_idx])
        
        # Normalize to ensure probabilities sum to 1
        calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
        return torch.from_numpy(calibrated_probs).float()

class SWAGScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Custom learning rate scheduler.
    Default implementation keeps the original learning rate constant.
    """

    def calculate_lr(self, current_epoch: float, previous_lr: float) -> float:
        """
        Calculate the learning rate for the epoch given by current_epoch.
        """
        # Implement a custom schedule if desired
    
        # t = ((current_epoch-1)%self.cycle_length + 1)/self.cycle_length
        # new_lr = (1- t)*self.alpha_min + (t)*self.alpha_max

        # return new_lr
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
        super().__init__(optimizer, last_epoch=-1, verbose=False)
        
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

    images, snow_labels, cloud_labels, labels = eval_dataset.tensors

    all_pred_probabilities = swag_inference.predict_probs(images)
    max_pred_probabilities, argmax_pred_labels = torch.max(all_pred_probabilities, dim=-1)
    
    predicted_labels = swag_inference.label_prediction(all_pred_probabilities)
    
    # --- NEW ---
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

        log_softmax = self.linear(x)

        return log_softmax


if __name__ == "__main__":
    main()