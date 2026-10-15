"""A scikit-learn compatible classifier for TabPFN-Tydra checkpoints."""

from __future__ import annotations

import json
import warnings
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


DEFAULT_REPO_ID = "Mieszko-Komisarczyk/Tydra"


class TydraClassifier(ClassifierMixin, BaseEstimator):
    """In-context tabular classifier backed by TabPFN-Tydra-4HT.

    ``fit`` stores the labeled rows as context. It does not update the neural
    network weights. The model files are downloaded from Hugging Face on the
    first call to ``fit`` unless ``repo_id`` points to a local directory.

    Parameters
    ----------
    repo_id:
        A Hugging Face model repository or a local directory containing
        ``config.json`` and ``model.safetensors``.
    revision:
        Optional Hugging Face revision (branch, tag, or commit).
    cache_dir:
        Optional Hugging Face download cache directory.
    device:
        CUDA device such as ``"cuda"`` or ``"cuda:1"``. ``"auto"`` selects
        CUDA automatically. Tydra's current Hydra kernels require an NVIDIA
        GPU and do not support CPU inference.
    n_ensemble_configurations:
        Number of class-permutation/preprocessing configurations to average.
    batch_size_inference:
        Number of ensemble configurations evaluated together.
    preprocess_transform:
        ``"mix"`` averages raw and power-transformed features; ``"none"``
        disables the additional power transform.
    categorical_features:
        Optional integer indices of categorical columns. These columns are not
        power-transformed when preprocessing is enabled.
    random_state:
        Seed used to order ensemble configurations.
    fp16_inference:
        Use CUDA autocast for model execution.
    """

    def __init__(
        self,
        repo_id: Union[str, Path] = DEFAULT_REPO_ID,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        device: str = "auto",
        n_ensemble_configurations: int = 4,
        batch_size_inference: int = 16,
        preprocess_transform: str = "mix",
        categorical_features: Optional[Sequence[int]] = None,
        random_state: Optional[int] = 0,
        fp16_inference: bool = False,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.device = device
        self.n_ensemble_configurations = n_ensemble_configurations
        self.batch_size_inference = batch_size_inference
        self.preprocess_transform = preprocess_transform
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.fp16_inference = fp16_inference

    def fit(self, X, y):
        """Store labeled context rows and load the pretrained model."""
        X, y = check_X_y(X, y, force_all_finite=False, dtype=np.float32)
        if np.isinf(X).any():
            raise ValueError("X may contain NaN values, but it may not contain infinity.")

        self._validate_parameters()
        self._ensure_model_loaded()

        if X.shape[1] > self.max_num_features_:
            raise ValueError(
                f"Tydra supports at most {self.max_num_features_} features; "
                f"received {X.shape[1]}."
            )

        self.label_encoder_ = LabelEncoder().fit(y)
        if len(self.label_encoder_.classes_) < 2:
            raise ValueError("Tydra requires at least two classes.")
        if len(self.label_encoder_.classes_) > self.max_num_classes_:
            raise ValueError(
                f"Tydra supports at most {self.max_num_classes_} classes; "
                f"received {len(self.label_encoder_.classes_)}."
            )

        self.classes_ = self.label_encoder_.classes_
        self.X_ = X
        self.y_ = self.label_encoder_.transform(y).astype(np.float32, copy=False)
        self.n_features_in_ = X.shape[1]

        recommended_context = int(self.config_.get("max_eval_pos", 1000))
        if len(X) > recommended_context:
            warnings.warn(
                f"The checkpoint was trained with at most {recommended_context} context "
                f"rows, but fit received {len(X)}. Inference can be slower and less reliable.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def predict_proba(self, X):
        """Predict class probabilities for query rows."""
        check_is_fitted(self, ("X_", "y_", "model_"))
        X = check_array(X, force_all_finite=False, dtype=np.float32)
        if np.isinf(X).any():
            raise ValueError("X may contain NaN values, but it may not contain infinity.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but fit was called with "
                f"{self.n_features_in_} features."
            )

        from tabpfn.scripts.transformer_prediction_interface import transformer_predict

        eval_position = len(self.X_)
        X_all = np.concatenate((self.X_, X), axis=0)
        y_all = np.concatenate((self.y_, np.zeros(len(X), dtype=np.float32)))
        X_tensor = torch.from_numpy(X_all).unsqueeze(1).to(self.device_)
        y_tensor = torch.from_numpy(y_all).unsqueeze(1).to(self.device_)

        probabilities, _ = transformer_predict(
            self.model_,
            X_tensor,
            y_tensor,
            eval_position,
            device=str(self.device_),
            max_features=self.max_num_features_,
            inference_mode=True,
            preprocess_transform=self.preprocess_transform,
            categorical_feats=(
                []
                if self.categorical_features is None
                else list(self.categorical_features)
            ),
            N_ensemble_configurations=self.n_ensemble_configurations,
            batch_size_inference=self.batch_size_inference,
            normalize_to_ranking=bool(self.config_.get("normalize_to_ranking", False)),
            normalize_with_sqrt=bool(self.config_.get("normalize_with_sqrt", False)),
            multiclass_decoder="permutation",
            feature_shift_decoder=False,
            differentiable_hps_as_style=bool(
                self.config_.get("differentiable_hps_as_style", False)
            ),
            fp16_inference=self.fp16_inference,
            seed=self.random_state,
            no_grad=True,
        )
        return probabilities.squeeze(0).detach().cpu().numpy()

    def predict(self, X):
        """Predict class labels for query rows."""
        indices = np.argmax(self.predict_proba(X), axis=1)
        return self.label_encoder_.inverse_transform(indices)

    def _validate_parameters(self) -> None:
        if self.n_ensemble_configurations < 1:
            raise ValueError("n_ensemble_configurations must be at least 1.")
        if self.batch_size_inference < 1:
            raise ValueError("batch_size_inference must be at least 1.")
        if self.preprocess_transform not in {"mix", "none", "power", "power_all"}:
            raise ValueError(
                "preprocess_transform must be one of: 'mix', 'none', 'power', "
                "or 'power_all'."
            )

    def _resolve_device(self) -> torch.device:
        requested = "cuda" if self.device == "auto" else self.device
        device = torch.device(requested)
        if device.type != "cuda":
            raise ValueError(
                "Tydra currently requires an NVIDIA CUDA GPU because its Hydra "
                "layers use Triton CUDA kernels."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Install a CUDA-enabled PyTorch build and run "
                "Tydra on an NVIDIA GPU."
            )
        return device

    def _model_files(self) -> tuple[Path, Path]:
        local_path = Path(self.repo_id).expanduser()
        if local_path.is_dir():
            return local_path / "config.json", local_path / "model.safetensors"

        download_kwargs = {
            "repo_id": str(self.repo_id),
            "revision": self.revision,
            "cache_dir": str(self.cache_dir) if self.cache_dir is not None else None,
        }
        config_path = hf_hub_download(filename="config.json", **download_kwargs)
        weights_path = hf_hub_download(filename="model.safetensors", **download_kwargs)
        return Path(config_path), Path(weights_path)

    def _ensure_model_loaded(self) -> None:
        if hasattr(self, "model_"):
            return

        self.device_ = self._resolve_device()
        try:
            import tabpfn.encoders as encoders
            from tabpfn.hybrid import HybridHydraTransformerModel
        except ImportError as error:
            raise ImportError(
                "Tydra's CUDA dependencies are missing. Install causal-conv1d and "
                "mamba-ssm==2.2.2 as described in the README."
            ) from error

        config_path, weights_path = self._model_files()
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"Expected config.json and model.safetensors in {self.repo_id!s}."
            )

        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        if config.get("architecture") != "HybridHydraTransformerModel":
            raise ValueError(
                "Unsupported checkpoint architecture: "
                f"{config.get('architecture', '<missing>')}."
            )

        uses_nan_encoder = any(
            float(config.get(key, 0.0)) > 0.0
            for key in ("nan_prob_no_reason", "nan_prob_a_reason", "nan_prob_unknown_reason")
        )
        encoder_factory = (
            encoders.NanHandlingEncoder
            if uses_nan_encoder
            else partial(encoders.Linear, replace_nan_by_zero=True)
        )
        emsize = int(config["emsize"])
        max_num_classes = int(config["max_num_classes"])
        y_encoder_factory = (
            encoders.get_Canonical(max_num_classes)
            if config.get("canonical_y_encoder", False)
            else encoders.Linear
        )
        model = HybridHydraTransformerModel(
            encoder=encoder_factory(int(config["num_features"]), emsize),
            n_out=max_num_classes,
            ninp=emsize,
            nhead=int(config["nhead"]),
            nhid=emsize * int(config["nhid_factor"]),
            nlayers=int(config["nlayers"]),
            y_encoder=y_encoder_factory(1, emsize),
            dropout=float(config["dropout"]),
            efficient_eval_masking=bool(config["efficient_eval_masking"]),
            full_attention=bool(config.get("enable_transformer_full_attn", False)),
            layer_types=config.get("hybrid_layer_types"),
        )
        model.criterion = torch.nn.CrossEntropyLoss(
            reduction="none", weight=torch.ones(max_num_classes)
        )
        state_dict = load_file(str(weights_path), device="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device_).eval()

        self.model_ = model
        self.config_ = config
        self.max_num_features_ = int(config["num_features"])
        self.max_num_classes_ = max_num_classes
