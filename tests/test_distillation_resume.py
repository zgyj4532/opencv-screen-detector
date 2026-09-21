"""A resumed LwF run must keep the original teacher, not a new random model."""

from dataclasses import replace

import torch

from experiment.cnn_fft_dwt_ablation import harness


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(3, 4)
        self.freq_branch = torch.nn.Linear(3, 4)
        self.spatial_norm = torch.nn.LayerNorm(4)
        self.freq_norm = torch.nn.LayerNorm(4)
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(8, 3))

    def forward(self, rgb, fft, dwt):
        spatial = self.spatial_norm(self.backbone(rgb))
        frequency = self.freq_norm(self.freq_branch(fft + dwt))
        return self.classifier(torch.cat([spatial, frequency], dim=1))


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        _, label = self.samples[index]
        features = torch.tensor([index / 10, 0.25, 0.75], dtype=torch.float32)
        return features, features * 0.3, features * 0.7, label


def test_distillation_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(harness, "EXP_DIR", tmp_path)
    monkeypatch.setattr(harness, "CachedDataset", TinyDataset)
    monkeypatch.setattr(harness, "create_model", lambda **_kwargs: TinyModel())
    monkeypatch.setattr(harness, "unfreeze_stages", lambda *_args: None)
    monkeypatch.setattr(harness, "load_canary_paths", set)
    # This test is CPU-only and must not allocate another CUDA context beside training.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(987)
    init_path = tmp_path / "teacher.pth"
    torch.save({"model_state_dict": TinyModel().state_dict()}, init_path)
    samples = [(str(tmp_path / f"sample_{index}.png"), index % 3) for index in range(9)]
    split = {"train": samples, "val": samples, "test": samples}
    cfg = harness.ExpConfig(
        id="continuous",
        seed=42,
        epochs_head=0,
        epochs_finetune=2,
        batch_size=3,
        lr=0.01,
        distill_alpha=1.0,
        init_checkpoint=str(init_path),
    )
    continuous = harness.train_one(cfg, split, "cpu", evaluate_test=False)
    resumed_cfg = replace(cfg, id="resumed")
    paused = harness.train_one(resumed_cfg, split, "cpu", evaluate_test=False, max_total_epochs=1)
    assert paused["status"] == "paused"
    resumed = harness.train_one(resumed_cfg, split, "cpu", evaluate_test=False, resume=True)
    expected = torch.load(tmp_path / "continuous/last.pth", map_location="cpu", weights_only=False)
    actual = torch.load(tmp_path / "resumed/last.pth", map_location="cpu", weights_only=False)
    for key, tensor in expected["model_state_dict"].items():
        torch.testing.assert_close(tensor, actual["model_state_dict"][key], rtol=0, atol=0)
    assert resumed["history"] == continuous["history"]
