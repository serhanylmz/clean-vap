import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import lightning as L

from os.path import isfile
import pandas as pd
import json
from typing import Optional, Mapping, Any
import matplotlib.pyplot as plt

import numpy as np # NEW!
from numba import njit, prange # NEW!

from vap.utils.audio import load_waveform, mono_to_stereo
from vap.utils.utils import vad_list_to_onehot
from vap.utils.plot import plot_melspectrogram, plot_vad


SAMPLE = Mapping[str, Tensor]

def stereo_mixer_with_delay(audio, sample_rate=16000):
    channels, seq_length = audio.shape
    max_delay_samples = int(0.3 * sample_rate)  # 300ms
    
    # Pad the input
    padded_left = torch.nn.functional.pad(audio[0], (max_delay_samples, max_delay_samples))
    padded_right = torch.nn.functional.pad(audio[1], (max_delay_samples, max_delay_samples))
    
    # Generate random scaling factors and delays
    left_to_right_scale = torch.rand(1).item() * 0.5
    right_to_left_scale = torch.rand(1).item() * 0.5
    left_to_right_delay = torch.randint(0, max_delay_samples, (1,)).item()
    right_to_left_delay = torch.randint(0, max_delay_samples, (1,)).item()
    
    # Mix channels
    mixed_left = padded_left[max_delay_samples:-max_delay_samples].clone()
    mixed_right = padded_right[max_delay_samples:-max_delay_samples].clone()
    
    mixed_left += right_to_left_scale * padded_right[max_delay_samples + right_to_left_delay : max_delay_samples + right_to_left_delay + seq_length]
    mixed_right += left_to_right_scale * padded_left[max_delay_samples + left_to_right_delay : max_delay_samples + left_to_right_delay + seq_length]
    
    # Normalize to prevent clipping
    max_val = torch.max(torch.abs(torch.stack([mixed_left, mixed_right])))
    if max_val > 1:
        mixed_left /= max_val
        mixed_right /= max_val
    
    return torch.stack((mixed_left, mixed_right))

def load_df(path: str) -> pd.DataFrame:
    def _vl(x):
        return json.loads(x)

    def _session(x):
        return str(x)

    converters = {
        "vad_list": _vl,
        "session": _session,
    }
    return pd.read_csv(path, converters=converters)


def force_correct_nsamples(w: Tensor, n_samples: int) -> Tensor:
    if w.shape[-1] == n_samples:
        return w
    if w.shape[-1] > n_samples:
        return w[..., :n_samples]
    else:
        diff = n_samples - w.shape[-1]
        z = torch.zeros((2, diff), dtype=w.dtype, device=w.device)
        w = torch.cat((w, z), dim=-1)
    return w

    # if w.shape[-1] < n_samples:
    #     ww[:, : w.shape[-1]] = w
    # else:
    #     ww = w[..., :n_samples]
    # return ww

    # if w.shape[-1] > n_samples:
    #     w = w[:, -n_samples:]
    #
    # elif w.shape[-1] < n_samples:
    #     w = torch.cat([w, torch.zeros_like(w)[:, : n_samples - w.shape[-1]]], dim=-1)
    #
    # return w


def plot_dset_sample(d):
    """
    VAD is by default longer than the audio (prediction horizon)
    So you will see zeros at the end where the VAD is defined but the audio not.
    """
    fig, ax = plt.subplots(2, 1, figsize=(9, 6))
    ax[0].set_title(d["session"])
    plot_melspectrogram(d["waveform"], ax=ax[:2])
    x = torch.arange(d["vad"].shape[0]) / dset.frame_hz
    plot_vad(x, d["vad"][:, 0], ax[0], ypad=2, label="VAD 0")
    plot_vad(x, d["vad"][:, 1], ax[1], ypad=2, label="VAD 1")
    _ = [a.legend() for a in ax]
    ax[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


class VAPDataset(Dataset):
    def __init__(
        self,
        path: str,
        horizon: float = 2,
        duration: float = 20,
        sample_rate: int = 16_000,
        frame_hz: int = 50,
        mono: bool = False,
        apply_mixing: bool = True,  # New parameter
    ) -> None:
        self.path = path
        self.df = load_df(path)

        self.sample_rate = sample_rate
        self.frame_hz = frame_hz
        self.horizon = horizon
        self.mono = mono
        self.apply_mixing = apply_mixing  # New attribute

        self.duration = duration
        self.n_samples = int(self.duration * self.sample_rate)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> SAMPLE:
        d = self.df.iloc[idx]
        dur = round(d["end"] - d["start"])
        w, _ = load_waveform(
            d["audio_path"],
            start_time=d["start"],
            end_time=d["end"],
            sample_rate=self.sample_rate,
            mono=self.mono,
        )

        w = force_correct_nsamples(w, self.n_samples)

        if not self.mono and w.shape[0] == 1:
            w = mono_to_stereo(w, d["vad_list"], sample_rate=self.sample_rate)

        # print(w.device)
        # print(w.shape)
        # print(w.dtype)

        # Apply stereo mixing if enabled
        if self.apply_mixing and not self.mono:
            w = stereo_mixer_with_delay(w, self.sample_rate)

        vad = vad_list_to_onehot(
            d["vad_list"], duration=dur + self.horizon, frame_hz=self.frame_hz
        )

        return {
            "session": d.get("session", ""),
            "waveform": w,
            "vad": vad,
            "dataset": d.get("dataset", ""),
        }


class VAPDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        horizon: float = 2,
        sample_rate: int = 16000,
        frame_hz: int = 50,
        mono: bool = False,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        apply_mixing: bool = True,  # NEW!
        **kwargs,
    ):
        super().__init__()

        # Files
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        # values
        self.mono = mono
        self.horizon = horizon
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz

        # DataLoder
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        # Mixing
        self.apply_mixing = apply_mixing  # NEW!

    def __repr__(self):
        s = self.__class__.__name__
        s += f"\n\tTrain: {self.train_path}"
        s += f"\n\tVal: {self.val_path}"
        s += f"\n\tTest: {self.test_path}"
        s += f"\n\tHorizon: {self.horizon}"
        s += f"\n\tSample rate: {self.sample_rate}"
        s += f"\n\tFrame Hz: {self.frame_hz}"
        s += f"\nData"
        s += f"\n\tbatch_size: {self.batch_size}"
        s += f"\n\tpin_memory: {self.pin_memory}"
        s += f"\n\tnum_workers: {self.num_workers}"
        s += f"\n\tprefetch_factor: {self.prefetch_factor}"
        return s

    def prepare_data(self):
        if self.train_path is not None:
            if not isfile(self.train_path):
                print("WARNING: no TRAINING data found: ", self.train_path)

        if self.val_path is not None:
            if not isfile(self.val_path):
                print("WARNING: no VALIDATION data found: ", self.val_path)

        if self.test_path is not None:
            if not isfile(self.test_path):
                print("WARNING: no TEST data found: ", self.test_path)

    def setup(self, stage: Optional[str] = "fit"):
        """Loads the datasets"""

        if stage in (None, "fit"):
            assert self.train_path is not None, "TRAIN path is None"
            assert self.val_path is not None, "VAL path is None"
            assert isfile(self.train_path), f"TRAIN path not found: {self.train_path}"
            assert isfile(self.val_path), f"VAL path not found: {self.val_path}"
            self.train_dset = VAPDataset(
                self.train_path,
                horizon=self.horizon,
                sample_rate=self.sample_rate,
                frame_hz=self.frame_hz,
                mono=self.mono,
                apply_mixing=self.apply_mixing,  # NEW!
            )
            self.val_dset = VAPDataset(
                self.val_path,
                horizon=self.horizon,
                sample_rate=self.sample_rate,
                frame_hz=self.frame_hz,
                mono=self.mono,
                apply_mixing=self.apply_mixing,  # NEW!
            )

        if stage in (None, "test"):
            assert self.test_path is not None, "TEST path is None"
            assert isfile(self.test_path), f"TEST path not found: {self.test_path}"
            self.test_dset = VAPDataset(
                self.test_path,
                horizon=self.horizon,
                sample_rate=self.sample_rate,
                frame_hz=self.frame_hz,
                mono=self.mono,
                apply_mixing=self.apply_mixing,  # NEW!
            )

    def collate_fn(self, batch: list[dict[str, Any]]):
        batch_stacked = {k: [] for k in batch[0].keys()}

        for b in batch:
            batch_stacked["session"].append(b["session"])
            batch_stacked["dataset"].append(b["dataset"])
            batch_stacked["waveform"].append(b["waveform"])
            batch_stacked["vad"].append(b["vad"])

        batch_stacked["waveform"] = torch.stack(batch_stacked["waveform"])
        batch_stacked["vad"] = torch.stack(batch_stacked["vad"])
        return batch_stacked
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
            shuffle=False,
        )


if __name__ == "__main__":

    from argparse import ArgumentParser
    from lightning import seed_everything
    from tqdm import tqdm

    seed_everything(0)

    parser = ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/sliding_window_dset.csv")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()

    if args.single:
        dset = VAPDataset(path=args.csv)
        idx = int(torch.randint(0, len(dset), (1,)).item())
        d = dset[idx]
        plot_dset_sample(d)

    else:

        dm = VAPDataModule(
            # train_path="data/splits/sliding_window_dset_train.csv",
            # val_path="data/splits/sliding_window_dset_val.csv",
            test_path=args.csv,
            batch_size=args.batch_size,  # as much as fit on gpu with model and max cpu cores
            num_workers=args.num_workers,  # number cpu cores
            prefetch_factor=args.prefetch_factor,  # how many batches to prefetch
        )
        dm.prepare_data()
        dm.setup("test")
        print(dm)
        print("VAPDataModule: ", len(dm.test_dset))
        dloader = dm.test_dataloader()
        batch = next(iter(dloader))
        for batch in tqdm(
            dloader,
            total=len(dloader),
            desc="Running VAPDatamodule (Training wont be faster than this...)",
        ):
            pass
