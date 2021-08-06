import math
import random

import julius
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffq import DiffQuantizer

import torch.hub
import io
import zlib
from pathlib import Path


###############################################################################
# Model definition
###############################################################################


class BLSTM(nn.Module):
    def __init__(self, dim, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim
        )
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        return x


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class Demucs(nn.Module):
    def __init__(
        self,
        sources=["drums", "bass", "other", "vocals"],
        audio_channels=2,
        channels=64,
        depth=6,
        rewrite=True,
        glu=True,
        rescale=0.1,
        resample=True,
        kernel_size=8,
        stride=4,
        growth=2.0,
        lstm_layers=2,
        context=3,
        samplerate=44100,
        segment_length=4 * 10 * 44100,
    ):
        """
        Args:
            sources (list[str]): list of source names
            audio_channels (int): stereo or mono
            channels (int): first convolution channels
            depth (int): number of encoder/decoder layers
            rewrite (bool): add 1x1 convolution to each encoder layer
                and a convolution to each decoder layer.
                For the decoder layer, `context` gives the kernel size.
            glu (bool): use glu instead of ReLU
            resample_input (bool): upsample x2 the input and downsample /2 the output.
            rescale (int): rescale initial weights of convolutions
                to get their standard deviation closer to `rescale`
            kernel_size (int): kernel size for convolutions
            stride (int): stride for convolutions
            growth (float): multiply (resp divide) number of channels by that
                for each layer of the encoder (resp decoder)
            lstm_layers (int): number of lstm layers, 0 = no lstm
            context (int): kernel size of the convolution in the
                decoder before the transposed convolution. If > 1,
                will provide some context from neighboring time
                steps.
            samplerate (int): stored as meta information for easing
                future evaluations of the model.
            segment_length (int): stored as meta information for easing
                future evaluations of the model. Length of the segments on which
                the model was trained.
        """

        super().__init__()
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.resample = resample
        self.channels = channels
        self.samplerate = samplerate
        self.segment_length = segment_length

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1
        in_channels = audio_channels
        for index in range(depth):
            encode = []
            encode += [nn.Conv1d(in_channels, channels, kernel_size, stride), nn.ReLU()]
            if rewrite:
                encode += [nn.Conv1d(channels, ch_scale * channels, 1), activation]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = len(self.sources) * audio_channels
            if rewrite:
                decode += [
                    nn.Conv1d(channels, ch_scale * channels, context),
                    activation,
                ]
            decode += [nn.ConvTranspose1d(channels, out_channels, kernel_size, stride)]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels = int(growth * channels)

        channels = in_channels

        if lstm_layers:
            self.lstm = BLSTM(channels, lstm_layers)
        else:
            self.lstm = None

        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length when context = 1. If context > 1,
        the two signals can be center trimmed to match.

        For training, extracts should have a valid length.For evaluation
        on full tracks we recommend passing `pad = True` to :method:`forward`.
        """
        if self.resample:
            length *= 2
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1
        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        if self.resample:
            length = math.ceil(length / 2)
        return int(length)

    def forward(self, mix):
        x = mix

        if self.resample:
            x = julius.resample_frac(x, 1, 2)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)
        if self.lstm:
            x = self.lstm(x)
        for decode in self.decoder:
            skip = center_trim(saved.pop(-1), x)
            x = x + skip
            x = decode(x)

        if self.resample:
            x = julius.resample_frac(x, 2, 1)

        x = x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1))
        return x


def load_model(name):
    model = Demucs()

    model_hash = {"demucs": "e07c671f", "demucs_quantized": "07afea75"}
    cp = name + "-" + model_hash[name] + ".th"

    if cp.exists():
        state = torch.load(cp)
    else:
        root = "https://dl.fbaipublicfiles.com/demucs/v3.0/"
        state = torch.hub.load_state_dict_from_url(
            root + cp, map_location="cpu", check_hash=True
        )

    if "quantized" in name:
        quantizer = DiffQuantizer(model, group_size=8, min_size=1)
        buf = io.BytesIO(zlib.decompress(state["compressed"]))
        state = torch.load(buf, "cpu")
        quantizer.restore_quantized_state(state)
        quantizer.detach()
    else:
        model.load_state_dict(state)

    return model


###############################################################################
# Data processing
###############################################################################


def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1: downmix channels
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2: only one channel, replicate.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3: take first N
        wav = wav[..., :channels, :]
    else:
        raise ValueError(
            "The audio file has less channels than requested but is not mono."
        )
    return wav


def preprocess_and_normalize_audio(wav, current_samplerate, audio_channels, samplerate):
    wav = convert_audio_channels(wav, audio_channels)
    wav = julius.resample_frac(wav, current_samplerate, samplerate)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    return wav, ref


class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        self.tensor = tensor
        self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, torch.Tensor)
        return TensorChunk(tensor_or_chunk)


def center_trim(tensor, reference):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
    return tensor


###############################################################################
# Running the model
###############################################################################


def run_model_with_splits_and_shifts(
    model, mix, shifts=None, split=False, overlap=0.25, transition_power=1.0
):
    """
    Apply model according to shifts or splits, merging out the outputs.

    Args:
        shifts (int): if > 0, will shift in time `mix` by a random amount between 0 and 0.5 sec
            and apply the oppositve shift to the output. This is repeated `shifts` time and
            all predictions are averaged. This effectively makes the model time equivariant
            and improves SDR by up to 0.2 points.
        split (bool): if True, the input will be broken down in 8 seconds extracts
            and predictions will be performed individually on each and concatenated.
            Useful for model with large memory footprint like Tasnet.
    """
    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."

    device = mix.device
    channels, length = mix.shape

    if split:
        out = torch.zeros(len(model.sources), channels, length, device=device)
        sum_weight = torch.zeros(length, device=device)

        segment = model.segment_length  # e.g. (8 * 44100)
        stride = int((1 - overlap) * segment)  # e.g. (1 - 0.25) * 8 = (6 * 44100)
        offsets = range(
            0, length, stride
        )  # e.g. (0, 6 * 44100, 12 * 44100, 18, 24, ...)

        # Split merging weights = normalized triangle, to a power exponent
        up = torch.arange(1, segment // 2 + 1)
        down = torch.arange(segment - segment // 2, 0, -1)
        tri = torch.cat([up, down]).to(device)
        weight = (tri / tri.max()) ** transition_power
        assert len(weight) == segment

        # e.g. chunks = (0, 8), (6, 14), (12, 20), (18, 26)...
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment)
            chunk_out = run_model_with_splits_and_shifts(model, chunk, shifts=shifts)
            chunk_length = chunk_out.shape[-1]
            out[..., offset : offset + segment] += weight[:chunk_length] * chunk_out
            sum_weight[offset : offset + segment] += weight[:chunk_length]
            offset += segment

        assert sum_weight.min() > 0
        out /= sum_weight
        return out

    elif shifts:
        max_shift = int(0.5 * model.samplerate)
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0
        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            shifted_out = run_model_with_splits_and_shifts(model, shifted)
            out += shifted_out[..., max_shift - offset :]
        out /= shifts
        return out

    else:
        valid_length = model.valid_length(length)
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(valid_length)
        with torch.no_grad():
            out = model(padded_mix.unsqueeze(0))[0]
        return center_trim(out, length)
