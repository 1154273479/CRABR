from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import SwinTinyEncoder
from models.model import ERGASegmenter, PlainUNet


MODEL_CHOICES = (
    "crab",
    "swin_unet",
    "u_mamba",
    "vm_unet_v2",
    "msvm_unet",
    "segmamba",
    "kmunet",
    "dcm_net",
    "cfm_unet",
    "i2u_net",
    "tbconvl_net",
)


def _out_channels(cfg: Dict) -> int:
    return 1 if cfg["model"]["task_mode"] == "binary" else int(cfg["model"]["num_classes"])


def _in_channels(cfg: Dict) -> int:
    return int(cfg["data"].get("in_channels", 1))


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(ConvBNAct(in_channels, out_channels), ConvBNAct(out_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSegmenter(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.net = PlainUNet(_in_channels(cfg), _out_channels(cfg), int(cfg["model"].get("classic_unet_base_channels", 64)))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"seg_logits": self.net(x)}


class FPNDecoder(nn.Module):
    def __init__(self, channels: Sequence[int], out_channels: int, decoder_channels: int = 256) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(ch, decoder_channels, kernel_size=1) for ch in channels])
        self.smooth = nn.ModuleList([ConvBNAct(decoder_channels, decoder_channels) for _ in channels])
        self.head = nn.Sequential(ConvBNAct(decoder_channels * len(channels), decoder_channels), nn.Conv2d(decoder_channels, out_channels, 1))

    def forward(self, features: Sequence[torch.Tensor], output_size: Sequence[int]) -> torch.Tensor:
        laterals = [proj(feat) for proj, feat in zip(self.lateral, features)]
        x = laterals[-1]
        pyramid = [self.smooth[-1](x)]
        for idx in range(len(laterals) - 2, -1, -1):
            x = F.interpolate(x, size=laterals[idx].shape[-2:], mode="bilinear", align_corners=False) + laterals[idx]
            pyramid.append(self.smooth[idx](x))
        target = features[0].shape[-2:]
        pyramid = [F.interpolate(feat, size=target, mode="bilinear", align_corners=False) for feat in pyramid]
        logits = self.head(torch.cat(pyramid, dim=1))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class SwinUNetLiteSegmenter(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        base_dim = int(cfg["model"].get("swin_base_dim", cfg["model"].get("embed_dim", 96)))
        stage_dims = [base_dim, base_dim * 2, base_dim * 4, 512]
        default_heads = tuple(max(1, dim // 32) for dim in stage_dims)
        num_heads = tuple(cfg["model"].get("swin_num_heads", default_heads))
        if any(dim % head != 0 for dim, head in zip(stage_dims, num_heads)):
            raise ValueError(
                f"Invalid swin_num_heads={num_heads} for stage dims {stage_dims}. "
                "Each stage dim must be divisible by its num_heads."
            )
        self.encoder = SwinTinyEncoder(_in_channels(cfg), base_dim=base_dim, num_heads=num_heads)
        self.decoder = FPNDecoder(self.encoder.out_channels, _out_channels(cfg), decoder_channels=int(cfg["model"].get("decoder_dim", 160)))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"seg_logits": self.decoder(self.encoder(x), x.shape[-2:])}


class MambaLikeBlock(nn.Module):
    """2D state-space-style block used for recent Mamba-inspired baselines.

    This is a dependency-free supervised adaptation, not the official selective-scan kernel.
    """

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.BatchNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.scan_h = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, pad), groups=channels)
        self.scan_v = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0), groups=channels)
        self.out_proj = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        value, gate = self.in_proj(self.norm(x)).chunk(2, dim=1)
        value = self.scan_h(value) + self.scan_v(value)
        value = self.act(value) * torch.sigmoid(gate)
        return residual + self.out_proj(value)


class KANLikeBlock(nn.Module):
    """Lightweight KAN-inspired channel mixer with polynomial basis features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 16)
        self.norm = nn.BatchNorm2d(channels)
        self.reduce = nn.Conv2d(channels * 3, hidden, kernel_size=1, bias=False)
        self.expand = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        basis = torch.cat([z, z.square(), torch.sin(z)], dim=1)
        return x + self.expand(self.reduce(basis))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attn(x)


class HybridMambaBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mamba_layers: int = 1, kernel_size: int = 7) -> None:
        super().__init__()
        self.local = DoubleConv(in_channels, out_channels)
        self.context = nn.Sequential(*[MambaLikeBlock(out_channels, kernel_size=kernel_size) for _ in range(mamba_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.context(self.local(x))


class KANMambaBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.local = DoubleConv(in_channels, out_channels)
        self.kan = KANLikeBlock(out_channels)
        self.context = MambaLikeBlock(out_channels, kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.context(self.kan(self.local(x)))


class CoupledFusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.local = DoubleConv(in_channels, out_channels)
        self.global_branch = nn.Sequential(ConvBNAct(in_channels, out_channels, kernel_size=1), MambaLikeBlock(out_channels, kernel_size=11))
        self.fuse = nn.Sequential(
            ConvBNAct(out_channels * 2, out_channels, kernel_size=1),
            ChannelAttention(out_channels),
            ConvBNAct(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.local(x), self.global_branch(x)], dim=1))


class VSSBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mamba_layers: int = 2, kernel_size: int = 7) -> None:
        super().__init__()
        self.proj = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.blocks = nn.Sequential(*[MambaLikeBlock(out_channels, kernel_size=kernel_size) for _ in range(mamba_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


class MultiScaleVSSBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch_channels = max(out_channels // 4, 8)
        self.pre = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(out_channels, branch_channels, kernel_size=3, padding=1, groups=1),
                nn.Conv2d(out_channels, branch_channels, kernel_size=5, padding=2, groups=1),
                nn.Conv2d(out_channels, branch_channels, kernel_size=7, padding=3, groups=1),
                nn.Conv2d(out_channels, branch_channels, kernel_size=11, padding=5, groups=1),
            ]
        )
        self.fuse = ConvBNAct(branch_channels * len(self.branches), out_channels, kernel_size=1)
        self.context = MambaLikeBlock(out_channels, kernel_size=9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))
        return self.context(x)


class SDIGate(nn.Module):
    """Semantics-detail infusion gate for VM-UNetV2-style decoder skips."""

    def __init__(self, skip_channels: int, semantic_channels: int) -> None:
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_channels, skip_channels, kernel_size=1)
        self.semantic_proj = nn.Conv2d(semantic_channels, skip_channels, kernel_size=1)
        self.gate = nn.Sequential(nn.Conv2d(skip_channels, skip_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, skip: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        semantic = F.interpolate(semantic, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        infused = self.skip_proj(skip) + self.semantic_proj(semantic)
        return skip * self.gate(infused)


class GenericRecentUNet(nn.Module):
    def __init__(self, cfg: Dict, block_kind: str = "hybrid", use_sdi: bool = False) -> None:
        super().__init__()
        base = int(cfg["model"].get("recent_base_channels", cfg["model"].get("sota_base_channels", 32)))
        ch = [base, base * 2, base * 4, base * 8, base * 16]
        block = self._block_factory(block_kind)
        self.pool = nn.MaxPool2d(2)
        self.enc1 = block(_in_channels(cfg), ch[0])
        self.enc2 = block(ch[0], ch[1])
        self.enc3 = block(ch[1], ch[2])
        self.enc4 = block(ch[2], ch[3])
        self.bottleneck = block(ch[3], ch[4])
        self.sdi4 = SDIGate(ch[3], ch[4]) if use_sdi else None
        self.sdi3 = SDIGate(ch[2], ch[4]) if use_sdi else None
        self.sdi2 = SDIGate(ch[1], ch[4]) if use_sdi else None
        self.sdi1 = SDIGate(ch[0], ch[4]) if use_sdi else None
        self.dec4 = block(ch[4] + ch[3], ch[3])
        self.dec3 = block(ch[3] + ch[2], ch[2])
        self.dec2 = block(ch[2] + ch[1], ch[1])
        self.dec1 = block(ch[1] + ch[0], ch[0])
        self.head = nn.Conv2d(ch[0], _out_channels(cfg), kernel_size=1)

    @staticmethod
    def _block_factory(block_kind: str):
        if block_kind == "vss":
            return lambda in_ch, out_ch: VSSBlock(in_ch, out_ch, mamba_layers=2, kernel_size=7)
        if block_kind == "msvss":
            return lambda in_ch, out_ch: MultiScaleVSSBlock(in_ch, out_ch)
        if block_kind == "segmamba":
            return lambda in_ch, out_ch: HybridMambaBlock(in_ch, out_ch, mamba_layers=2, kernel_size=9)
        if block_kind == "kan_mamba":
            return lambda in_ch, out_ch: KANMambaBlock(in_ch, out_ch)
        if block_kind == "cfm":
            return lambda in_ch, out_ch: CoupledFusionBlock(in_ch, out_ch)
        return lambda in_ch, out_ch: HybridMambaBlock(in_ch, out_ch, mamba_layers=1, kernel_size=7)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        s4 = self.sdi4(e4, b) if self.sdi4 is not None else e4
        s3 = self.sdi3(e3, b) if self.sdi3 is not None else e3
        s2 = self.sdi2(e2, b) if self.sdi2 is not None else e2
        s1 = self.sdi1(e1, b) if self.sdi1 is not None else e1
        d4 = self.dec4(torch.cat([self._up(b, s4), s4], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, s3), s3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, s2), s2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, s1), s1], dim=1))
        return {"seg_logits": self.head(d1)}


class HieraBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, window_size: int = 7) -> None:
        super().__init__()
        self.local = ConvBNAct(channels, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.local(x)
        b, c, h, w = residual.shape
        pooled = F.adaptive_avg_pool2d(residual, (min(self.window_size, h), min(self.window_size, w)))
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attended = attended.transpose(1, 2).reshape(b, c, pooled.shape[-2], pooled.shape[-1])
        attended = F.interpolate(attended, size=(h, w), mode="bilinear", align_corners=False)
        return residual + attended


class CrossBranchFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.cnn_to_ssm = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), nn.Sigmoid())
        self.ssm_to_cnn = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), nn.Sigmoid())
        self.out = ConvBNAct(channels * 2, channels, kernel_size=1)

    def forward(self, cnn_feat: torch.Tensor, ssm_feat: torch.Tensor) -> torch.Tensor:
        cnn_refined = cnn_feat * self.ssm_to_cnn(ssm_feat)
        ssm_refined = ssm_feat * self.cnn_to_ssm(cnn_feat)
        return self.out(torch.cat([cnn_refined, ssm_refined], dim=1))


class DCMNetSegmenter(nn.Module):
    """2025 DCM-Net-inspired dual CNN/Mamba encoder with cross-branch fusion."""

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        base = int(cfg["model"].get("recent_base_channels", cfg["model"].get("sota_base_channels", 32)))
        ch = [base, base * 2, base * 4, base * 8]
        self.pool = nn.MaxPool2d(2)
        self.cnn_stages = nn.ModuleList(
            [
                DoubleConv(_in_channels(cfg), ch[0]),
                DoubleConv(ch[0], ch[1]),
                DoubleConv(ch[1], ch[2]),
                DoubleConv(ch[2], ch[3]),
            ]
        )
        self.ssm_stages = nn.ModuleList(
            [
                HybridMambaBlock(_in_channels(cfg), ch[0], mamba_layers=1, kernel_size=7),
                HybridMambaBlock(ch[0], ch[1], mamba_layers=1, kernel_size=7),
                HybridMambaBlock(ch[1], ch[2], mamba_layers=2, kernel_size=9),
                HybridMambaBlock(ch[2], ch[3], mamba_layers=2, kernel_size=9),
            ]
        )
        self.fusions = nn.ModuleList([CrossBranchFusion(c) for c in ch])
        self.bottleneck = HybridMambaBlock(ch[-1], ch[-1] * 2, mamba_layers=2, kernel_size=11)
        self.dec4 = DoubleConv(ch[-1] * 2 + ch[-1], ch[-1])
        self.dec3 = DoubleConv(ch[-1] + ch[2], ch[2])
        self.dec2 = DoubleConv(ch[2] + ch[1], ch[1])
        self.dec1 = DoubleConv(ch[1] + ch[0], ch[0])
        self.head = nn.Conv2d(ch[0], _out_channels(cfg), kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cnn_x = x
        ssm_x = x
        fused_features = []
        for idx, (cnn_stage, ssm_stage, fusion) in enumerate(zip(self.cnn_stages, self.ssm_stages, self.fusions)):
            if idx > 0:
                cnn_x = self.pool(cnn_x)
                ssm_x = self.pool(ssm_x)
            cnn_x = cnn_stage(cnn_x)
            ssm_x = ssm_stage(ssm_x)
            fused_features.append(fusion(cnn_x, ssm_x))

        f1, f2, f3, f4 = fused_features
        b = self.bottleneck(self.pool(f4))
        d4 = self.dec4(torch.cat([self._up(b, f4), f4], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, f3), f3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, f2), f2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, f1), f1], dim=1))
        return {"seg_logits": self.head(d1)}


class MFIIBlock(nn.Module):
    """Multi-Functional Information Interaction block for I2U-Net.

    Performs cross-path and cross-layer interaction between dual U-Net paths.
    """

    def __init__(self, channels: int, prev_channels: int = 0) -> None:
        super().__init__()
        self.cross_path_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.has_cross_layer = prev_channels > 0
        if self.has_cross_layer:
            self.cross_layer_proj = nn.Conv2d(prev_channels, channels, kernel_size=1, bias=False)
            self.cross_layer_fuse = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            )
            self.refine = ConvBNAct(channels, channels)

    def forward(self, path_a: torch.Tensor, path_b: torch.Tensor, prev_layer: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.cross_path_gate(torch.cat([path_a, path_b], dim=1))
        a_refined = path_a + gate * path_b
        b_refined = path_b + (1 - gate) * path_a
        if self.has_cross_layer and prev_layer is not None:
            prev_up = F.interpolate(prev_layer, size=a_refined.shape[-2:], mode='bilinear', align_corners=False)
            prev_up = self.cross_layer_proj(prev_up)
            a_refined = self.cross_layer_fuse(torch.cat([a_refined, prev_up], dim=1))
            b_refined = self.refine(b_refined + prev_up)
        return a_refined, b_refined


class HIFABlock(nn.Module):
    """Holistic Information Fusion and Augmentation bridge for I2U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.channel_attn = ChannelAttention(out_channels)

    def forward(self, path_a: torch.Tensor, path_b: torch.Tensor) -> torch.Tensor:
        return self.channel_attn(self.fuse(torch.cat([path_a, path_b], dim=1)))


class I2UNetSegmenter(nn.Module):
    """I2U-Net: dual-path U-Net with rich information interaction (MedIA 2024)."""

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        base = int(cfg["model"].get("sota_base_channels", 32))
        ch = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2)

        self.enc_a = nn.ModuleList([DoubleConv(_in_channels(cfg) if i == 0 else ch[i - 1], ch[i]) for i in range(5)])
        self.enc_b = nn.ModuleList([DoubleConv(_in_channels(cfg) if i == 0 else ch[i - 1], ch[i]) for i in range(5)])
        self.mfii_enc = nn.ModuleList([MFIIBlock(ch[i], prev_channels=(ch[i - 1] if i > 0 else 0)) for i in range(5)])

        self.hifa = HIFABlock(ch[4], ch[4])

        self.dec_a = nn.ModuleList([DoubleConv(ch[i + 1] + ch[i], ch[i]) for i in range(4)])
        self.dec_b = nn.ModuleList([DoubleConv(ch[i + 1] + ch[i], ch[i]) for i in range(4)])
        self.mfii_dec = nn.ModuleList([MFIIBlock(ch[i]) for i in range(4)])

        self.head = nn.Conv2d(ch[0] * 2, _out_channels(cfg), kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc_a_feats, enc_b_feats = [], []
        xa, xb = x, x
        prev = None
        for i in range(5):
            if i > 0:
                xa = self.pool(xa)
                xb = self.pool(xb)
            xa = self.enc_a[i](xa)
            xb = self.enc_b[i](xb)
            xa, xb = self.mfii_enc[i](xa, xb, prev)
            enc_a_feats.append(xa)
            enc_b_feats.append(xb)
            prev = xa

        bottleneck = self.hifa(enc_a_feats[-1], enc_b_feats[-1])
        da, db = bottleneck, bottleneck

        for i in range(3, -1, -1):
            da = self.dec_a[i](torch.cat([self._up(da, enc_a_feats[i]), enc_a_feats[i]], dim=1))
            db = self.dec_b[i](torch.cat([self._up(db, enc_b_feats[i]), enc_b_feats[i]], dim=1))
            da, db = self.mfii_dec[i](da, db, None)

        out = torch.cat([da, db], dim=1)
        return {"seg_logits": self.head(out)}


class BiConvLSTMCell(nn.Module):
    """Bidirectional Convolutional LSTM cell for TBConvL-Net."""

    def __init__(self, channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.gates_fwd = nn.Conv2d(channels + hidden_channels, hidden_channels * 4, kernel_size, padding=pad)
        self.gates_bwd = nn.Conv2d(channels + hidden_channels, hidden_channels * 4, kernel_size, padding=pad)
        self.out_proj = nn.Conv2d(hidden_channels * 2, channels, kernel_size=1)
        self.hidden_channels = hidden_channels

    def _step(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor, gates_conv: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)
        gates = gates_conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        device = x.device
        h_fwd = torch.zeros(b, self.hidden_channels, h, w, device=device)
        c_fwd = torch.zeros_like(h_fwd)
        h_bwd = torch.zeros_like(h_fwd)
        c_bwd = torch.zeros_like(h_fwd)
        h_fwd, c_fwd = self._step(x, h_fwd, c_fwd, self.gates_fwd)
        h_bwd, c_bwd = self._step(x, h_bwd, c_bwd, self.gates_bwd)
        return self.out_proj(torch.cat([h_fwd, h_bwd], dim=1))


class MiniViTBlock(nn.Module):
    """Lightweight ViT block for TBConvL-Net's transformer component."""

    def __init__(self, channels: int, num_heads: int = 4, window_size: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels))
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = min(self.window_size, h, w)
        pooled = F.adaptive_avg_pool2d(x, (ws, ws))
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = tokens + self.attn(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens), need_weights=False)[0]
        tokens = tokens + self.ffn(self.norm2(tokens))
        out = tokens.transpose(1, 2).reshape(b, c, ws, ws)
        return x + F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)


class TBConvLNetSegmenter(nn.Module):
    """TBConvL-Net: hybrid CNN + BiConvLSTM + ViT (arXiv 2409.03367)."""

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        base = int(cfg["model"].get("sota_base_channels", 32))
        ch = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(_in_channels(cfg), ch[0])
        self.enc2 = DoubleConv(ch[0], ch[1])
        self.enc3 = DoubleConv(ch[1], ch[2])
        self.enc4 = DoubleConv(ch[2], ch[3])

        self.bottleneck_cnn = DoubleConv(ch[3], ch[4])
        self.bottleneck_lstm = BiConvLSTMCell(ch[4], ch[4] // 2)
        self.bottleneck_vit = MiniViTBlock(ch[4], num_heads=8)

        self.dec4 = DoubleConv(ch[4] + ch[3], ch[3])
        self.dec3 = DoubleConv(ch[3] + ch[2], ch[2])
        self.dec2 = DoubleConv(ch[2] + ch[1], ch[1])
        self.dec1 = DoubleConv(ch[1] + ch[0], ch[0])

        self.lstm_skips = nn.ModuleList([BiConvLSTMCell(ch[i], ch[i] // 2) for i in range(4)])
        self.head = nn.Conv2d(ch[0], _out_channels(cfg), kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck_cnn(self.pool(e4))
        b = b + self.bottleneck_lstm(b)
        b = self.bottleneck_vit(b)

        skips = [e1, e2, e3, e4]
        skips = [skip + self.lstm_skips[i](skip) for i, skip in enumerate(skips)]

        d4 = self.dec4(torch.cat([self._up(b, skips[3]), skips[3]], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, skips[2]), skips[2]], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, skips[1]), skips[1]], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, skips[0]), skips[0]], dim=1))
        return {"seg_logits": self.head(d1)}

def build_sota_model(model_name: str, cfg: Dict) -> nn.Module:
    name = model_name.lower()
    if name == "crab":
        return ERGASegmenter(cfg)
    if name == "swin_unet":
        return SwinUNetLiteSegmenter(cfg)
    if name == "u_mamba":
        return GenericRecentUNet(cfg, block_kind="hybrid", use_sdi=False)
    if name == "vm_unet_v2":
        return GenericRecentUNet(cfg, block_kind="vss", use_sdi=True)
    if name == "msvm_unet":
        return GenericRecentUNet(cfg, block_kind="msvss", use_sdi=False)
    if name == "segmamba":
        return GenericRecentUNet(cfg, block_kind="segmamba", use_sdi=False)
    if name == "kmunet":
        return GenericRecentUNet(cfg, block_kind="kan_mamba", use_sdi=False)
    if name == "dcm_net":
        return DCMNetSegmenter(cfg)
    if name == "cfm_unet":
        return GenericRecentUNet(cfg, block_kind="cfm", use_sdi=False)
    if name == "i2u_net":
        return I2UNetSegmenter(cfg)
    if name == "tbconvl_net":
        return TBConvLNetSegmenter(cfg)
    raise ValueError(f"Unsupported SOTA model: {model_name}. Valid: {', '.join(MODEL_CHOICES)}")
