from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .disjoint_boundary import DisjointBoundaryGenerator
from .encoder import DoubleConv, _make_norm
from .erga import ERGAModule, GatedCrossLevelFusion, HierarchicalCrossLevelFusion, HighLevelTransformer, LightweightAttention, RegionBranch, SoftEdgeBranch
from .feature_enhancement import HighResImageFusion, LowResSemanticFusion
from .anatomical_geometry_prior import CurvatureAwareBoundaryRefinement, InterClassDistanceConsistency
from .improved_decoder import ImprovedDecoder
from .inner_boundary import InnerBoundaryGenerator
from .pretrained_encoder import build_encoder
from .pyramid_transformer_encoder import PyramidTransformerEncoder
from .relation_aware_overlap import RelationAwareOverlapPrior
from .relation_aware_region_to_edge import RelationAwareEdgeToRegionConstraint, RelationAwareRegionToEdgeConstraint
from .shared_boundary import SharedBoundaryGenerator


class PlainUNet(nn.Module):
    """Original-style U-Net baseline: ReLU convs, no BN, transposed-conv upsampling."""

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        c5 = c1 * 16
        self.enc1 = self._block(in_channels, c1)
        self.enc2 = self._block(c1, c2)
        self.enc3 = self._block(c2, c3)
        self.enc4 = self._block(c3, c4)
        self.bottleneck = self._block(c4, c5)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = self._block(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = self._block(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = self._block(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = self._block(c1 + c1, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _match_skip(skip: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if skip.shape[-2:] == x.shape[-2:]:
            return skip
        return F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, self._match_skip(e4, d4)], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self._match_skip(e3, d3)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self._match_skip(e2, d2)], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self._match_skip(e1, d1)], dim=1))
        return self.head(d1)


class ERGASegmenter(nn.Module):
    VALID_RELATIONS = {'disjoint', 'intersection', 'containment'}
    _TASK_MODE_MAP = {'binary': 'exclusive', 'multiclass': 'exclusive', 'multilabel': 'independent'}

    @staticmethod
    def _infer_default_relations(task_mode: str, num_classes: int) -> list[str]:
        if task_mode == 'exclusive' and num_classes <= 1:
            return ['disjoint']
        elif task_mode == 'exclusive':
            return ['disjoint', 'containment']
        else:
            return ['disjoint', 'intersection', 'containment']

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        data_cfg = cfg['data']
        model_cfg = cfg['model']

        # Task mode: backward-compatible mapping to exclusive/independent.
        raw_task_mode = model_cfg['task_mode']
        self.task_mode = self._TASK_MODE_MAP.get(raw_task_mode, raw_task_mode)
        if self.task_mode not in {'exclusive', 'independent'}:
            raise ValueError(f'Unsupported task_mode: {raw_task_mode}. Valid: binary, multiclass, exclusive, multilabel, independent')

        # Seg classes.
        if raw_task_mode == 'binary':
            self.seg_classes = 1
        else:
            self.seg_classes = int(model_cfg['num_classes'])

        # Region mode: auto-infer from task_mode if not explicitly set, or use config value.
        default_region_mode = 'single_label' if self.task_mode == 'exclusive' else 'multi_label'
        self.region_mode = model_cfg.get('region_mode', default_region_mode)
        if self.region_mode not in {'single_label', 'multi_label'}:
            raise ValueError(f'Unsupported region_mode: {self.region_mode}')

        self.region_num_classes = int(model_cfg['region_num_classes'])

        # Relation types: auto-infer or manual override from config.
        default_relations = self._infer_default_relations(self.task_mode, self.seg_classes)
        configured_relations = model_cfg.get('relation_types', None)
        if configured_relations is not None:
            invalid = set(configured_relations) - self.VALID_RELATIONS
            if invalid:
                raise ValueError(f'Invalid relation_types: {invalid}. Valid: {self.VALID_RELATIONS}')
            self.active_relations = set(configured_relations)
        else:
            self.active_relations = set(default_relations)

        common_dim = int(model_cfg['common_dim'])
        decoder_dim = int(model_cfg['decoder_dim'])
        self.inject_edge_skip = bool(model_cfg['inject_edge_skip'])
        ablation_cfg = model_cfg.get('ablation', {})
        self.disable_edge_branch = bool(ablation_cfg.get('disable_edge_branch', False))
        self.disable_region_branch = bool(ablation_cfg.get('disable_region_branch', False))
        self.disable_overlap_prior = bool(ablation_cfg.get('disable_overlap_prior', False))
        self.disable_boundary_generators = bool(ablation_cfg.get('disable_boundary_generators', False))
        self.disable_region_to_edge_constraint = bool(ablation_cfg.get('disable_region_to_edge_constraint', False))
        self.disable_icdc = bool(ablation_cfg.get('disable_icdc', False))
        self.disable_coarse_sdm = bool(ablation_cfg.get('disable_coarse_sdm', False))
        self.disable_erga = bool(ablation_cfg.get('disable_erga', False))
        self.disable_lightweight_attention = bool(ablation_cfg.get('disable_lightweight_attention', False))
        self.disable_transformer = bool(ablation_cfg.get('disable_transformer', False))
        self.disable_prior_injection = bool(ablation_cfg.get('disable_prior_injection', False))
        self.disable_feature_enhancement = bool(ablation_cfg.get('disable_feature_enhancement', False))
        self.disable_uncertainty_head = bool(ablation_cfg.get('disable_uncertainty_head', False))
        self.use_edge_to_region_constraint = bool(model_cfg.get('use_edge_to_region_constraint', False))
        self.use_plain_unet_baseline = bool(ablation_cfg.get('use_plain_unet_baseline', False))
        self.cross_level_fusion_mode = ablation_cfg.get('cross_level_fusion_mode', 'gated')
        if bool(ablation_cfg.get('disable_cross_level_fusion', False)):
            self.cross_level_fusion_mode = 'f3_only'
        if self.cross_level_fusion_mode not in {'gated', 'sum', 'mean', 'f3_only'}:
            raise ValueError(f'Unsupported cross_level_fusion_mode: {self.cross_level_fusion_mode}')

        if self.use_plain_unet_baseline:
            self.unet = PlainUNet(
                in_channels=int(data_cfg['in_channels']),
                out_channels=self.seg_classes,
                base_channels=int(model_cfg.get('classic_unet_base_channels', 64)),
            )
            return

        self.encoder, encoder_channels = build_encoder(cfg)
        self.skip_projs = nn.ModuleList([nn.Conv2d(ch, common_dim, kernel_size=1) for ch in encoder_channels])

        self.high_res_fusion = None if self.disable_feature_enhancement else HighResImageFusion(int(data_cfg.get('in_channels', 1)), common_dim)
        self.low_res_fusion = None if self.disable_feature_enhancement else LowResSemanticFusion(common_dim)

        self.fusion = HierarchicalCrossLevelFusion(common_dim) if self.cross_level_fusion_mode == 'gated' else None
        self.edge_branch = None if self.disable_edge_branch else SoftEdgeBranch(common_dim, self.seg_classes, self.task_mode, model_cfg['edge_tau'])
        self.region_branch = None if self.disable_region_branch else RegionBranch(common_dim, self.region_num_classes, self.region_mode)

        self.overlap_prior = None if self.disable_overlap_prior else RelationAwareOverlapPrior(self.region_num_classes, self.region_mode, self.task_mode)
        boundary_context_ch = common_dim + self.seg_classes
        self.shared_boundary_gen = None if (self.disable_overlap_prior or self.disable_boundary_generators) else SharedBoundaryGenerator(context_channels=boundary_context_ch)
        self.inner_boundary_gen = None if (self.disable_overlap_prior or self.disable_boundary_generators) else InnerBoundaryGenerator(context_channels=boundary_context_ch)
        self.disjoint_boundary_gen = None if (self.disable_overlap_prior or self.disable_boundary_generators) else DisjointBoundaryGenerator(context_channels=boundary_context_ch)
        self.region_to_edge_constraint = None if self.disable_region_to_edge_constraint else RelationAwareRegionToEdgeConstraint(
            edge_classes=self.seg_classes,
            region_classes=self.region_num_classes,
            edge_channels=common_dim,
            alpha=model_cfg['alpha'],
            beta=model_cfg['beta'],
            gamma=model_cfg.get('gamma', 0.3),
            active_relations=self.active_relations,
            containment_scale=float(model_cfg.get('containment_scale', 0.3)),
        )
        self.edge_to_region_constraint = None if (self.disable_region_to_edge_constraint or not self.use_edge_to_region_constraint) else RelationAwareEdgeToRegionConstraint(
            edge_classes=self.seg_classes,
            region_classes=self.region_num_classes,
            region_channels=common_dim,
            alpha=model_cfg['alpha'],
            beta=model_cfg['beta'],
            gamma=model_cfg.get('gamma', 0.3),
            active_relations=self.active_relations,
        )

        self.erga = None if self.disable_erga else ERGAModule(
            common_dim,
            int(model_cfg['erga_hidden_dim']),
            use_prior_attention=bool(model_cfg.get('use_prior_attention', False)),
            relation_types=tuple(sorted(self.active_relations)),
        )
        self.lightweight_attn = None if self.disable_lightweight_attention else LightweightAttention(common_dim, int(model_cfg['channel_rank']))
        self.transformer = None if self.disable_transformer else HighLevelTransformer(
            common_dim,
            int(model_cfg['num_heads']),
            int(model_cfg['window_size']),
            int(model_cfg['transformer_depth']),
        )

        self.edge_skip_proj = nn.Conv2d(common_dim, common_dim, kernel_size=1, bias=False)
        self.decoder = ImprovedDecoder(
            common_dim, decoder_dim,
            use_prior_injection=not self.disable_prior_injection,
            num_classes=self.seg_classes,
            use_scale_routing=bool(model_cfg.get('use_scale_routing', True)),
        )
        self.seg_head = nn.Sequential(DoubleConv(decoder_dim, decoder_dim), nn.Conv2d(decoder_dim, self.seg_classes, kernel_size=1))
        self.aux_seg_head = nn.Conv2d(decoder_dim, self.seg_classes, kernel_size=1)
        self.sdm_head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim // 2, kernel_size=3, padding=1),
            _make_norm(decoder_dim // 2),
            nn.GELU(),
            nn.Conv2d(decoder_dim // 2, self.seg_classes, kernel_size=1),
            nn.Tanh(),
        )
        self.curvature_refine = CurvatureAwareBoundaryRefinement(self.seg_classes, decoder_dim)
        self.inter_class_dist = (
            InterClassDistanceConsistency(self.seg_classes)
            if self.seg_classes > 1 and not self.disable_icdc
            else None
        )
        self.coarse_sdm_head = nn.Sequential(
            nn.Conv2d(common_dim, common_dim // 2, kernel_size=3, padding=1, bias=False),
            _make_norm(common_dim // 2),
            nn.GELU(),
            nn.Conv2d(common_dim // 2, self.seg_classes, kernel_size=1),
            nn.Tanh(),
        ) if self.seg_classes > 1 and not self.disable_coarse_sdm else None

        # Boundary uncertainty head: predicts per-pixel confidence [0,1]
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim // 4, kernel_size=3, padding=1, bias=False),
            _make_norm(decoder_dim // 4),
            nn.GELU(),
            nn.Conv2d(decoder_dim // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        ) if not self.disable_uncertainty_head else None

        # Freeze modules for inactive relation types.
        self._freeze_inactive_relations()

        # Warmup soft detach ratio for overlap prior input (curriculum coupling).
        # 0.0 = full detach (safe early training), 1.0 = full gradient flow.
        self.register_buffer('_prior_grad_ratio', torch.tensor(0.05))

    def _freeze_inactive_relations(self) -> None:
        """Freeze parameters of modules corresponding to inactive relation types."""
        relation_to_boundary = {'intersection': self.shared_boundary_gen, 'containment': self.inner_boundary_gen, 'disjoint': self.disjoint_boundary_gen}
        for rel in ('intersection', 'containment', 'disjoint'):
            if rel not in self.active_relations:
                bg = relation_to_boundary.get(rel)
                if bg is not None:
                    bg.requires_grad_(False)
                if self.erga is not None and rel in self.erga.boundary_adapters:
                    self.erga.boundary_adapters[rel].requires_grad_(False)
                if self.region_to_edge_constraint is not None and rel in self.region_to_edge_constraint.gates:
                    self.region_to_edge_constraint.gates[rel].requires_grad_(False)

    def set_prior_grad_ratio(self, ratio: float) -> None:
        """Update gradient blending ratio for overlap prior input (curriculum coupling).

        Schedule: epoch 0-20 → 0.05, epoch 20-50 → 0.2, epoch 50+ → 1.0
        """
        self._prior_grad_ratio.fill_(max(0.0, min(1.0, ratio)))

    def _blend_for_prior(self, region_logits: torch.Tensor) -> torch.Tensor:
        """Apply soft detach: ratio * x + (1-ratio) * x.detach()"""
        r = self._prior_grad_ratio
        if r >= 1.0:
            return region_logits
        return r * region_logits + (1.0 - r) * region_logits.detach()

    def _fuse_cross_level_features(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.cross_level_fusion_mode == 'gated':
            return self.fusion(feats)

        target_size = feats[2].shape[-2:]
        aligned = [
            feat if feat.shape[-2:] == target_size else F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            for feat in feats
        ]
        if self.cross_level_fusion_mode == 'sum':
            return torch.stack(aligned, dim=0).sum(dim=0), None
        if self.cross_level_fusion_mode == 'mean':
            return torch.stack(aligned, dim=0).mean(dim=0), None
        if self.cross_level_fusion_mode == 'f3_only':
            return feats[2], None
        raise RuntimeError(f'Unsupported cross_level_fusion_mode: {self.cross_level_fusion_mode}')

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | list[torch.Tensor]]:
        input_size = x.shape[-2:]
        if self.use_plain_unet_baseline:
            seg_logits = self.unet(x)
            edge_shape = (x.shape[0], self.seg_classes, max(input_size[0] // 16, 1), max(input_size[1] // 16, 1))
            region_shape = (x.shape[0], self.region_num_classes, max(input_size[0] // 16, 1), max(input_size[1] // 16, 1))
            prior_shape = (x.shape[0], 1, max(input_size[0] // 16, 1), max(input_size[1] // 16, 1))
            edge_zero = x.new_zeros(edge_shape)
            region_zero = x.new_zeros(region_shape)
            prior_zero = x.new_zeros(prior_shape)
            return {
                'seg_logits': seg_logits,
                'edge_logits': [edge_zero, edge_zero, edge_zero],
                'region_logits': region_zero,
                'region_attention': region_zero,
                'edge_attention': edge_zero,
                'edge_attention_disjoint': edge_zero,
                'edge_attention_intersection': edge_zero,
                'edge_attention_inner': edge_zero,
                'o_disjoint': prior_zero,
                'o_intersection': prior_zero,
                'o_containment': prior_zero,
                'shared_boundary_prior': prior_zero,
                'inner_boundary_prior': prior_zero,
                'disjoint_boundary_prior': prior_zero,
            }

        # Encoder outputs:
        # F1 [B, C1, H/4,  W/4]
        # F2 [B, C2, H/8,  W/8]
        # F3 [B, C3, H/16, W/16]
        # F4 [B, C4, H/32, W/32]
        encoder_feats = self.encoder(x)
        f1, f2, f3, f4 = [proj(feat) for proj, feat in zip(self.skip_projs, encoder_feats)]

        # Feature enhancement: inject image details into F1/F2, cross-attend F3/F4.
        if self.high_res_fusion is not None and self.low_res_fusion is not None:
            f1, f2 = self.high_res_fusion(x, f1, f2)
            f3, f4 = self.low_res_fusion(f3, f4)

        # Shared fused context: F_fuse [B, common_dim, H/16, W/16]
        f_fuse, f_high = self._fuse_cross_level_features([f1, f2, f3, f4])

        # Coarse SDM from f_fuse for early geometry-relation coupling
        coarse_sdm = self.coarse_sdm_head(f_fuse) if self.coarse_sdm_head is not None else None

        # Dual branches.
        if self.edge_branch is not None:
            f_edge, a_edge, edge_logits = self.edge_branch([f1, f2, f_fuse])
        else:
            f_edge = None
            a_edge = None
            edge_logits = [torch.zeros(x.shape[0], self.seg_classes, *f_fuse.shape[-2:], device=x.device, dtype=x.dtype) for _ in range(3)]

        if self.region_branch is not None:
            f_region, a_region, region_logits = self.region_branch([f3, f4, f_fuse])
        else:
            f_region = None
            a_region = None
            region_logits = torch.zeros(x.shape[0], self.region_num_classes, *f_fuse.shape[-2:], device=x.device, dtype=x.dtype)

        # Relation-aware priors.
        if self.overlap_prior is not None and a_region is not None:
            prior_input = self._blend_for_prior(region_logits)
            o_disjoint, o_intersection, o_containment = self.overlap_prior(prior_input, a_region)
        else:
            prior_shape = (x.shape[0], 1, *f_fuse.shape[-2:])
            o_disjoint = torch.zeros(prior_shape, device=x.device, dtype=x.dtype)
            o_intersection = torch.zeros_like(o_disjoint)
            o_containment = torch.zeros_like(o_disjoint)

        # Geometry-relation closed loop: ICDC enhances priors BEFORE boundary generation
        if self.inter_class_dist is not None and coarse_sdm is not None:
            coarse_geo = self.inter_class_dist(coarse_sdm)
            geo_alpha = self.inter_class_dist.alpha
            o_disjoint = o_disjoint + geo_alpha * coarse_geo[:, 0:1]
            o_intersection = o_intersection + geo_alpha * coarse_geo[:, 1:2]
            o_containment = o_containment + geo_alpha * coarse_geo[:, 2:3]

        # Relation-aware region-to-edge constraint (before boundary generation).
        if self.region_to_edge_constraint is not None and f_edge is not None and a_edge is not None:
            f_edge_refined, a_edge_refined, a_edge_disjoint, a_edge_inter, a_edge_inner = self.region_to_edge_constraint(
                f_edge,
                a_edge,
                a_region,
                o_disjoint,
                o_intersection,
                o_containment,
            )
        else:
            f_edge_refined = f_edge
            a_edge_refined = a_edge
            edge_attn_shape = (x.shape[0], self.seg_classes, *f_fuse.shape[-2:])
            a_edge_disjoint = torch.zeros(edge_attn_shape, device=x.device, dtype=x.dtype) if a_edge is None else torch.zeros_like(a_edge)
            a_edge_inter = torch.zeros(edge_attn_shape, device=x.device, dtype=x.dtype) if a_edge is None else torch.zeros_like(a_edge)
            a_edge_inner = torch.zeros(edge_attn_shape, device=x.device, dtype=x.dtype) if a_edge is None else torch.zeros_like(a_edge)

        # Boundary generation (uses edge-refined context).
        if self.overlap_prior is not None and a_region is not None:
            if a_edge_refined is not None:
                a_edge_ctx = F.interpolate(a_edge_refined, size=f_fuse.shape[-2:], mode='bilinear', align_corners=False)
                boundary_context = torch.cat([f_fuse, a_edge_ctx], dim=1)
            else:
                boundary_context = torch.cat([f_fuse, torch.zeros(x.shape[0], self.seg_classes, *f_fuse.shape[-2:], device=x.device, dtype=x.dtype)], dim=1)
            b_shared = self.shared_boundary_gen(o_intersection, boundary_context) if ('intersection' in self.active_relations and self.shared_boundary_gen is not None) else torch.zeros_like(o_intersection)
            b_inner = self.inner_boundary_gen(o_containment, boundary_context) if ('containment' in self.active_relations and self.inner_boundary_gen is not None) else torch.zeros_like(o_containment)
            b_disjoint = self.disjoint_boundary_gen(o_disjoint, boundary_context) if ('disjoint' in self.active_relations and self.disjoint_boundary_gen is not None) else torch.zeros_like(o_disjoint)
        else:
            b_shared = torch.zeros_like(o_disjoint)
            b_inner = torch.zeros_like(o_disjoint)
            b_disjoint = torch.zeros_like(o_disjoint)

        if self.edge_to_region_constraint is not None and f_region is not None and a_region is not None and a_edge is not None:
            f_region_refined, a_region_refined = self.edge_to_region_constraint(
                f_region,
                a_region,
                a_edge,
                b_shared,
                b_inner,
                b_disjoint,
            )
        else:
            f_region_refined = f_region
            a_region_refined = a_region

        # ERGA bottleneck -> lightweight attention -> high-level transformer.
        erga_has_valid_input = f_edge_refined is not None and f_region_refined is not None
        if self.erga is not None and erga_has_valid_input:
            bottleneck, edge_detail = self.erga(f_edge_refined, f_region_refined, b_shared, b_inner, b_disjoint, f_fuse)
        else:
            bottleneck = f_fuse
            edge_detail = None

        # Collect gate activations for entropy/diversity regularization
        gate_activations: list[torch.Tensor] = []
        if self.fusion is not None and hasattr(self.fusion, '_last_gate_activations'):
            gate_activations.extend(self.fusion._last_gate_activations)
        if self.erga is not None and hasattr(self.erga, '_last_gate_activations'):
            gate_activations.extend(self.erga._last_gate_activations)

        if self.lightweight_attn is not None:
            bottleneck = self.lightweight_attn(bottleneck)
        if self.transformer is not None:
            bottleneck = self.transformer(bottleneck)

        skip1 = f1
        if self.inject_edge_skip and (edge_detail is not None or not self.disable_edge_branch):
            if edge_detail is not None:
                edge_skip = F.interpolate(edge_detail, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            else:
                edge_skip = F.interpolate(f_edge_refined, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            skip1 = skip1 + self.edge_skip_proj(edge_skip)

        decoder_feature, decoder_mid = self.decoder(
            bottleneck,
            skip1,
            f2,
            f3,
            f4,
            None if self.disable_prior_injection else f_edge_refined,
            None if self.disable_prior_injection else f_region_refined,
            None if self.disable_prior_injection else b_shared,
            None if self.disable_prior_injection else b_inner,
            None if self.disable_prior_injection else b_disjoint,
        )
        decoder_feature = F.interpolate(decoder_feature, size=input_size, mode='bilinear', align_corners=False)
        sdm_pred = self.sdm_head(decoder_feature)
        sdm_for_cabr = self._blend_for_prior(sdm_pred)
        decoder_feature = self.curvature_refine(sdm_for_cabr, decoder_feature)

        # Boundary uncertainty: high uncertainty = low confidence boundary
        uncertainty = self.uncertainty_head(decoder_feature) if self.uncertainty_head is not None else None
        seg_logits = self.seg_head(decoder_feature)

        aux_seg_logits = F.interpolate(self.aux_seg_head(decoder_mid), size=input_size, mode='bilinear', align_corners=False)

        return {
            'seg_logits': seg_logits,
            'sdm_pred': sdm_pred,
            'coarse_sdm_pred': coarse_sdm,
            'uncertainty': uncertainty,
            'aux_seg_logits': aux_seg_logits,
            'edge_logits': edge_logits,
            'region_logits': region_logits,
            'region_attention': a_region_refined,
            'region_attention_raw': a_region,
            'edge_attention': a_edge_refined,
            'edge_attention_disjoint': a_edge_disjoint,
            'edge_attention_intersection': a_edge_inter,
            'edge_attention_inner': a_edge_inner,
            'o_disjoint': o_disjoint,
            'o_intersection': o_intersection,
            'o_containment': o_containment,
            'shared_boundary_prior': b_shared,
            'inner_boundary_prior': b_inner,
            'disjoint_boundary_prior': b_disjoint,
            'router_entropy': self.decoder.scale_router.entropy_loss(bottleneck) if self.decoder.scale_router is not None else torch.zeros(1, device=x.device),
            'prior_attention_weights': (
                self.erga.prior_attention.last_weights
                if self.erga is not None and self.erga.prior_attention is not None and self.erga.prior_attention.last_weights is not None
                else torch.empty(0, device=x.device, dtype=x.dtype)
            ),
            'gate_activations': gate_activations,
        }
