# DP-AdaSNN core; AGPL-3.0. See THIRD_PARTY_NOTICES.md.
import copy
import math
from .activation import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class HACPhysicsLoss(nn.Module):

    def __init__(self, shadow_weight=1.0, focus_weight=1.0, shadow_margin=0.2, ts_high=0.7, ts_low=0.3):
        super().__init__()
        self.shadow_weight = shadow_weight
        self.focus_weight = focus_weight
        self.shadow_margin = shadow_margin
        self.ts_high = ts_high
        self.ts_low = ts_low

    @staticmethod
    def compute_excitation(v_mem_seq, pos_thresh, neg_thresh):
        """Return the existing bipolar HAC/TSD excitation terms without changing their definition."""
        safe_pos_th = pos_thresh + 1e-05
        safe_neg_th = neg_thresh + 1e-05
        E_pos = F.relu(v_mem_seq) / safe_pos_th
        E_neg = F.relu(-v_mem_seq) / safe_neg_th
        E_combined = torch.max(E_pos, E_neg)
        return (E_pos, E_neg, E_combined)

    def compute_terms(self, v_mem_seq, time_surface_seq, pos_thresh, neg_thresh):
        E_pos, E_neg, E_combined = self.compute_excitation(v_mem_seq, pos_thresh, neg_thresh)
        focus_mask = F.relu(time_surface_seq - self.ts_high) / (1.0 - self.ts_high + 1e-06)
        shadow_mask = F.relu(self.ts_low - time_surface_seq) / (self.ts_low + 1e-06)
        shadow_mask = torch.pow(shadow_mask, 2)
        area_shadow = shadow_mask.sum() + 0.0001
        area_focus = focus_mask.sum() + 0.0001
        C = E_combined.shape[2]
        excessive_excitation = F.relu(E_combined - self.shadow_margin)
        loss_shadow = (excessive_excitation * shadow_mask).sum() / (area_shadow * C)
        E_spatial_mean = torch.mean(E_combined, dim=2, keepdim=True)
        excitation_gap = F.relu(1.0 - E_spatial_mean)
        loss_focus = (excitation_gap * focus_mask).sum() / area_focus
        total_physics_loss = self.shadow_weight * loss_shadow + self.focus_weight * loss_focus
        return {'E_pos': E_pos, 'E_neg': E_neg, 'E_combined': E_combined, 'focus_mask': focus_mask, 'shadow_mask': shadow_mask, 'area_focus': area_focus, 'area_shadow': area_shadow, 'loss_shadow': loss_shadow, 'loss_focus': loss_focus, 'total_physics_loss': total_physics_loss}

    def forward(self, v_mem_seq, time_surface_seq, pos_thresh, neg_thresh):
        return self.compute_terms(v_mem_seq, time_surface_seq, pos_thresh, neg_thresh)['total_physics_loss']

class DPAdaSNNEmbedding(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, Ts, kwargs_spikes, split=False, spike_attach=False, write_zero=False, abs=False, depth=2, log_alpha=1.0, readout='avg', neg_thresh_init=2.0, use_neg_spike=True, use_depthwise_conv=True, use_physics_loss=True, use_log_compress=True, use_time_surface=True, use_adaptive_thresh=True, gain_k=3.0, v_min=0.5, v_max=3.0, not_rpd=False, physics_shadow_weight=1.0, physics_focus_weight=1.0, output_clamp=3.0, input_count_cap=None):
        super(DPAdaSNNEmbedding, self).__init__()
        self.output_clamp = float(output_clamp)
        if not math.isfinite(self.output_clamp) or self.output_clamp <= 0:
            raise ValueError(f'output_clamp must be a finite positive value, got {output_clamp!r}')
        self.input_count_cap = None if input_count_cap is None else float(input_count_cap)
        if self.input_count_cap is not None and (not math.isfinite(self.input_count_cap) or self.input_count_cap <= 0):
            raise ValueError(f'input_count_cap must be a finite positive value or None, got {input_count_cap!r}')
        self.kernel_size = kernel_size
        self.kwargs_spikes = kwargs_spikes
        self.Ts = Ts
        self.abs = abs
        self.split = split
        self.readout = readout
        self.write_zero = write_zero
        self.log_compress = use_log_compress
        self.alpha = log_alpha
        self.gain_K = float(gain_k)
        self.use_adaptive_thresh = use_adaptive_thresh
        self.use_neg_spike = use_neg_spike
        self.use_depthwise_conv = use_depthwise_conv
        self.use_time_surface = use_time_surface
        self.nb_steps = self.kwargs_spikes.get('nb_steps', 6)
        init_thresh = self.kwargs_spikes.get('thresh', 1.0)
        self.vreset = self.kwargs_spikes.get('vreset', 0.0)
        spike_fn = self.kwargs_spikes.get('spike_fn', 'Rectangle')
        if isinstance(spike_fn, str):
            spike_fn = globals().get(spike_fn, Rectangle)
        self.act_fun = self.warp_spike_fn(spike_fn)
        self.depth = int(depth)
        self.gate_conv = self.build_conv(out_channel, out_channel * 2, kernel_size, depth=self.depth, is_raw_input=False)
        self.input_conv = self.build_conv(in_channel, out_channel * 2, kernel_size, depth=self.depth, is_raw_input=True)
        if self.split:
            self.gate_conv_agg = nn.Conv2d(out_channel, out_channel * 2, kernel_size, padding=kernel_size // 2)
            self.input_conv_agg = nn.Conv2d(in_channel, out_channel * 2, kernel_size, padding=kernel_size // 2)
        self.spike_attach = spike_attach
        self._init_weight()
        self.not_rpd = bool(not_rpd)
        self.use_physics_loss = use_physics_loss
        self.physics_loss = HACPhysicsLoss(shadow_weight=physics_shadow_weight, focus_weight=physics_focus_weight)
        if self.use_physics_loss:
            pass
        else:
            self.physics_loss = None
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        if self.use_adaptive_thresh:
            init_val = math.log(0.25)
            self.thresh_param = nn.Parameter(torch.full((1, out_channel, 1, 1), init_val, dtype=torch.float32))
            self.thresh = None
            if self.use_neg_spike:
                self.register_buffer('neg_thresh_fixed', torch.full((1, out_channel, 1, 1), float(neg_thresh_init), dtype=torch.float32))
                self.neg_thresh_param = None
                self.neg_thresh = None
            else:
                self.neg_thresh_param = None
                self.neg_thresh = None
        else:
            self.thresh_param = None
            self.thresh = float(init_thresh)
            if self.use_neg_spike:
                self.register_buffer('neg_thresh_fixed', torch.full((1, out_channel, 1, 1), float(neg_thresh_init), dtype=torch.float32))
            else:
                self.neg_thresh_fixed = None

    def build_conv(self, in_channel, out_channel, kernel_size, depth=1, is_raw_input=True):
        convs = []
        if is_raw_input and in_channel == 2 and getattr(self, 'use_depthwise_conv', True):
            convs.append(nn.Conv2d(in_channel, in_channel, kernel_size, padding=kernel_size // 2, groups=in_channel, bias=False))
            convs.append(nn.ReLU(inplace=True))
            convs.append(nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False))
        else:
            convs.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False))
        for _ in range(depth - 1):
            convs.append(nn.ReLU(inplace=True))
            convs.append(nn.Conv2d(out_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False))
        return nn.Sequential(*convs)

    def warp_spike_fn(self, spike_fn):
        if isinstance(spike_fn, nn.Module):
            return copy.deepcopy(spike_fn)
        elif issubclass(spike_fn, torch.autograd.Function):
            return spike_fn.apply
        elif issubclass(spike_fn, torch.nn.Module):
            return spike_fn()

    def _init_weight(self):
        for m in self.input_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        for m in self.gate_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='sigmoid')
        if self.split:
            nn.init.kaiming_uniform_(self.input_conv_agg.weight, nonlinearity='sigmoid')
            nn.init.orthogonal_(self.gate_conv_agg.weight, gain=nn.init.calculate_gain('relu'))

    def update(self, vmem, gate, current, dynamic_thresh):
        vmem = gate * vmem + current
        spike_pos = self.act_fun(vmem - dynamic_thresh)
        if self.use_neg_spike:
            neg_th = self.neg_thresh_fixed
            spike_neg = self.act_fun(-vmem - neg_th)
            spike = spike_pos + spike_neg
            if self.vreset is None:
                vmem_update = vmem - dynamic_thresh * spike_pos + neg_th * spike_neg
            else:
                vmem_update = vmem * (1 - spike) + self.vreset * spike
        else:
            spike_neg = torch.zeros_like(spike_pos)
            spike = spike_pos
            if self.vreset is None:
                vmem_update = vmem - dynamic_thresh * spike_pos
            else:
                vmem_update = vmem * (1 - spike) + self.vreset * spike
        return (vmem_update, vmem, spike, spike_pos, spike_neg)

    def forward(self, events):
        param_dtype = next(self.parameters()).dtype
        events = events.to(param_dtype)
        time_surface = None
        if events.dim() < 5:
            B, C, H, W = events.shape
            return torch.zeros((self.Ts, B, 2, H, W), device=events.device, dtype=param_dtype)
        elif events.dim() > 5:
            events = events.flatten(end_dim=-5)
            events = events.transpose(0, 1)
        else:
            events = events.transpose(0, 1)
        T, B, C, H, W = events.shape
        if T != self.nb_steps:
            raise ValueError(f'DPAdaSNNEmbedding expected T == nb_steps ({self.nb_steps}), but received T={T}.')
        if C == 4:
            events_count = events[:, :, :2, :, :]
            if self.training and self.use_time_surface:
                raw_ts = events[:, :, 2:, :, :]
                ts_collapsed, _ = torch.max(raw_ts, dim=2, keepdim=True)
                ts_reshaped = ts_collapsed.view(-1, 1, H, W)
                ts_dilated = torch.nn.functional.max_pool2d(ts_reshaped, kernel_size=3, stride=1, padding=1)
                time_surface = ts_dilated.view(T, B, 1, H, W)
            events = events_count
        if self.input_count_cap is not None:
            events = torch.clamp(events, max=self.input_count_cap)
        if self.log_compress:
            abs_events = torch.abs(events)
            compressed = torch.sign(events) * (torch.log1p(self.alpha * abs_events) / self.alpha)
            events = compressed * self.gain_K
        spike_last = torch.zeros_like(events[0], dtype=param_dtype)
        vmem = torch.zeros_like(events[0], dtype=param_dtype)
        aggregation = torch.zeros([self.Ts] + list(events.shape[1:]), device=events.device, dtype=param_dtype)
        seg_ind = torch.zeros_like(events[0]).long()
        vmem_avg = torch.zeros_like(events[0])
        t_last = torch.zeros_like(events[0]).long() - 1
        needs_membrane_history = self.training and self.use_physics_loss
        if needs_membrane_history:
            membrane_history = torch.zeros_like(events, dtype=param_dtype, device=events.device)
        else:
            membrane_history = None
        if self.use_adaptive_thresh:
            dynamic_thresh = self.v_min + (self.v_max - self.v_min) * torch.sigmoid(self.thresh_param)
        else:
            dynamic_thresh = self.thresh
        neg_dynamic_thresh = self.neg_thresh_fixed if self.use_neg_spike else dynamic_thresh
        for t in range(self.nb_steps):
            gate_input = spike_last.to(dtype=param_dtype)
            state = self.gate_conv(gate_input)
            g_rec, c_rec = state.chunk(2, dim=-3)
            inpt = self.input_conv(events[t])
            g_in, c_in = inpt.chunk(2, dim=-3)
            gate = torch.sigmoid(g_in + g_rec)
            current = c_in + c_rec
            vmem, vmem_no_reset, spike_last, spike_pos, spike_neg = self.update(vmem, gate, current, dynamic_thresh)
            spike_last = spike_last.to(dtype=param_dtype)
            if membrane_history is not None:
                membrane_history[t] = vmem_no_reset
            vmem_avg += vmem_no_reset
            spike_indices = spike_last.nonzero()
            seg_pos = seg_ind[spike_indices[:, 0], spike_indices[:, 1], spike_indices[:, 2], spike_indices[:, 3]]
            valid_pos = seg_pos < self.Ts
            seg_pos, spike_indices = (seg_pos[valid_pos], spike_indices[valid_pos])
            if self.readout == 'sum':
                v = vmem_avg[spike_indices[:, 0], spike_indices[:, 1], spike_indices[:, 2], spike_indices[:, 3]]
            elif self.readout == 'last':
                v = vmem[spike_indices[:, 0], spike_indices[:, 1], spike_indices[:, 2], spike_indices[:, 3]]
            elif self.readout == 'avg':
                v = vmem_avg[spike_indices[:, 0], spike_indices[:, 1], spike_indices[:, 2], spike_indices[:, 3]] / (t - t_last[spike_indices[:, 0], spike_indices[:, 1], spike_indices[:, 2], spike_indices[:, 3]])
            if self.spike_attach:
                v *= spike_last[spike_indices[:, 0], spike_indices[:, 1], spike_indices[:, 2], spike_indices[:, 3]]
            aggregation[seg_pos, spike_indices[:, 0], spike_indices[:, 1], spike_indices[:, 2], spike_indices[:, 3]] += v
            seg_ind[spike_indices[:, 0], spike_indices[:, 1], spike_indices[:, 2], spike_indices[:, 3]] += 1
            t_last[spike_indices[:, 0], spike_indices[:, 1], spike_indices[:, 2], spike_indices[:, 3]] = t
            vmem_avg[spike_last.bool()] = 0
        if self.not_rpd:
            no_spike_pos = (1 - spike_last).nonzero()
            seg_pos = seg_ind[no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]]
            valid_pos = seg_pos < self.Ts
            seg_pos, no_spike_pos = (seg_pos[valid_pos], no_spike_pos[valid_pos])
            if self.readout == 'sum':
                v = vmem_avg[no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]]
            elif self.readout == 'last':
                v = vmem[no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]]
            elif self.readout == 'avg':
                v = vmem_avg[no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]] / (self.nb_steps - 1 - t_last[no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]])
            if self.write_zero:
                v *= 0
            aggregation[seg_pos, no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]] += v
        if self.training and self.use_physics_loss:
            if time_surface is not None:
                aux_loss = self.physics_loss(membrane_history, time_surface, dynamic_thresh, neg_dynamic_thresh)
            else:
                aux_loss = None
        current_thresh_val = dynamic_thresh.detach() if self.use_adaptive_thresh else dynamic_thresh
        if self.abs:
            output_preclamp = torch.nn.functional.relu(aggregation) / current_thresh_val
        else:
            output_preclamp = torch.nn.functional.relu(aggregation) / current_thresh_val - torch.nn.functional.relu(-aggregation) / self.neg_thresh_fixed
        output_clamp = float(getattr(self, 'output_clamp', 3.0))
        aggregation = torch.clamp(output_preclamp, min=-output_clamp, max=output_clamp)
        if self.training and self.use_physics_loss:
            return (aggregation, aux_loss)
        return aggregation
