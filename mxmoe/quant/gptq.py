# Adapted from https://github.com/IST-DASLab/gptq
import math
import time

import torch
import torch.nn as nn
import numpy as np
import transformers

# from quant import *
from mxmoe.quant.moe_utils import load_hf_model, load_tokenizer
from mxmoe.quant.data_utils import get_wikitext2
from tqdm import tqdm

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits:
            self.maxq = torch.tensor(-1) 

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-xmin / self.scale)
            #   self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


class GPTQ:

    def __init__(self, layer: nn.Linear):
        assert isinstance(layer, nn.Linear)
        self.layer = layer
        self.dev = self.layer.weight.device
        # [N, K]
        W = layer.weight.data.clone()
        self.rows = W.shape[0]      # N
        self.columns = W.shape[1]   # K
        # [K, K]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1])) # [M, K]
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t()) # [K, K]

    def fasterquant(
        self,
        blocksize=128,
        percdamp=.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        return_quant_state: bool = False,
    ):
        """GPTQ weight update.

        When ``return_quant_state=True``, also returns a dict with:
          - ``scale``: [out_features, n_groups] float
          - ``zero``:  [out_features, n_groups] float (GPTQ additive zero-point)
          - ``g_idx``: [in_features] int32 group id per input channel
          - ``weight``: fake-quant float weight [out, in] (same as written to the layer)
        """
        # [N, K]
        W = self.layer.weight.data.clone()
        W = W.float()
        rows, columns = self.rows, self.columns

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)
        # [K, K]
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        g_idx = torch.tensor(
            [i // groupsize if groupsize > 0 else 0 for i in range(columns)],
            dtype=torch.int32,
            device=self.dev,
        )

        if static_groups:
            import copy
            groups = []
            # Find params on original column order before act-order permute.
            step = groupsize if groupsize > 0 else columns
            for i in range(0, columns, step):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + step)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
            g_idx = g_idx[perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        # Per-column scale/zero in the (possibly permuted) column order used during quant.
        scale_cols = torch.zeros((rows, columns), device=self.dev, dtype=torch.float32)
        zero_cols = torch.zeros((rows, columns), device=self.dev, dtype=torch.float32)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                # scale/zero shaped [N, 1] for weight quantizer
                sc = self.quantizer.scale.flatten().float()
                z = self.quantizer.zero.flatten().float()
                if sc.numel() == 1:
                    sc = sc.expand(rows)
                    z = z.expand(rows)
                scale_cols[:, i1 + i] = sc
                zero_cols[:, i1 + i] = z

                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print(f"Time {time.time() - tick:.2f}; Error: {torch.sum(Losses).item()}")

        if actorder:
            Q = Q[:, invperm]
            scale_cols = scale_cols[:, invperm]
            zero_cols = zero_cols[:, invperm]
            g_idx = g_idx[invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        if not return_quant_state:
            return None

        # Compact per-group scale/zero: take first column of each group (static groups).
        if groupsize is None or groupsize <= 0:
            n_groups = 1
            scales = scale_cols[:, :1].contiguous()
            zeros = zero_cols[:, :1].contiguous()
            g_idx = torch.zeros(columns, dtype=torch.int32, device=self.dev)
        else:
            n_groups = (columns + groupsize - 1) // groupsize
            scales = torch.zeros((rows, n_groups), device=self.dev, dtype=torch.float32)
            zeros = torch.zeros((rows, n_groups), device=self.dev, dtype=torch.float32)
            for g in range(n_groups):
                col = min(g * groupsize, columns - 1)
                scales[:, g] = scale_cols[:, col]
                zeros[:, g] = zero_cols[:, col]
            g_idx = torch.tensor(
                [min(i // groupsize, n_groups - 1) for i in range(columns)],
                dtype=torch.int32,
                device=self.dev,
            )

        return {
            "scale": scales.cpu(),
            "zero": zeros.cpu(),
            "g_idx": g_idx.cpu().to(torch.int32),
            "weight": Q.detach().cpu(),
        }

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()

# Find all layers of a certain type in a given module.
def find_layers(module: nn.Module, filters: list[str]=[], layers=[nn.Linear], name=''):
    if type(module) in layers and any([f in name for f in filters]):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, filters=filters, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.no_grad()
def llama_sequential(args, model, dataloader, dev):
    seqlen = dataloader[0].shape[1]
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    ori_dev = model.model.embed_tokens.weight.device

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    # model.model.norm = model.model.norm.to(dev)
    rotary_emb = getattr(model.model, "rotary_emb", None)
    if rotary_emb is not None:
        model.model.rotary_emb = rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    print(model.model.embed_tokens.weight.device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    # Newer transformers (e.g. 4.5x Qwen2MoE) require position_embeddings on layer forward.
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            # model(batch[0].to(dev))
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].to(ori_dev)
    model.model.embed_tokens = model.model.embed_tokens.to(ori_dev)
    # model.model.norm = model.model.norm.to(ori_dev)
    if rotary_emb is not None:
        model.model.rotary_emb = rotary_emb.to(ori_dev)
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    print('Ready.')

    filters = ['w1', 'w2', 'w3', 'up_proj', 'gate_proj', 'down_proj']
    if args.quant_attn:
        filters.extend(['q_proj', 'k_proj', 'v_proj', 'o_proj'])

    quantizers = {}
    for i in tqdm(range(len(layers)), desc="Layers"):
        if "cpu" in ori_dev.type:
            layer = layers[i].to(dev)
        else:
            layer = layers[i]

        full = find_layers(layer, filters=filters)

        # if args.true_sequential:
        #     sequential = [
        #         ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
        #         ['self_attn.o_proj'],
        #         ['mlp.up_proj', 'mlp.gate_proj'],
        #         ['mlp.down_proj']
        #     ]
        # else:
        #     sequential = [list(full.keys())]
        sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(f"Layer-{i}: `{name}` ...")
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                quantizers[f'model.layers.{i}.{name}'] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

        if "cpu" in ori_dev.type:
            layer = layers[i].to(ori_dev)

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

def gptq_param_parser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='model ID.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=42, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[1, 2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )

    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )

    parser.add_argument(
        '--quant-attn', action='store_true',
        help='Whether to quantize weights in attention module.'
    )
    parser.add_argument(
        '--rotation', action='store_true',
        help='Whether to apply hadamard rotation(Qurot) to the weight matrix.'
    )
    parser.add_argument(
        '--online-had', action='store_true',
        help='enable online hadamard rotation.'
    )

    return parser

if __name__ == '__main__':
    from project_config import ID2NAME
    parser = gptq_param_parser()
    args = parser.parse_args()
    transformers.set_seed(args.seed)

    model_id = args.model
    device_map = "cpu" if model_id in ["qwen2_moe_57b", "mixtral"] else None
    model = load_hf_model(model_id, device_map=device_map)
    tokenizer = load_tokenizer(model_id)
    trainloader, testloader = get_wikitext2(args.nsamples, args.seed, 4096, tokenizer, model_id)

    DEV = torch.device('cuda:0')
    if args.wbits < 16:
        if args.rotation:
            from mxmoe.quant.rotation import ModelRotator
            rotator = ModelRotator(model, "hadamard")
            rotator.rotate_model(model, enable_online_rotation=args.online_had)

        tick = time.time()
        quantizers = llama_sequential(args, model, trainloader, DEV)
        print(f"Time Cost:{time.time() - tick:.3f}")

    # # Evaluation
    # from evaluator import Evaluator
    # evaluator = Evaluator(tokenizer, model_id, "wikitext2")
    # model = model.to(DEV)
    # evaluator.eval_ppl(model, input_len=4096)
    
    # Save fake-quantized model weights
    if args.wbits < 16 and args.save:
        # llama_pack3(model, quantizers)
        qattn = "_qattn" if args.quant_attn else ""
        rotation = "_had" if args.rotation else ""
        sym = "_sym" if args.sym else ""
        save_name = f"{args.save}/{model_id}-w{args.wbits}_g{args.groupsize}{sym}{rotation}{qattn}_n{args.nsamples}.pt"
        torch.save(model.state_dict(), save_name)
        print(f"Saved to `{save_name}`")
