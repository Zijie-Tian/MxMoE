import torch
import torch.nn as nn

from tqdm import tqdm
from transformers import PreTrainedTokenizer

from mxmoe.quant.data_utils import get_wikitext2


class Evaluator:
    def __init__(self, tokenizer: PreTrainedTokenizer, model_id: str, ppl_test_data: str="wikitext2"):
        self.seed = 42
        self.model_id = model_id

        if ppl_test_data == "wikitext2":
            _, testenc = get_wikitext2(100, self.seed, 4096, tokenizer, model_id, test_only=True)
            # self.trainloader = trainloader
            self.ppl_testenc = testenc
            self.ppl_dataset="wikitext2"
        else:
            raise ValueError(f"Unknown ppl test data: {ppl_test_data}")

    @torch.no_grad()
    def eval_ppl(self, model, input_len=4096):
        """Perplexity on wikitext2.

        Compatible with accelerate multi-device / GPU+CPU maps from ``load_hf_model``:
        put inputs on ``model.device`` (entry device) and labels on the logits device.
        NOTE: when ``lm_head`` is CPU/disk-offloaded, ``lm_head.weight.device`` is
        ``meta`` between forwards — never move tensors to it; use ``logits.device``.
        Do **not** call ``model.to(...)`` or move whole layers after accelerate dispatch.
        """
        model_use_cache = model.config.use_cache
        model.config.use_cache = False
        nsamples = self.ppl_testenc.numel() // input_len
        nlls = []

        loss_fct = nn.CrossEntropyLoss()
        for i in tqdm(range(nsamples), desc="Eval ppl", unit="sample"):
            # [bs, input_len]
            batch = self.ppl_testenc[:, (i * input_len) : ((i + 1) * input_len)].to(model.device)
            outputs = model.model(batch)
            hidden_states = outputs[0]
            # [bs, input_len, vocab_size]
            logits = model.lm_head(hidden_states)
            # [bs, input_len-1, vocab_size]
            shift_logits = logits[:, :-1, :]
            # [bs, input_len-1]
            shift_labels = batch[:, 1:].to(logits.device)
            loss = loss_fct(
                # [bs * (input_len-1), vocab_size]
                shift_logits.view(-1, shift_logits.size(-1)),
                # [bs * (input_len-1)]
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * input_len
            nlls.append(neg_log_likelihood)


        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * input_len)).item()
        print(self.ppl_dataset, ppl)

        model.config.use_cache = model_use_cache

        return ppl
    

    def eval_mmlu(self, model):
        pass


    def eval_tasks(self, model, task_list: list[str], bs=128, backend="hf"):
        import lm_eval
        from lm_eval import tasks
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM

        if backend == "hf":
            hflm = HFLM(pretrained=model, batch_size=bs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # tasks_list = tasks.get_task_dict(task_names)
        # task_names = lm_eval_utils.pattern_match(tasks, ALL_TASKS)
        eval_tasks = tasks.get_task_dict(task_list)
        print(f"Evaluated task_names: {eval_tasks.keys()}")
        results = lm_eval.simple_evaluate(hflm, tasks=list(eval_tasks.keys()), batch_size=bs)['results']

        metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
        # print(metric_vals)
        print(f"Evaluated tasks: {eval_tasks.keys()}\n{metric_vals}")
        return metric_vals
