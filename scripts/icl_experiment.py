"""Reproducible BrickGPT ICL pilot; geometry/connectivity only, never physics."""
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import brickgpt_stage1 as setup
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, set_seed
from peft import PeftModel
from brickgpt.data import Brick, BrickStructure
from brickgpt.models.llm import LLM

ROOT = setup.ROOT
OUT = ROOT / 'output/icl_4x4'
OLD = {(1, 1), (1, 2), (1, 4), (1, 6), (1, 8), (2, 2), (2, 4), (2, 6)}
NEW = {(4, 4)}
CONDITIONS = ['original', 'description', 'fewshot_1', 'fewshot_2', 'fewshot_5', 'fewshot_10', 'merge_pairs_2']
CHAT_CONDITIONS = ['chat_2', 'chat_5', 'local_pairs_3', 'official_library_only', 'official_fewshot_2',
                   'official_zero_shot', 'official_fewshot_template_2']


def canonical(b):
    return tuple(sorted((b.h, b.w)))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def parse(text):
    return BrickStructure.from_txt(text)










class GeometryGenerationError(RuntimeError):
    pass




class BrickSyntaxProcessor(LogitsProcessor):
    def __init__(self, tokenizer, start_length):
        self.start = start_length
        fields = [tuple(str(i) for i in range(1, 9)) + (tokenizer.eos_token,), ('x',),
            tuple(str(i) for i in range(1, 9)), (' (',), tuple(str(i) for i in range(20)),
            (',',), tuple(str(i) for i in range(20)), (',',), tuple(str(i) for i in range(20)), (')\n',)]
        self.allowed = []
        for field in fields:
            ids = [tokenizer.encode(s, add_special_tokens=False) for s in field]
            assert all(len(x) == 1 for x in ids), field
            self.allowed.append([x[0] for x in ids])

    def __call__(self, input_ids, scores):
        field = (input_ids.shape[1] - self.start) % 10
        mask = torch.full_like(scores, -torch.inf)
        mask[:, self.allowed[field]] = scores[:, self.allowed[field]]
        return mask


class LocalLLM(LLM):
    def __init__(self, model_name, device):
        self.device = device
        adapter = ROOT / 'temp/models/BrickGPT'
        self.tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(setup.resolve_base_model(None), local_files_only=True,
            torch_dtype=torch.float16, device_map={'': device}, low_cpu_mem_usage=True)
        self.model = PeftModel.from_pretrained(base, adapter, local_files_only=True).eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.kv_cache = self.kv_cache_saved = None
        self.input_ids_cache = self.input_ids_cache_saved = None


def metrics(bricks, task):
    bs = bricks.bricks
    valid_types = all(canonical(b) in OLD | NEW for b in bs)
    geometry = bool(bs) and valid_types and not bricks.has_collisions() and not bricks.has_out_of_bounds_bricks()
    connected = bool(geometry and bricks.is_connected())
    new_count = sum(canonical(b) in NEW for b in bs)
    result = {'n_bricks': len(bs), 'n_new_bricks': new_count, 'uses_new_brick': new_count > 0,
        'new_brick_fraction': new_count / len(bs) if bs else 0,
        'new_type_counts': dict(Counter(f'{min(b.h,b.w)}x{max(b.h,b.w)}' for b in bs if canonical(b) in NEW)),
        'geometry_valid': geometry, 'connected': connected, 'physically_stable': None}
    if task['kind'] == 'replacement':
        source = parse(task['source'])
        same = np.array_equal(source.voxel_occupancy > 0, bricks.voxel_occupancy > 0)
        result['exact_voxel_preservation'] = bool(same)
        result['voxel_iou'] = float(np.count_nonzero((source.voxel_occupancy > 0) & (bricks.voxel_occupancy > 0)) /
            max(1, np.count_nonzero((source.voxel_occupancy > 0) | (bricks.voxel_occupancy > 0))))
        result['target_type_used'] = any(canonical(b) == tuple(task['target_type']) for b in bs)
    return result


def run(args):
    design = json.loads((OUT / 'design.json').read_text(encoding='utf-8'))
    module = importlib.import_module('brickgpt.models.brickgpt')
    module.LLM = LocalLLM

    class Pilot(module.BrickGPT):
        def generate_brick(self, prompt=None, temperature=None):
            start = (prompt.shape[1] if isinstance(prompt, torch.Tensor) else self.llm.input_ids_cache.shape[1])
            ids = self.llm(prompt, return_as_ids=True, max_new_tokens=10,
                temperature=temperature or self.temperature, top_k=0, top_p=1.,
                logits_processor=LogitsProcessorList([BrickSyntaxProcessor(self.llm.tokenizer, start)]))
            eos = bool(ids.numel() and ids[0].item() == self.llm.tokenizer.eos_token_id)
            if eos:
                self.eos = True
            return self.llm.tokenizer.decode(ids, skip_special_tokens=True)

        @staticmethod
        def _try_adding_brick(text, bricks, rejected):
            if text in rejected:
                return 'already_rejected'
            try:
                b = Brick.from_txt(text)
            except ValueError:
                return 'ill_formatted'
            if canonical(b) not in OLD | NEW:
                return 'not_in_expanded_library'
            if not bricks.brick_in_bounds(b):
                return 'out_of_bounds'
            if bricks.brick_collides(b):
                return 'collision'
            return 'success'

        def generate_brick_with_rejection_sampling(self, prompt=None, bricks=None):
            if bricks is None:
                bricks = BrickStructure([])
            reasons = Counter()
            rejected = set()
            temperature = self.temperature
            for i in range(self.max_brick_rejections + 1):
                if self.run_time_limit and time.perf_counter() - self.run_started > self.run_time_limit:
                    self.last_bricks = BrickStructure(list(bricks.bricks))
                    raise GeometryGenerationError('wallclock_run_limit')
                self.llm.save_state()
                text = self.generate_brick(prompt, temperature)
                reason = 'eos' if not text else self._try_adding_brick(text, bricks, rejected)
                self.proposals.append({'attempt': self.attempt, 'text': text, 'decision': reason})
                if reason in ['success', 'eos']:
                    if reason == 'success' and (len(bricks) + 1) % 32 == 0:
                        print(f'  attempt {self.attempt}: {len(bricks)+1} bricks', flush=True)
                    return text, reasons
                reasons[reason] += 1
                self.llm.rollback_to_saved_state()
                rejected.add(text)
                if reason == 'already_rejected':
                    temperature = min(self.max_temperature, temperature + self.temperature_increase)
            self.last_bricks = BrickStructure(list(bricks.bricks))
            raise GeometryGenerationError(f'Rejection limit exceeded: {dict(reasons)}')

        def _generate_structure(self, caption, starting_bricks=None):
            self.attempt += 1
            self.eos = False
            if self.attempt == 0 and self.current_task.get('prefix'):
                starting_bricks = parse(self.current_task['prefix'])
            print(f'  generation attempt {self.attempt}', flush=True)
            if self.current_condition.startswith('chat_'):
                # Same sampling/geometry checks, with demonstrations represented
                # as actual alternating user/assistant messages.
                bricks = BrickStructure(list(starting_bricks.bricks)) if starting_bricks else BrickStructure([])
                messages = list(self.current_messages)
                if len(bricks):
                    messages.append({'role':'assistant','content':bricks.to_txt()})
                    prompt = self.llm.tokenizer.apply_chat_template(messages, continue_final_message=True, return_tensors='pt')
                else:
                    prompt = self.llm.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
                reasons = Counter()
                for index in range(self.max_bricks):
                    text, rejected = self.generate_brick_with_rejection_sampling(prompt if index == 0 else None, bricks=bricks)
                    reasons.update(rejected)
                    if not text:
                        break
                    bricks.add_brick(Brick.from_txt(text))
                result = bricks, reasons
            else:
                result = super()._generate_structure(caption, starting_bricks=starting_bricks or BrickStructure([]))
            self.last_bricks = result[0]
            self.attempt_metrics.append({'attempt': self.attempt, 'eos': self.eos,
                                         **metrics(result[0], self.current_task)})
            return result

        def _is_stable(self, bricks):
            # Empty prefixes must be accepted by the upstream trimming loop;
            # final evaluation still rejects empty generated structures.
            return not len(bricks) or bool(bricks.is_connected())

        def _remove_all_bricks_after_first_unstable_brick(self, bricks):
            if not len(bricks):
                return bricks
            trimmed = super()._remove_all_bricks_after_first_unstable_brick(bricks)
            if hasattr(self, 'rollback_events'):
                self.rollback_events.append({'attempt':self.attempt, 'before':len(bricks),
                    'retained':len(trimmed), 'removed':len(bricks)-len(trimmed),
                    'retained_bricks':trimmed.to_txt()})
            return trimmed

    cfg = module.BrickGPTConfig(str(ROOT / 'temp/models/BrickGPT'), device='cuda', use_gurobi=False,
        max_bricks=getattr(args,'max_bricks',192),
        max_brick_rejections=getattr(args,'max_brick_rejections',40),
        max_regenerations=getattr(args,'max_regenerations',2), temperature=0.6)
    print('Loading BrickGPT once for all conditions...', flush=True)
    model = Pilot(cfg)
    model.run_time_limit = getattr(args,'seconds_per_run',None)
    for h,w in NEW:
        assert model._try_adding_brick(f'{h}x{w} (0,0,0)', BrickStructure([]), set()) == 'success'
    assert model._try_adding_brick('3x3 (0,0,0)', BrickStructure([]), set()) == 'not_in_expanded_library'
    # A disconnected prefix may trim down to empty; this must terminate.
    assert not len(model._remove_all_bricks_after_first_unstable_brick(
        BrickStructure([Brick(h=2,w=2,x=0,y=0,z=1)])))
    manifest = {'config': asdict(cfg), 'syntax_mask': 'format only; dimensions 1..8, coordinates 0..19; identical across conditions',
        'validator_library': sorted(OLD | NEW), 'original_condition_validator_also_expanded': True,
        'constraint_does_not_require_novel_bricks': True, 'physics_enabled': False,
        'generation_note': 'Ten-token brick generated in one call with position-dependent format mask; upstream cache and rollback logic retained. max_bricks limits new bricks per attempt, including rollback continuations.',
        'versions': {p: setup.importlib.metadata.version(p) for p in ['torch','transformers','peft','tokenizers']},
        'gpu': torch.cuda.get_device_name(0),
        'design_sha256': hashlib.sha256((OUT / 'design.json').read_bytes()).hexdigest()}
    write_json(OUT / ('manifest_chat.json' if args.chat else 'manifest.json'), manifest)
    if args.chat:
        write_json(OUT / 'chat_extension_design.json', {'conditions':CHAT_CONDITIONS, 'seeds':[42],
            'n_runs':len(CHAT_CONDITIONS)*len(design['tasks']), 'reason':'Check multi-turn/local replacement examples, matched upstream wording, and the actual upstream zero/few-shot functions after negative preliminary results and the user question about prompt consistency.',
            'exploratory_extension':True})
    plan = [(task, seed, cond) for task in design['tasks']
            for seed in ([42] if args.chat else design['seeds'])
            for cond in (CHAT_CONDITIONS if args.chat else CONDITIONS)]
    if args.smoke:
        plan = [(design['tasks'][-1], 42, 'description')]
    if args.limit:
        plan = plan[:args.limit]
    for index, (task, seed, condition) in enumerate(plan):
        run_id = f'{task["task_id"]}__s{seed}__{condition}'
        dest = OUT / ('smoke' if args.smoke else 'runs') / run_id
        if (dest / 'result.json').exists() and not args.overwrite:
            try:
                cached = json.loads((dest/'result.json').read_text(encoding='utf-8'))
                reusable = cached['run_id']==run_id and 'complete_connected_valid' in cached
                reusable = reusable and cached.get('design_sha256',manifest['design_sha256'])==manifest['design_sha256']
            except (ValueError,KeyError):
                reusable = False
            if reusable:
                print(f'[{index+1}/{len(plan)}] cached {run_id}', flush=True)
                continue
        dest.mkdir(parents=True, exist_ok=True)
        instruction = build_instruction(design, condition, task)
        model.instruction_fn = lambda caption, text=instruction: text
        model.current_task = task
        model.current_condition = condition
        model.proposals = []
        model.attempt_metrics = []
        model.rollback_events = []
        model.attempt = -1
        model.eos = False
        model.last_bricks = BrickStructure([])
        model.temperature = 1.0 if condition.endswith('_hot') else cfg.temperature
        messages = make_messages(design, condition, task)
        model.current_messages = messages
        input_messages = messages + ([{'role':'assistant','content':task['prefix']}] if task.get('prefix') else [])
        template_kwargs = {'continue_final_message':True} if task.get('prefix') else {'add_generation_prompt':True}
        prompt_tokens = len(model.llm.tokenizer.apply_chat_template(input_messages, **template_kwargs))
        (dest / 'prompt.txt').write_text(instruction, encoding='utf-8')
        write_json(dest / 'messages.json', input_messages)
        (dest / 'chat_prompt.txt').write_text(model.llm.tokenizer.apply_chat_template(input_messages,
            **template_kwargs, tokenize=False), encoding='utf-8')
        set_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model.run_started = start
        error = None
        print(f'[{index+1}/{len(plan)}] START {run_id}; prompt={prompt_tokens} tokens', flush=True)
        try:
            with torch.inference_mode():
                output = model(task['caption'])
            bricks = output['bricks']
            regen = output['n_regenerations']
        except (GeometryGenerationError, torch.OutOfMemoryError) as exc:
            error = str(exc)
            bricks = model.last_bricks
            regen = model.attempt
        torch.cuda.synchronize()
        result = {'run_id':run_id, 'task_id':task['task_id'], 'kind':task['kind'], 'condition':condition,
            'design_sha256':manifest['design_sha256'],
            'seed':seed, 'temperature':model.temperature,
            'prompt_tokens':prompt_tokens, 'seconds':time.perf_counter()-start,
            'eos':model.eos, 'error':error, 'regenerations':regen,
            'peak_allocated_vram_gib':torch.cuda.max_memory_allocated()/1024**3, **metrics(bricks,task)}
        result['connectivity_rollbacks'] = len(model.rollback_events)
        result['rollback_events'] = model.rollback_events
        result['candidate_rejections'] = dict(Counter(p['decision'] for p in model.proposals
            if p['decision'] not in ['success','eos']))
        result['candidate_retry_count'] = sum(result['candidate_rejections'].values())
        result['complete_connected_valid'] = bool(not error and model.eos and result['geometry_valid'] and result['connected'])
        result['success_with_new'] = result['complete_connected_valid'] and result['uses_new_brick']
        if task['kind'] == 'replacement':
            result['exact_replacement_success'] = bool(result['success_with_new'] and result['exact_voxel_preservation'] and result['target_type_used'])
        if globals().get('POSTPROCESS_RESULT'):
            result = POSTPROCESS_RESULT(result,task)
        (dest / 'bricks.txt').write_text(bricks.to_txt(), encoding='utf-8')
        write_json(dest / 'proposals.json', model.proposals)
        write_json(dest / 'attempts.json', model.attempt_metrics)
        write_json(dest / 'result.json', result)
        print(f'[{index+1}/{len(plan)}] DONE new={result["n_new_bricks"]}/{result["n_bricks"]} connected={result["connected"]} eos={result["eos"]} time={result["seconds"]:.1f}s error={error}', flush=True)
        model.llm.reset_cache()
        model.llm.input_ids_cache = model.llm.input_ids_cache_saved = model.llm.kv_cache_saved = None
        torch.cuda.empty_cache()
        if globals().get('AFTER_RUN'):
            AFTER_RUN(result)
