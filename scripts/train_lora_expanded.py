"""Run expanded ICL A then B, including all generation evaluations."""
import gc
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import lora_runtime as train
import context_data as data
import eval_runtime as ev
import icl_experiment as engine
from prepare_context_expanded import GROUPS, OUT as DATA
import torch

OUT = data.ROOT / 'output/icl_lora_train_expanded_v1'
EPOCHS = 5
ACCUMULATION = 8


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def encode(tokenizer, variant):
    samples = {}
    for split in ('train', 'validation'):
        rows = list(data.read_jsonl(DATA / f'{split}.jsonl.gz'))
        if split == 'validation':
            rows = random.Random(42).sample(rows, 5)
        items = []
        for row in rows:
            messages = row[f'messages_{variant}']
            prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
            ids = tokenizer.apply_chat_template(messages, tokenize=True)
            assert ids[:len(prefix)] == prefix and ids[-1] == tokenizer.eos_token_id
            assert len(ids) <= tokenizer.model_max_length, row['sample_id']
            assert tokenizer.decode(ids[len(prefix):]) == messages[-1]['content'] + tokenizer.eos_token
            items.append(dict(input_ids=ids, answer_start=len(prefix), answer_tokens=len(ids)-len(prefix),
                              sample_id=row['sample_id']))
        samples[split] = items
    return samples


def test(tasks, dest, checkpoint, variant):
    if (dest / 'completed.json').exists():
        return read(dest / 'completed.json')
    ev.OUT = dest
    ev.INSTANCE = None
    ev.EvalLLM.adapter_path = str(checkpoint)
    engine.LocalLLM = ev.EvalLLM
    engine.CONDITIONS = ['after']
    engine.build_instruction = ev.instruction
    engine.make_messages = ev.messages
    engine.POSTPROCESS_RESULT = ev.postprocess
    engine.AFTER_RUN = ev.preview

    def summarize():
        from collections import Counter
        rows = []
        for path in sorted(dest.glob('*/runs/*/result.json')):
            row = read(path)
            decisions = Counter()
            for item in read(path.with_name('proposals.json')):
                try:
                    bricks = data.parse(item['text'])
                except ValueError:
                    continue
                if bricks and f'{min(bricks[0][:2])}x{max(bricks[0][:2])}' == row['group']:
                    decisions[item['decision']] += 1
            rows.append(dict(task_id=row['task_id'], group=row['group'],
                             proposed=sum(decisions.values()), accepted=decisions['success'],
                             decisions=dict(decisions), final=row['n_new_bricks'],
                             finished_with_new=row['finished_with_new']))
        groups = {}
        for group in sorted({t['group'] for t in tasks}):
            subset = [r for r in rows if r['group'] == group]
            groups[group] = dict(completed=len(subset), planned=sum(t['group']==group for t in tasks),
                                 tasks_proposing=sum(r['proposed']>0 for r in subset),
                                 proposals=sum(r['proposed'] for r in subset),
                                 accepted=sum(r['accepted'] for r in subset),
                                 rejected=sum(r['proposed']-r['accepted'] for r in subset),
                                 tasks_using=sum(r['final']>0 for r in subset),
                                 finished_with_new=sum(r['finished_with_new'] for r in subset))
        result = dict(completed=len(rows), planned=len(tasks), groups=groups, cases=rows)
        engine.write_json(dest / 'summary.json', result)
        return result

    ev.summarize = summarize
    for group in sorted({t['group'] for t in tasks}):
        cases = [dict(t, task_id=t['sample_id'], kind='text', messages=t[f'messages_{variant}'])
                 for t in tasks if t['group']==group]
        engine.OUT = dest / group
        engine.NEW = {tuple(sorted(pair)) for pair in GROUPS[group]}
        engine.write_json(engine.OUT/'design.json', dict(tasks=cases, group=group, seeds=[42],
                          conditions=['after'], checkpoint=str(checkpoint)))
        engine.run(SimpleNamespace(chat=False, smoke=False, overwrite=False, limit=None,
                   max_bricks=192, max_brick_rejections=40, max_regenerations=0, seconds_per_run=180))
        ev.INSTANCE = None
        gc.collect()
        torch.cuda.empty_cache()
    result = summarize()
    assert result['completed'] == len(tasks)
    engine.write_json(dest / 'completed.json', result)
    return result


def optimizer_for(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=1e-4, weight_decay=.01, fused=True)


def experiment(variant, tests):
    train.OUT = OUT / variant.upper()
    train.OUT.mkdir(parents=True, exist_ok=True)
    if (train.OUT / 'completed.json').exists():
        return read(train.OUT / 'completed.json')
    train.save_status('encoding')
    tokenizer = train.AutoTokenizer.from_pretrained(train.MERGED, local_files_only=True)
    samples = encode(tokenizer, variant)
    batches = math.ceil(len(samples['train'])/ACCUMULATION)
    total = EPOCHS*batches
    saves = {math.ceil(total*n/5) for n in range(1, 6)}
    train.setup.save(train.OUT/'config.json', dict(
        experiment=variant, samples=len(samples['train']), validation_samples=len(samples['validation']),
        epochs=EPOCHS, updates_per_epoch=batches, total_updates=total, checkpoint_steps=sorted(
            saves | {batches*n for n in range(1, EPOCHS+1)}), bf16=True,
        micro_batch=1, gradient_accumulation=ACCUMULATION, rank=16, alpha=32, dropout=.05,
        lr=1e-4, warmup_updates=16, data=str(DATA)))
    latest = train.OUT/'latest_checkpoint.json'
    if not (train.OUT/'preflight.json').exists():
        train.save_status('preflight')
        model = train.load_model()
        optimizer = optimizer_for(model)
        longest = max(samples['train'], key=lambda row: len(row['input_ids']))
        torch.cuda.reset_peak_memory_stats()
        value, norm = train.update(model, optimizer, [longest])
        train.setup.save(train.OUT/'preflight.json', dict(passed=True, loss=value, grad_norm=norm,
                         longest_tokens=len(longest['input_ids']),
                         peak_gib=torch.cuda.max_memory_allocated()/2**30))
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    # Load only one model at a time, including when switching to generation.
    for epoch in range(EPOCHS):
        prior = read(latest) if latest.exists() else None
        if not prior or prior['epoch'] <= epoch:
            train.set_seed(42)
            model = train.load_model(prior['path'] if prior else None)
            optimizer = optimizer_for(model)
            scheduler = train.get_cosine_schedule_with_warmup(optimizer, 16, total)
            model.validation_sample_ids = [r['sample_id'] for r in samples['validation']]
            step, cursor, best = 0, 0, float('inf')
            if prior:
                state = torch.load(Path(prior['path'])/'training_state.pt', map_location='cpu', weights_only=True)
                optimizer.load_state_dict(state['optimizer'])
                scheduler.load_state_dict(state['scheduler'])
                torch.set_rng_state(state['torch_rng'])
                torch.cuda.set_rng_state_all(state['cuda_rng'])
                step, cursor, best = state['step'], state['cursor'], state['best_loss']
                del state
            order = list(samples['train'])
            random.Random(42+epoch).shuffle(order)
            for index in range(cursor, len(order), ACCUMULATION):
                start = time.perf_counter()
                batch = order[index:index+ACCUMULATION]
                value, norm = train.update(model, optimizer, batch)
                scheduler.step()
                step += 1
                row = dict(epoch=epoch+1, step=step, total_steps=total, train_loss=value,
                           grad_norm=norm, lr=scheduler.get_last_lr()[0],
                           seconds=time.perf_counter()-start)
                train.save_status('training', **row)
                train.log_event('update', **row)
                print(f'{variant.upper()} TRAIN {step}/{total} loss={value:.5f}', flush=True)
                if step in saves and index+len(batch) < len(order):
                    train.save_checkpoint(model, optimizer, scheduler, step, epoch, index+len(batch), best)
            value, seconds = train.evaluate(model, samples['validation'])
            train.log_event('validation', epoch=epoch+1, validation_loss=value, seconds=seconds)
            train.save_checkpoint(model, optimizer, scheduler, step, epoch+1, 0, min(best, value), value)
            del model, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()
        checkpoint = train.OUT/'checkpoints'/f'step_{(epoch+1)*batches:04d}'
        train.save_status('epoch_testing', epoch=epoch+1, step=(epoch+1)*batches, total_steps=total)
        test(tests['epoch'], train.OUT/'epoch_tests'/f'epoch_{epoch+1}', checkpoint, variant)
    train.save_status('final_testing', step=total, total_steps=total)
    result = test(tests['final'], train.OUT/'final_4x4', checkpoint, variant)
    train.setup.save(train.OUT/'completed.json', result)
    train.save_status('completed', step=total, total_steps=total)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train.OUT = OUT
    data.GROUPS.update(GROUPS)
    torch.cuda.set_device(0)
    assert torch.cuda.is_bf16_supported()
    train.prepare_base()
    tests = read(DATA/'fixed_tests.json')
    results = {}
    for variant in ('a', 'b'):
        train.setup.save(OUT/'status.json', dict(state='running', experiment=variant.upper()))
        results[variant] = experiment(variant, tests)
    train.setup.save(OUT/'comparison.json', {k: v['groups'] for k, v in results.items()})
    train.setup.save(OUT/'status.json', dict(state='completed'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        failed_experiment = train.OUT.name
        train.save_status('failed', error=str(exc))
        train.setup.save(OUT/'status.json', dict(state='failed', experiment=failed_experiment, error=str(exc)))
        raise
