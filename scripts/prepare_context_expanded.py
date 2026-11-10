"""Prepare paired full-build and local-merge ICL episodes."""
from collections import defaultdict
import json
import random

import context_data as d
from data_helpers import rank_key, short_key, write_jsonl, seed

SOURCE = d.ROOT / 'data/brickgpt_icl_ab_rebuild_inputs_v1'

OUT = d.ROOT / 'data/brickgpt_icl_ab_v1'
GROUPS = {
    f'{h}x{w}': sorted({(h, w), (w, h)})
    for h in range(1, 10) for w in range(h, 10) if (h, w) not in d.OLD
}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def merge_candidates(bricks):
    """Find exact rectangular intervals once for all dimension groups."""
    intervals = defaultdict(list)
    for i, (_, _, x, y, z) in enumerate(bricks):
        min_x, min_y, max_x, max_y = x, y, x, y
        area = 0
        for j in range(i, len(bricks)):
            h, w, x, y, layer = bricks[j]
            if layer != z:
                break
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x+h), max(max_y, y+w)
            h, w = max_x-min_x, max_y-min_y
            if max(h, w) > 9:
                break
            area += bricks[j][0] * bricks[j][1]
            if j > i and area == h*w:
                group = f'{min(h,w)}x{max(h,w)}'
                if group in GROUPS:
                    intervals[group].append((i, j, (h, w, min_x, min_y, z)))
    return intervals


def scan(structures):
    pools = defaultdict(dict)
    for index, (sid, row) in enumerate(structures.items(), 1):
        original = d.parse(row['bricks'])
        occupied = d.cells(original)
        for group, intervals in merge_candidates(original).items():
            modified, operations, cursor = [], [], 0
            for start, end, replacement in intervals:
                if start < cursor:
                    continue
                before = original[start:end+1]
                assert d.cells(before) == d.cells([replacement])
                assert replacement[2:] == tuple(min(b[i] for b in before) for i in (2, 3, 4))
                modified.extend(original[cursor:start])
                modified.append(replacement)
                operations.append(dict(original_steps=list(range(start+1, end+2)),
                                       modified_steps=[len(modified)], before=d.to_text(before),
                                       replacement=d.to_text([replacement])))
                cursor = end+1
            modified.extend(original[cursor:])
            assert occupied == d.cells(modified), sid
            pools[group][sid] = dict(
                variant_id=f'{sid}:{group}', structure_id=sid, group=group,
                original_brick_count=len(original), bricks=d.to_text(modified),
                operations=operations,
            )
        if index % 1000 == 0:
            print(f'Scanned {index}/{len(structures)} structures', flush=True)
    return pools


def episode(group, target, caption, contexts, structures, pools, key):
    rng = random.Random(seed(key))
    context = []
    for sid in contexts:
        variant = pools[group][sid]
        context.append(dict(structure_id=sid, variant_id=variant['variant_id'],
                            caption_index=rng.randrange(len(structures[sid]['captions'])),
                            operation_index=rng.randrange(len(variant['operations']))))
    return dict(sample_id=key, group=group, structure_id=target, caption_index=caption,
                variant_id=f'{target}:{group}', context=context)


def render(row, structures, variants, test=False):
    group = row['group']
    target = structures[row['structure_id']]
    caption = d.request_caption(target['captions'][row['caption_index']], group)
    context = row['context']
    assert len({row['structure_id'], *(x['structure_id'] for x in context)}) == 4
    # Test generation only needs the prompt, even for an older target with no merge.
    lookup = variants if row['variant_id'] in variants else {**variants, row['variant_id']: {'bricks': ''}}
    a = d.render_messages(row, structures, lookup)
    local = []
    for demo in context:
        op = variants[demo['variant_id']]['operations'][demo['operation_index']]
        dim = op['replacement'].split()[0]
        local.append(f'### Input:\nUse the new {dim} brick to replace these consecutive bricks.\n'
                     f'{op["before"]}\n### Output:\n{op["replacement"]}')
    prompt = a[1]['content']
    head, examples = prompt.split('Here are some example LEGO models:', 1)
    _, tail = examples.rsplit('Do NOT copy the examples', 1)
    b = [a[0], {'role': 'user', 'content':
         head + 'The examples show how consecutive bricks can be replaced by one new brick '
         'while covering exactly the same cells. For the final task, generate the full assembly sequence.\n\n'
         + '\n'.join(local) + '\nDo NOT copy the examples' + tail}, a[2]]
    assert caption in b[1]['content'] and a[2] == b[2]
    return dict(row, caption=caption, messages_a=a[:2] if test else a, messages_b=b[:2] if test else b)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d.GROUPS.update(GROUPS)
    structures = {r['structure_id']: r for r in d.read_jsonl(SOURCE / 'structures.jsonl.gz')
                  if r.get('geometry_valid') and r['captions']}
    pools = scan(structures)
    fixed = load(SOURCE / 'epoch_test_targets.json')['tasks']
    final = load(SOURCE / 'final_4x4_test_targets.json')['tasks']
    assert len(final) == len({t['structure_id'] for t in final}) == 30
    old_target_ids = {t['structure_id'] for t in fixed + final}
    seen38 = sorted(sid for sid in pools['3x8'] if structures[sid]['split'] == 'test'
                    and sid not in old_target_ids)
    extra = [dict(task_id=f'3x8_case_{i:02d}', structure_id=sid, group='3x8', caption_index=0)
             for i, sid in enumerate(random.Random(42).sample(seen38, 5), 1)]
    target_ids = old_target_ids | {t['structure_id'] for t in extra}
    tests = {}
    reserved = set(target_ids)
    for task in fixed + extra + final:
        key = task['task_id']
        if key in tests:
            assert tests[key]['structure_id'] == task['structure_id']
            continue
        group, sid = task['group'], task['structure_id']
        available = sorted(s for s in pools[group] if s not in target_ids and structures[s]['split'] == 'test')
        prior = [x['structure_id'] for x in task.get('context', [])]
        contexts = prior if len(prior) == 3 and set(prior) <= set(available) else random.Random(seed(key)).sample(available, 3)
        reserved.update(contexts)
        tests[key] = episode(group, sid, task['caption_index'], contexts, structures, pools, key)

    shortlists = {}
    selected = {}
    for group in GROUPS.keys() - {'4x4'}:
        shortlists[group] = sorted(pools[group].values(), key=rank_key)[:500]
        selected[group] = sorted(shortlists[group], key=short_key)[:250]

    # One assignment per structure, with deterministic per-group balancing.
    splits = {sid: 'test' for sid in reserved}
    for group in sorted(selected, key=lambda g: (len(selected[g]), g)):
        ids = [v['structure_id'] for v in selected[group]]
        free = sorted(set(ids) - splits.keys())
        random.Random(seed(f'42:split:{group}')).shuffle(free)
        counts = {s: sum(splits.get(sid) == s for sid in ids) for s in ('train', 'validation', 'test')}
        quotas = dict(train=int(len(ids)*.8), validation=int(len(ids)*.1))
        quotas['test'] = len(ids) - sum(quotas.values())
        for sid in free:
            split = max(counts, key=lambda s: quotas[s] - counts[s])
            splits[sid] = split
            counts[split] += 1

    variants = {v['variant_id']: v for vs in selected.values() for v in vs}
    for test in tests.values():
        group = test['group']
        for sid in [test['structure_id'], *(x['structure_id'] for x in test['context'])]:
            if sid in pools[group]:
                v = pools[group][sid]
                variants[v['variant_id']] = v
    stats = {}
    samples = defaultdict(list)
    for group in sorted(selected):
        counts = {}
        skipped = {}
        for split in ('train', 'validation', 'test'):
            ids = sorted(v['structure_id'] for v in selected[group] if splits[v['structure_id']] == split)
            counts[split] = len(ids)
            if len(ids) < 4:
                skipped[split] = len(ids)
                continue
            rng = random.Random(seed(f'42:episodes:{group}:{split}'))
            targets = rng.sample(ids, min(100, len(ids))) if split == 'train' else ids
            for sid in targets:
                contexts = rng.sample([s for s in ids if s != sid], 3)
                row = episode(group, sid, rng.randrange(len(structures[sid]['captions'])),
                              contexts, structures, pools, f'{split}:{group}:{sid}')
                samples[split].append(render(row, structures, variants))
        stats[group] = dict(candidates=len(pools[group]), top500=len(shortlists[group]),
                            selected=len(selected[group]), split_structures=counts,
                            train_samples=min(100, counts['train']) if counts['train'] >= 4 else 0,
                            excluded_insufficient_context=skipped)
    test_ids = {sid for row in tests.values()
                for sid in [row['structure_id'], *(x['structure_id'] for x in row['context'])]}
    train_ids = {sid for row in samples['train']
                 for sid in [row['structure_id'], *(x['structure_id'] for x in row['context'])]}
    assert not test_ids & train_ids
    assert not any(row['group'] == '4x4' for rows in samples.values() for row in rows)
    for split, rows in samples.items():
        for row in rows:
            assert all(splits[sid] == split for sid in
                       [row['structure_id'], *(x['structure_id'] for x in row['context'])])
        write_jsonl(OUT / f'{split}.jsonl.gz', rows)
    rendered_tests = {k: render(row, structures, variants, test=True) for k, row in tests.items()}
    save('fixed_tests.json', dict(epoch=[rendered_tests[t['task_id']] for t in fixed+extra],
                                 final=[rendered_tests[t['task_id']] for t in final]))
    write_jsonl(OUT / 'variants.jsonl.gz', variants.values())
    write_jsonl(OUT / 'structures.jsonl.gz', (dict(structures[sid], split=split) for sid, split in splits.items()))
    save('selection.json', {g: dict(top500=[v['structure_id'] for v in shortlists[g]],
                                   selected=[v['structure_id'] for v in selected[g]]) for g in sorted(selected)})
    totals = {s: len(samples[s]) for s in ('train', 'validation', 'test')}
    audit = dict(groups=stats, samples=totals, test_structure_train_overlap=0,
                 geometry_verified_variants=sum(map(len, pools.values())),
                 verified_merges=sum(len(v['operations']) for pool in pools.values() for v in pool.values()),
                 epoch_tests=20, final_tests=30, structures_scanned=len(structures),
                 four_by_four_candidates=len(pools['4x4']),
                 test_targets_without_consecutive_merge=[
                     k for k, row in tests.items() if row['structure_id'] not in pools[row['group']]
                 ])
    save('audit.json', audit)
    lines = ['# Expanded brick ICL data', '', f'Samples per experiment: {totals}.',
             f'Geometry verified: {audit["geometry_verified_variants"]} variants, {audit["verified_merges"]} merges.',
             'Fixed test structure overlap with training: 0. Epoch tests: 20. Final tests: 30.',
             '', '| Size | Candidates | Top 500 | Selected | Train/val/test structures | Training samples |',
             '|---|---:|---:|---:|---|---:|']
    for g, row in stats.items():
        sizes = '/'.join(str(row['split_structures'][s]) for s in ('train', 'validation', 'test'))
        lines.append(f'| {g} | {row["candidates"]} | {row["top500"]} | {row["selected"]} | {sizes} | {row["train_samples"]} |')
    (OUT / 'DATA_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k != 'groups'}), flush=True)


if __name__ == '__main__':
    main()
