"""Shared geometry, official-template renderer and normalized dataset reader."""
import ast
import gzip
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/icl_lora_context_v1'
CACHE=ROOT/'temp/icl_lora_context_v1'
GROUPS={'1x3':[(1,3),(3,1)],'2x3':[(2,3),(3,2)],'2x8':[(2,8),(8,2)],'4x4':[(4,4)]}
OLD={(1,1),(1,2),(1,4),(1,6),(1,8),(2,2),(2,4),(2,6)}
LINE=re.compile(r'^(\d+)x(\d+) \((-?\d+),(-?\d+),(-?\d+)\)$')


def read_jsonl(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def parse(text):
    result=[]
    for line in text.splitlines():
        if not line.strip():continue
        match=LINE.fullmatch(line.strip())
        if not match:raise ValueError('invalid_brick_syntax')
        result.append(tuple(map(int,match.groups())))
    return result


def to_text(bricks):
    return ''.join(f'{h}x{w} ({x},{y},{z})\n' for h,w,x,y,z in bricks)


def cells(bricks):
    occupied=set()
    if not bricks:raise ValueError('empty_structure')
    for h,w,x,y,z in bricks:
        if h<1 or w<1 or min(x,y,z)<0 or x+h>20 or y+w>20 or z>=20:raise ValueError('out_of_bounds')
        new={(a*20+b)*20+z for a in range(x,x+h) for b in range(y,y+w)}
        if occupied & new:raise ValueError('collision')
        occupied.update(new)
    return occupied


def transform(bricks,group):
    changes={};consumed=set();operations=[]
    if group in ('1x3','2x3'):
        narrow=1 if group=='1x3' else 2
        for i,(h,w,x,y,z) in enumerate(bricks):
            if (h,w)==(narrow,6):new=[(narrow,3,x,y,z),(narrow,3,x,y+3,z)]
            elif (h,w)==(6,narrow):new=[(3,narrow,x,y,z),(3,narrow,x+3,y,z)]
            else:continue
            changes[i]=new;operations.append({'original_steps':[i+1],'replacement':to_text(new)})
    elif group=='2x8':
        lookup={b:i for i,b in enumerate(bricks)}
        for i in sorted(range(len(bricks)),key=lambda j:bricks[j]):
            if i in consumed:continue
            h,w,x,y,z=bricks[i]
            if (h,w)==(1,8):neighbor=(1,8,x+1,y,z);new=(2,8,x,y,z)
            elif (h,w)==(8,1):neighbor=(8,1,x,y+1,z);new=(8,2,x,y,z)
            else:continue
            j=lookup.get(neighbor)
            if j is None or j in consumed:continue
            first=min(i,j);changes[first]=[new];consumed.update((i,j))
            operations.append({'original_steps':sorted([i+1,j+1]),'replacement':to_text([new])})
    elif group=='4x4':
        remaining=set(range(len(bricks)))
        for anchor in sorted(range(len(bricks)),key=lambda i:(bricks[i][4],bricks[i][2],bricks[i][3])):
            _,_,x,y,z=bricks[anchor]
            if x+4>20 or y+4>20:continue
            inside=[i for i in remaining if bricks[i][4]==z and bricks[i][2]>=x and bricks[i][3]>=y
                and bricks[i][2]+bricks[i][0]<=x+4 and bricks[i][3]+bricks[i][1]<=y+4]
            if len(inside)<2 or sum(bricks[i][0]*bricks[i][1] for i in inside)!=16:continue
            first=min(inside);new=(4,4,x,y,z);changes[first]=[new]
            consumed.update(inside);remaining.difference_update(inside)
            operations.append({'original_steps':sorted(i+1 for i in inside),'replacement':to_text([new])})
    else:raise ValueError(group)
    modified=[];positions={}
    for i,b in enumerate(bricks):
        if i in changes:
            positions[i+1]=list(range(len(modified)+1,len(modified)+len(changes[i])+1));modified.extend(changes[i])
        elif i not in consumed:modified.append(b)
    for op in operations:op['modified_steps']=positions[min(op['original_steps'])]
    return modified,operations


def official_functions():
    source=ROOT/'BrickGPT/src/brickgpt/models/brickgpt.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    names={'create_instruction','create_instruction_zero_shot','create_instruction_few_shot','_create_example_instruction'}
    module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    namespace={};exec(compile(module,str(source),'exec'),namespace)
    assert names.issubset(namespace)
    return namespace


OFFICIAL=official_functions()


def request_caption(caption,group):
    dimensions=' and '.join(f'{h}x{w}' for h,w in GROUPS[group])
    return caption.strip()+f' Use the newly added {dimensions} bricks.'


def extend_library(text,group):
    line=next(x for x in text.splitlines() if x.startswith('Allowed brick dimensions'))
    return text.replace(line,line[:-1]+', '+', '.join(f'{h}x{w}' for h,w in GROUPS[group])+'.',1)


def render_parts(episode,structures,variants):
    """Regex-safe token boundaries; exact official few-shot formatting, including newlines."""
    group=episode['group'];target=structures[episode['structure_id']]
    caption=request_caption(target['captions'][episode['caption_index']],group)
    base=extend_library(OFFICIAL['create_instruction'](''),group)
    zero=OFFICIAL['create_instruction_zero_shot']('')[len(OFFICIAL['create_instruction'](''))+2:]
    head=('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.'
        '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'+base)
    parts=[(('head',group),head),(('caption_plain',group,target['structure_id'],episode['caption_index']),caption+'\n\n'),
        (('zero',),zero+'\n\nHere are some example LEGO models:\n\n')]
    for demo in episode['context']:
        row=structures[demo['structure_id']];variant=variants[demo['variant_id']]
        cap=request_caption(row['captions'][demo['caption_index']],group)
        parts.extend([(('input_marker',),'### Input:\n'),
            (('caption_output',group,row['structure_id'],demo['caption_index']),cap+'\n\n### Output:\n'),
            (('demo_bricks',variant['variant_id']),variant['bricks']+'\n\n')])
    parts.extend([(('final_request',),'Do NOT copy the examples, but create your own LEGO model for the following input.\n\n### Input:\n'),
        (('caption_output',group,target['structure_id'],episode['caption_index']),caption+'\n\n### Output:\n'),
        (('assistant_header',),'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'),
        (('answer',episode['variant_id']),variants[episode['variant_id']]['bricks']+'<|eot_id|>')])
    return parts


def render_messages(episode,structures,variants):
    group=episode['group']
    OFFICIAL['_few_shot_examples']=[{'caption':request_caption(structures[d['structure_id']]['captions'][d['caption_index']],group),
        'bricks':variants[d['variant_id']]['bricks']} for d in episode['context']]
    caption=request_caption(structures[episode['structure_id']]['captions'][episode['caption_index']],group)
    prompt=extend_library(OFFICIAL['create_instruction_few_shot'](caption),group)
    return [{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':prompt},
        {'role':'assistant','content':variants[episode['variant_id']]['bricks']}]


class ContextDataset:
    """CPU-only reader. iter_messages yields ready-to-tokenize conversations and loss metadata."""
    def __init__(self,directory=OUT):
        self.directory=Path(directory)
        self.structures={r['structure_id']:r for r in read_jsonl(self.directory/'structures.jsonl.gz')}
        self.variants={r['variant_id']:r for r in read_jsonl(self.directory/'variants.jsonl.gz')}

    def iter_messages(self,split):
        for episode in read_jsonl(self.directory/'episodes'/f'{split}.jsonl.gz'):
            yield {'messages':render_messages(episode,self.structures,self.variants),
                'metadata':episode,'loss_on':'final_assistant_only'}
