"""Shared ranking and serialization for expanded ICL data."""
from fractions import Fraction
import gzip
import hashlib
import json


def write_jsonl(path,rows):
    with gzip.open(path,'wt',encoding='utf-8',compresslevel=3) as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')


def seed(key):return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')


def rank_key(v):
    replaced=sum(len(o['original_steps']) for o in v['operations'])
    return (-Fraction(replaced,v['original_brick_count']),-replaced,v['structure_id'])


def short_key(v):return (len(v['bricks']),*rank_key(v))
