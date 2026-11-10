"""Checkpoint inference and result rendering for expanded ICL."""
import json
import textwrap
import context_data as data
import icl_experiment as engine
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=data.ROOT
OUT=ROOT/'output/icl_lora_train_expanded_v1'
BASE=ROOT/'temp/icl_lora_train_600_bf16_v1/brickgpt_merged_bf16'
INSTANCE=None



class EvalLLM(engine.LocalLLM):
    adapter_path = None

    def __init__(self,model_name,device):
        global INSTANCE
        self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
        base=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16,
            device_map={'':device},attn_implementation='sdpa',local_files_only=True)
        adapter = self.adapter_path
        if adapter is None:
            raise ValueError('An evaluation checkpoint is required')
        self.model=PeftModel.from_pretrained(base,adapter,local_files_only=True,is_trainable=False).eval()
        self.model.config.use_cache=True;self.model.generation_config.use_cache=True
        self.model.config.pad_token_id=self.tokenizer.pad_token_id
        self.kv_cache=self.kv_cache_saved=None
        self.input_ids_cache=self.input_ids_cache_saved=None
        INSTANCE=self

    def __call__(self,*args,**kwargs):
        with torch.autocast('cuda',dtype=torch.bfloat16):return super().__call__(*args,**kwargs)


def instruction(design,condition,task):
    if INSTANCE is not None:
        if condition=='before':INSTANCE.model.base_model.disable_adapter_layers()
        else:INSTANCE.model.base_model.enable_adapter_layers()
    return task['messages'][1]['content']


def messages(design,condition,task):return task['messages']




def postprocess(result,task):
    result['group']=task['group']
    result['structure_id']=task['structure_id']
    result['finished_with_new']=bool(not result['error'] and result['eos'] and result['geometry_valid'] and result['uses_new_brick'])
    result['stop_reason']='time_limit' if result['error']=='wallclock_run_limit' else 'rejection_limit' if result['error'] else 'eos' if result['eos'] else 'brick_limit'
    return result


def preview(result):
    group=result['group'];dest=OUT/group/'runs'/result['run_id']
    bs=engine.parse((dest/'bricks.txt').read_text(encoding='utf-8')).bricks
    new={tuple(sorted(x)) for x in data.GROUPS[group]}
    fig=plt.figure(figsize=(7,7));ax=fig.add_subplot(projection='3d')
    for b in bs:ax.bar3d(b.x,b.y,b.z,b.h,b.w,1,color='#ef8a35' if engine.canonical(b) in new else '#719ebd',edgecolor='#303840',linewidth=.3,shade=True)
    ax.set(xlim=(0,20),ylim=(0,20),zlim=(0,20),xlabel='X',ylabel='Y',zlabel='Z')
    ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-55)
    ax.set_title(f'{result["run_id"]}\nNew bricks={result["n_new_bricks"]}; EOS={result["eos"]}; connected={result["connected"]}')
    design=json.loads((OUT/group/'design.json').read_text(encoding='utf-8'))
    task=next(t for t in design['tasks'] if t['task_id']==result['task_id'])
    caption=textwrap.fill('Task prompt: '+task['caption'],width=82)
    context='Context structures: '+', '.join(x['structure_id'][:8] for x in task['context'])
    fig.text(.04,.025,caption+'\n'+context+'\nFull prompt: prompt_full.png (same directory)',fontsize=9,va='bottom')
    fig.tight_layout(rect=(0,.17,1,1));fig.savefig(dest/'preview.png',dpi=135);plt.close(fig)
    prompt=task['messages'][1]['content']
    lines=[]
    for line in prompt.splitlines():lines.extend(textwrap.wrap(line,105) or [''])
    pf=plt.figure(figsize=(12,max(5,len(lines)*.135)))
    pf.text(.025,.995,'\n'.join(lines),family='monospace',fontsize=8,va='top')
    pf.savefig(dest/'prompt_full.png',dpi=120,bbox_inches='tight');plt.close(pf)
    summarize()
