"""BF16 LoRA setup, loss, and checkpoint utilities."""
import gc
import json
import os
import sys
from datetime import datetime,timezone
import io_helpers as setup
sys.path.insert(0,str(setup.ROOT/'BrickGPT/src'))
os.environ.setdefault('HF_HUB_CACHE',str(setup.ROOT/'temp/hf-cache/hub'))
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM,AutoTokenizer,get_cosine_schedule_with_warmup,set_seed
from peft import PeftModel,LoraConfig,get_peft_model
from brickgpt_stage1 import resolve_base_model

OUT=setup.ROOT/'output/icl_lora_train_expanded_v1'
MERGED=setup.ROOT/'temp/icl_lora_train_600_bf16_v1/brickgpt_merged_bf16'


def log_event(event,**fields):
    with (OUT/'experiment_events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'time_utc':datetime.now(timezone.utc).isoformat(),
            'pid':os.getpid(),'event':event,**fields})+'\n')


def save_status(state,**fields):
    row={'state':state,'updated_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),**fields}
    path=OUT/'status.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(row,indent=2),encoding='utf-8');temp.replace(path)


def prepare_base():
    if (MERGED/'config.json').exists() and list(MERGED.glob('*.safetensors')):return
    save_status('preparing_base')
    base=AutoModelForCausalLM.from_pretrained(resolve_base_model(None),torch_dtype=torch.float32,
        low_cpu_mem_usage=True,local_files_only=True,device_map={'':'cpu'})
    base=PeftModel.from_pretrained(base,setup.ROOT/'temp/models/BrickGPT',local_files_only=True)
    base=base.merge_and_unload(safe_merge=True).to(torch.bfloat16)
    MERGED.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(setup.ROOT/'temp/models/BrickGPT',local_files_only=True)
    assert tokenizer.pad_token_id!=tokenizer.eos_token_id
    base.config.pad_token_id=tokenizer.pad_token_id;base.config.eos_token_id=tokenizer.eos_token_id
    base.generation_config.pad_token_id=tokenizer.pad_token_id;base.generation_config.eos_token_id=tokenizer.eos_token_id
    base.save_pretrained(MERGED,safe_serialization=True);tokenizer.save_pretrained(MERGED)
    del base;gc.collect()


def load_model(adapter=None):
    base=AutoModelForCausalLM.from_pretrained(MERGED,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',device_map={'':'cuda'},local_files_only=True)
    if adapter:model=PeftModel.from_pretrained(base,adapter,is_trainable=True,local_files_only=True)
    else:model=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=16,lora_alpha=32,lora_dropout=.05,
        target_modules=['q_proj','v_proj'],bias='none'))
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    assert all(p.dtype==torch.bfloat16 for n,p in model.named_parameters() if not p.requires_grad)
    assert all(p.dtype==torch.float32 for p in model.parameters() if p.requires_grad)
    return model


def answer_loss(head,hidden,ids,start,chunk=128,use_checkpoint=True):
    states=hidden[:,start-1:-1,:].reshape(-1,hidden.shape[-1]);targets=ids[:,start:].reshape(-1)
    assert len(states)==len(targets)>0
    def project(h,y):return F.cross_entropy(head(h).float(),y,reduction='sum')
    losses=[]
    for i in range(0,len(targets),chunk):
        h=states[i:i+chunk];y=targets[i:i+chunk]
        losses.append(checkpoint(project,h,y,use_reentrant=False) if use_checkpoint else project(h,y))
    return torch.stack(losses).sum()


def loss(model,item,training=True):
    ids=torch.tensor([item['input_ids']],device='cuda',dtype=torch.long)
    core=model.get_base_model()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        hidden=core.model(input_ids=ids,use_cache=False).last_hidden_state
        value=answer_loss(core.lm_head,hidden,ids,item['answer_start'],use_checkpoint=training)
    return value


def update(model,optimizer,items):
    model.train();optimizer.zero_grad(set_to_none=True)
    denominator=sum(x['answer_tokens'] for x in items);value=0.
    for item in items:
        numerator=loss(model,item)
        assert torch.isfinite(numerator),'nonfinite loss'
        (numerator/denominator).backward();value+=float(numerator.detach())
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step()
    return value/denominator,float(norm)


@torch.no_grad()
def evaluate(model,items):
    model.eval();total=0.;tokens=0;start=time.perf_counter()
    for item in items:total+=float(loss(model,item,training=False));tokens+=item['answer_tokens']
    torch.cuda.synchronize()
    return total/tokens,time.perf_counter()-start


@torch.no_grad()




def save_checkpoint(model,optimizer,scheduler,step,epoch,cursor,best_loss,validation_loss=None):
    dest=OUT/'checkpoints'/f'step_{step:04d}';dest.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(dest);AutoTokenizer.from_pretrained(MERGED,local_files_only=True).save_pretrained(dest)
    state={'step':step,'epoch':epoch,'cursor':cursor,'best_loss':best_loss,'validation_loss':validation_loss,
        'validation_sample_ids':getattr(model,'validation_sample_ids',[]),
        'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
    torch.save(state,dest/'training_state.pt')
    setup.save(OUT/'latest_checkpoint.json',{'path':str(dest),'step':step,'epoch':epoch,'cursor':cursor})
    if validation_loss is not None and validation_loss<=best_loss:
        setup.save(OUT/'best_checkpoint.json',{'path':str(dest),'step':step,'validation_loss':validation_loss,
            'validation_samples':len(getattr(model,'validation_sample_ids',[]))})
    log_event('checkpoint_saved',step=step,epoch=epoch,cursor=cursor,path=str(dest),validation_loss=validation_loss)
    return dest
