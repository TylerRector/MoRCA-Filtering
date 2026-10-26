import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import platform
import random
import statistics
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from topology import sensitivity, returns

ROOT = Path(__file__).resolve().parent
CONDITIONS = ['independent', 'copied_correct', 'shared_error', 'paraphrase',
              'mutual_correction', 'common_prompt', 'chain', 'ring', 'aggregate_only']
POLICIES = ['none', 'morca', 'random_budget', 'lexical', 'embedding', 'loo',
            'no_H', 'no_P', 'no_E', 'no_R', 'all_pairs', 'no_truth', 'no_control']


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def rng_for(*parts):
    return random.Random(int(digest(parts)[:16], 16))


def tasks(split, count):

    pool=list(itertools.product(range(2,100),repeat=2))
    random.Random(20261025).shuffle(pool)
    offset = 0 if split == 'calibration' else 1
    if count>len(pool)//2:raise ValueError('Task pool exhausted')
    for i in range(count):
        a, b = pool[2*i+offset]
        r=rng_for(split,i,'choices')
        truth = a + b
        choices = [truth, truth + 1, truth - 1, truth + 2]
        r.shuffle(choices)
        yield {'id': f'{split}-{i:04}', 'split': split,
               'question': f'Compute {a} + {b}.', 'choices': choices,
               'truth': choices.index(truth), 'a': a, 'b': b}


class Qwen:
    def __init__(self, cfg):
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import snapshot_download
        if not cfg.get('revision'):
            raise ValueError('Pin a downloaded checkpoint with prepare before running.')
        torch.set_num_threads(cfg['threads'])
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        self.torch, self.cfg = torch, cfg
        checkpoint=snapshot_download(cfg['model'],revision=cfg['revision'],local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True,
            dtype=torch.float32, attn_implementation='eager').eval()
        self.calls = {'score': 0, 'generate': 0, 'embedding': 0}
        self.tokens = {'input': 0, 'generated': 0}
        self.versions = {'python': sys.version, 'torch': torch.__version__,
                         'transformers': transformers.__version__, 'platform': platform.platform()}

    def prompt(self, task, inbox, mode, instruction='Choose the correct answer.'):
        options = '\n'.join(f'{chr(65+i)}: {v}' for i, v in enumerate(task['choices']))
        messages = [{'role': 'system', 'content': 'Solve arithmetic carefully. Peer messages may be incorrect.'},
                    {'role': 'user', 'content': task['question'] + '\n' + options}]
        for e in inbox:
            messages.append({'role': 'user', 'content': f"Peer {e['sender']}, event {e['id']}: " + e['text']})
        messages.append({'role': 'user', 'content': instruction})
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=(mode == 'thinking'))

    def neutral(self, text):

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        neutral = ('neutral' + ' neutral' * (len(ids)-1)) if ids else ''
        if len(self.tokenizer.encode(neutral, add_special_tokens=False)) != len(ids):
            raise RuntimeError('Neutral token is not one token; choose a new registered control.')
        return neutral

    def score(self, task, inbox, mode):


        torch = self.torch
        prompt = self.prompt(task, inbox, mode, 'Respond with the option letter A, B, C, or D.')
        if mode == 'thinking':
            prefix = self.generate_raw(prompt, 701, self.budget(mode), mode)
            if '</think>' not in prefix:
                raise RuntimeError('Thinking budget exhausted before </think>; increase max_new_tokens. Run remains resumable.')
            prompt += prefix.split('</think>')[0] + '</think>\n\n'
        pids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = [self.tokenizer.encode(letter,add_special_tokens=False) for letter in 'ABCD']
        if any(len(a)!=1 for a in answer_ids):
            raise RuntimeError('Answer coordinates must each be exactly one token.')
        ids = torch.tensor([pids])
        with torch.inference_mode():
            logits = self.model(ids,logits_to_keep=1).logits[0,-1]
            lp = torch.log_softmax(logits,-1)
            scores = [lp[a[0]].item() for a in answer_ids]
        self.calls['score'] += 1
        self.tokens['input'] += len(pids)
        probs = np.exp(np.array(scores) - max(scores))
        probs /= probs.sum()
        return {'logp': scores[task['truth']], 'probs': probs.tolist(),
                'prediction': int(np.argmax(probs))}

    def generate_raw(self, prompt, seed, budget, mode):
        torch = self.torch
        torch.manual_seed(seed)
        inp = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        with torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=budget, do_sample=True,
                    temperature=0.6 if mode == 'thinking' else 0.7,
                    top_p=0.95 if mode == 'thinking' else 0.8, top_k=20,
                    pad_token_id=self.tokenizer.eos_token_id)
        new = out[0, inp.input_ids.shape[1]:]
        self.calls['generate'] += 1
        self.tokens['input'] += inp.input_ids.numel()
        self.tokens['generated'] += len(new)
        return self.tokenizer.decode(new, skip_special_tokens=False)

    def generate(self, task, inbox, mode, seed):
        text = self.generate_raw(self.prompt(task, inbox, mode,
            'Send a concise solution and the answer letter to a peer.'), seed,
            self.budget(mode), mode)
        if mode == 'thinking':
            if '</think>' not in text:
                raise RuntimeError('Incomplete thinking message; increase token budget.')
            text = text.split('</think>', 1)[1]
        return text.replace('<|im_end|>', '').strip()

    def budget(self,mode):
        return self.cfg.get('max_thinking_tokens',self.cfg['max_new_tokens']) if mode=='thinking' else self.cfg['max_new_tokens']

    def embedding(self, text):
        ids = self.tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids
        with self.torch.inference_mode():
            x = self.model.get_input_embeddings()(ids).mean(1)[0].numpy()
        self.calls['embedding'] += 1
        return x / max(np.linalg.norm(x), 1e-12)

    def token_count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def schedule(condition):
    if condition in ('independent', 'common_prompt'):
        return [(0, 2), (1, 2)]
    if condition == 'chain':
        return [(0, 1), (1, 2)]
    if condition == 'aggregate_only':
        return [(2, 0), (1, 2), (0, 1)]

    return [(0, 1), (1, 2), (2, 0), (0, 2)]


def network(model, task, condition, mode, seed, forced=None, blocked=None, decisions=None):
    events = []
    for i, (sender, receiver) in enumerate(schedule(condition)):
        incoming = [e for e in events if e['receiver'] == sender and e['id'] != blocked and e.get('accepted',True)]
        if condition in ('independent', 'common_prompt', 'aggregate_only'):
            incoming = []
        generation_seed = int(digest([task['id'], seed, i])[:8], 16)
        if condition == 'common_prompt':
            generation_seed = int(digest([task['id'], seed])[:8], 16)
        text = model.generate(task, incoming, mode, generation_seed)
        injected = False
        if condition in ('copied_correct', 'shared_error', 'paraphrase'):
            answer = task['truth'] if condition == 'copied_correct' else (task['truth']+1) % 4
            claim = f"The answer is {chr(65+answer)}: {task['choices'][answer]}."
            text = (f'I checked the result; {claim}' if condition == 'paraphrase' and i % 2 else claim)
            injected = True
        if condition == 'mutual_correction' and i == 0:
            text = f"My initial guess is {chr(65+(task['truth']+1)%4)}; please verify."
            injected = True
        if forced and i == forced[0]:
            text = forced[1]
        events.append({'id': i, 'sender': sender, 'receiver': receiver,
            'send_time': 2*i, 'receive_time': 2*i+1, 'text': text,
            'parents': [e['id'] for e in incoming], 'injected': injected,
            'generation_seed': generation_seed, 'protected': False})
        if decisions is not None:
            events[-1].update({k:decisions[i][k] for k in ('weight','coin','accepted')})
    return events


def ancestry(events, i, threshold=0.):
    result = {i}
    for p in events[i]['parents']:
        if events[p].get('uptake',0.)>=threshold and not events[i]['injected']:
            result |= ancestry(events, p, threshold)
    return result


def measure(model, task, events, condition, mode, seed, cfg):
    evidence = []
    for j in range(3):
        incoming = [e for e in events if e['receiver'] == j]
        d = len(incoming)
        if not d:
            continue
        subsets = {(), tuple(range(d))}
        subsets.update((i,) for i in range(d))
        subsets.update(tuple(k for k in range(d) if k != i) for i in range(d))
        subsets.update(itertools.combinations(range(d), 2))
        values = {}
        unmatched = {}
        expected_length=model.token_count(model.prompt(task,incoming,mode)) if isinstance(model,Qwen) else None
        for subset in sorted(subsets):
            inbox = [dict(e, text=e['text'] if i in subset else model.neutral(e['text']))
                     for i, e in enumerate(incoming)]
            if expected_length is not None and model.token_count(model.prompt(task,inbox,mode))!=expected_length:
                raise RuntimeError('Full chat-template token count changed under matched masking; control requires recalibration.')
            values[subset] = model.score(task, inbox, mode)

            unmatched[subset] = model.score(task,[e for i,e in enumerate(incoming) if i in subset],mode)
        full = values[tuple(range(d))]
        for i, e in enumerate(incoming):
            leave = values[tuple(k for k in range(d) if k != i)]
            e['uptake'] = min(1., float(np.linalg.norm(np.array(full['probs'])-leave['probs'])))
            linked, all_pairs, unmatched_pairs, pair_records = [], [], [], []
            for k, f in enumerate(incoming):
                if i == k:
                    continue
                pair = tuple(sorted((i, k)))
                delta = values[pair]['logp']-values[(i,)]['logp']-values[(k,)]['logp']+values[()]['logp']
                term = max(0., -delta)
                all_pairs.append(term)
                unmatched_delta = unmatched[pair]['logp']-unmatched[(i,)]['logp']-unmatched[(k,)]['logp']+unmatched[()]['logp']
                pair_records.append({'partner':f['id'],'redundancy':term,'unmatched_redundancy':max(0.,-unmatched_delta)})
                if ancestry(events, e['id']) & ancestry(events, f['id']):
                    linked.append(term)
                    unmatched_pairs.append(max(0.,-unmatched_delta))
            evidence.append({'event': e['id'], 'H': min(1., max(0., leave['logp']-full['logp'])/cfg['B']),
                'R': min(1., sum(linked)/(cfg['B_R']*max(1,len(linked)))),
                'R_all': min(1., sum(all_pairs)/(cfg['B_R']*max(1,len(all_pairs)))),
                'P': 0., 'E': 0., 'uptake': e['uptake'],
                'pairs':pair_records,
                'H_unmatched':min(1.,max(0.,unmatched[tuple(k for k in range(d) if k!=i)]['logp']-unmatched[tuple(range(d))]['logp'])/cfg['B']),
                'R_unmatched':min(1.,sum(unmatched_pairs)/(cfg['B_R']*max(1,len(unmatched_pairs)))),
                'values': {','.join(map(str,k)):v for k,v in values.items()}})

    for item in evidence:
        linked=[p for p in item['pairs'] if ancestry(events,item['event'],cfg['uptake_threshold'])
                & ancestry(events,p['partner'],cfg['uptake_threshold'])]
        item['R']=min(1.,sum(p['redundancy'] for p in linked)/(cfg['B_R']*max(1,len(linked))))
        item['R_unmatched']=min(1.,sum(p['unmatched_redundancy'] for p in linked)/(cfg['B_R']*max(1,len(linked))))
    paths = returns(events, cfg['uptake_threshold'])
    fidelities = {e['id']: [] for e in events}
    probe_records = []
    for path in paths:


        if any(events[i]['injected'] for i in path[1:]):
            continue
        first, last = events[path[0]], events[path[-1]]
        origin = first['sender']
        probe = 'Consider instead option ' + chr(65+(task['truth']+1)%4) + '.'
        variants = {}
        for name, payload in [('probe', probe), ('control', model.neutral(probe))]:
            for routed in (True, False):
                replay = network(model, task, condition, mode, seed,
                    forced=(first['id'], payload), blocked=None if routed else first['id'])

                inbox = [e for e in replay if e['receiver']==origin and e['id']<=last['id']]
                variants[name, routed] = np.array(model.score(task, inbox, mode)['probs'])
        d_c = (variants['probe',True]-variants['control',True])-(variants['probe',False]-variants['control',False])
        base = [e for e in events if e['receiver']==origin and e['id']<first['id']]
        d0 = np.array(model.score(task, base+[dict(first,text=probe)],mode)['probs'])-np.array(
            model.score(task,base+[dict(first,text=model.neutral(probe))],mode)['probs'])

        rp = network(model,task,condition,mode,seed,forced=(first['id'],probe))
        rc = network(model,task,condition,mode,seed,forced=(first['id'],model.neutral(probe)))
        hop_uptake = []
        for hop in path[1:]:
            a = model.score(task,[rp[hop]],mode)['probs']
            b = model.score(task,[rc[hop]],mode)['probs']
            hop_uptake.append(float(np.linalg.norm(np.array(a)-b)))
        eligible = np.linalg.norm(d0)>=cfg['uptake_threshold'] and all(x>=cfg['uptake_threshold'] for x in hop_uptake)
        fidelity = max(0., 1-float(np.linalg.norm(d_c-d0))/(float(np.linalg.norm(d0))+1e-9)) if eligible else 0.
        for i in path:
            fidelities[i].append(fidelity)
        probe_records.append({'path':path,'d0':d0.tolist(),'dC':d_c.tolist(),
                              'hop_uptake':hop_uptake,'eligible':bool(eligible),'fidelity':fidelity})
    for item in evidence:
        i = item['event']
        item['P'] = sensitivity(events, i)
        item['E'] = statistics.mean(fidelities[i]) if fidelities[i] else 0.
    return evidence, probe_records


def weights(evidence, policy, cfg):
    result = []
    for e in evidence:
        h,p,echo,r = (e[k] for k in ('H','P','E','R'))
        if policy=='no_H': h=0.
        if policy=='no_P': p=1.
        if policy=='no_E': echo=1.
        if policy=='no_R': r=1.
        if policy=='all_pairs': r=e.get('R_all',r)
        if policy=='no_control':h,r=e['H_unmatched'],e['R_unmatched']
        if policy=='no_truth': h=r=0.
        if policy=='loo': p=0.
        s = h+cfg['kappa']*p*echo*r
        result.append(1/(1+cfg['lambda']*s))
    return result


def route(model, events, evidence, policy, cfg, key):
    if policy=='none': ws=[1.]*len(events)
    elif policy=='random_budget':
        target=weights(evidence,'morca',cfg)
        counts=[model.token_count(e['text']) for e in events]
        w=sum(a*b for a,b in zip(target,counts))/max(1,sum(counts))
        ws=[w]*len(events)
    elif policy in ('lexical','embedding'):
        ws=[]
        for i,e in enumerate(events):
            previous=[f for f in events[:i] if f['receiver']==e['receiver']]
            if policy=='lexical':
                a=set(e['text'].lower().split())
                sims=[len(a & set(f['text'].lower().split()))/max(1,len(a | set(f['text'].lower().split()))) for f in previous]
            else:
                a=model.embedding(e['text'])
                sims=[float(a@model.embedding(f['text'])) for f in previous]
            ws.append(0. if sims and max(sims)>=0.9 else 1.)
    else: ws=weights(evidence,policy,cfg)
    routed=[]
    for e,w in zip(events,ws):
        coin=rng_for(*key,policy,e['id']).random()
        accepted=e['protected'] or coin<w
        routed.append(dict(e,weight=w,coin=coin,accepted=bool(accepted)))
    return routed


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, sort_keys=True)+'\n')
        f.flush()


def read_records(path,recover=False):
    if not path.exists():return []
    lines=path.read_text().splitlines(keepends=True)
    result=[]
    for i,line in enumerate(lines):
        try:result.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1 or line.endswith('\n'):raise
            print('Ignoring incomplete final record from interrupted write.',file=sys.stderr)
            if recover:
                path.with_suffix('.incomplete').write_text(line)
                path.write_text(''.join(lines[:i]))
    return result


def audit_final(model,task,events,mode,cfg):

    inbox=[e for e in events if e['receiver']==2]
    full=model.score(task,inbox,mode)
    labels=[]
    for e in inbox:
        control=[dict(f,text=model.neutral(f['text'])) if f['id']==e['id'] else f for f in inbox]
        leave=model.score(task,control,mode)
        marginal=full['logp']-leave['logp']
        labels.append({'event':e['id'],'helpful':marginal>cfg['uptake_threshold'],
                       'harmful':marginal < -cfg['uptake_threshold'],'marginal':marginal})
    return labels


def run(model,cfg,out):
    out.mkdir(parents=True,exist_ok=True)
    manifest={'config':cfg,'config_hash':digest(cfg),'versions':model.versions,
              'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}}
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:
        raise ValueError('Output manifest differs. Use a fresh output directory.')
    mp.write_text(json.dumps(manifest,indent=2))
    raw=out/'records.jsonl'
    existing=read_records(raw,recover=True)
    done={tuple(x['key']) for x in existing}
    for split in ('calibration','evaluation'):
        for task in tasks(split,split_count(cfg,split)):
            for mode,condition,seed in itertools.product(cfg['modes'],cfg.get('conditions',CONDITIONS),cfg['seeds']):
                key=(task['id'],mode,condition,seed)
                if key in done: continue
                before=copy.deepcopy(model.calls)
                tokens_before=copy.deepcopy(getattr(model,'tokens',{}))
                started=time.perf_counter()
                events=network(model,task,condition,mode,seed)
                if split=='calibration':
                    ev,probes=measure(model,task,events,condition,mode,seed,cfg)
                    ev.sort(key=lambda x:x['event'])
                    row={'key':key,'task':task,'events':events,'evidence':ev,'probes':probes}
                else:

                    calibration=[r for r in existing if r['task']['split']=='calibration'
                                 and r['key'][1:3]==[mode,condition]]
                    ev=[{k:statistics.mean(r['evidence'][i][k] for r in calibration)
                         for k in ('H','P','E','R','R_all','H_unmatched','R_unmatched')} for i in range(len(events))]
                    outcomes={}
                    for policy in cfg.get('policies',POLICIES):
                        policy_started=time.perf_counter()
                        policy_calls=copy.deepcopy(model.calls)
                        policy_tokens=copy.deepcopy(getattr(model,'tokens',{}))
                        decisions=route(model,events,ev,policy,cfg,key)


                        routed=network(model,task,condition,mode,seed,decisions=decisions)
                        inbox=[e for e in routed if e['receiver']==2 and e['accepted']]
                        outcome=model.score(task,inbox,mode)
                        counts=[model.token_count(e['text']) for e in routed]
                        outcome.update(accuracy=int(outcome['prediction']==task['truth']),
                            delivered_tokens=sum(n for n,e in zip(counts,routed) if e['accepted']),
                            expected_tokens=sum(n*e['weight'] for n,e in zip(counts,routed)),
                            suppressed=sum(not e['accepted'] for e in routed),
                            protected_drops=sum(e['protected'] and not e['accepted'] for e in routed),routing=routed)
                        outcome['inference_calls']={k:model.calls[k]-policy_calls[k] for k in policy_calls}
                        outcome['inference_tokens']={k:model.tokens[k]-policy_tokens[k] for k in policy_tokens}
                        outcome['seconds']=time.perf_counter()-policy_started
                        outcomes[policy]=outcome
                    labels=audit_final(model,task,events,mode,cfg)
                    for policy,v in outcomes.items():
                        routing={e['id']:e for e in v['routing']}
                        useful=[e for e in labels if e['helpful']]
                        v['helpful_edges']=len(useful)
                        v['helpful_suppressed']=sum(not routing[e['event']]['accepted'] for e in useful)
                    row={'key':key,'task':task,'events':events,'evidence':ev,'outcomes':outcomes,'posthoc_labels':labels}
                row.update(seconds=time.perf_counter()-started,
                           calls={k:model.calls[k]-before[k] for k in before},
                           inference_tokens={k:model.tokens[k]-tokens_before[k] for k in tokens_before})
                append(raw,row); existing.append(json.loads(json.dumps(row))); done.add(key)
                print(f'{key}: {row["seconds"]:.1f}s',flush=True)
    analyze(out,cfg)


def analyze(out,cfg):
    rows=read_records(out/'records.jsonl')
    flat=[]
    for row in rows:
        for policy,v in row.get('outcomes',{}).items():
            flat.append(dict(task=row['key'][0],mode=row['key'][1],condition=row['key'][2],seed=row['key'][3],
                policy=policy,**{k:v[k] for k in ('accuracy','logp','delivered_tokens','expected_tokens','suppressed','protected_drops','helpful_edges','helpful_suppressed')}))
    if not flat: raise ValueError('No evaluation outcomes available.')
    with (out/'outcomes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    report=['# MoRCA results','', 'Synthetic arithmetic; prospective routed rollouts with calibrated event-position gates. Incomplete proposal coverage; see README.',
            'Intervals cluster by task, preserving paired conditions and routing seeds. Modes analyzed separately.','',
            '| Mode | Policy | Accuracy | Paired accuracy change vs none (95% CI) | Mean reference log score | Mean tokens |',
            '|---|---|---:|---:|---:|---:|']
    summaries=[]
    for mode,policy in itertools.product(cfg['modes'],cfg.get('policies',POLICIES)):
        group=[r for r in flat if r['mode']==mode and r['policy']==policy]
        if not group:continue
        baseline={(r['task'],r['condition'],r['seed']):r for r in flat if r['mode']==mode and r['policy']=='none'}
        clusters={}
        for r in group:
            b=baseline[r['task'],r['condition'],r['seed']]
            clusters.setdefault(r['task'],[]).append(r['accuracy']-b['accuracy'])
        diffs=[statistics.mean(x) for x in clusters.values()]
        rng=np.random.default_rng(2026)
        boot=[float(np.mean(rng.choice(diffs,len(diffs),replace=True))) for _ in range(cfg['bootstrap_samples'])]
        lo,hi=np.quantile(boot,[.025,.975])
        summary={'mode':mode,'policy':policy,'accuracy':statistics.mean(r['accuracy'] for r in group),
                 'paired_accuracy_delta':statistics.mean(diffs),'ci95':[lo,hi],
                 'reference_log_score':statistics.mean(r['logp'] for r in group),
                 'mean_tokens':statistics.mean(r['delivered_tokens'] for r in group),
                 'tasks':len(clusters),'protected_drops':sum(r['protected_drops'] for r in group)}
        summary['false_suppression_rate']=sum(r['helpful_suppressed'] for r in group)/max(1,sum(r['helpful_edges'] for r in group))
        summary['helpful_edge_denominator']=sum(r['helpful_edges'] for r in group)
        summaries.append(summary)
        report.append(f'| {mode} | {policy} | {summary["accuracy"]:.3f} | {summary["paired_accuracy_delta"]:+.3f} [{lo:+.3f}, {hi:+.3f}] | {summary["reference_log_score"]:.3f} | {summary["mean_tokens"]:.1f} |')
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    diagnostics={'calibration_return_lengths':{},'harm_detection_auroc':{},'calibration_factor_means':{},
                 'note':'Harm AUROC uses posthoc held-out marginal labels; this is not validated redundancy AUROC.'}
    for row in rows:
        for probe in row.get('probes',[]):
            length=str(len(probe['path']))
            diagnostics['calibration_return_lengths'][length]=diagnostics['calibration_return_lengths'].get(length,0)+1
    for mode in cfg['modes']:
        diagnostics['calibration_factor_means'][mode]={}
        for condition in cfg.get('conditions',CONDITIONS):
            evidence=[e for row in rows if row['task']['split']=='calibration' and row['key'][1:3]==[mode,condition]
                      for e in row['evidence']]
            if evidence:
                diagnostics['calibration_factor_means'][mode][condition]={k:statistics.mean(e[k] for e in evidence)
                                                                         for k in ('H','P','E','R')}
        pairs=[]
        for row in rows:
            if row['key'][1]!=mode or 'outcomes' not in row:continue
            routing={e['id']:e for e in row['outcomes']['morca']['routing']}
            pairs += [(int(l['harmful']),1-routing[l['event']]['weight']) for l in row['posthoc_labels']]
        positive=[s for y,s in pairs if y];negative=[s for y,s in pairs if not y]
        diagnostics['harm_detection_auroc'][mode]=(
            sum((p>n)+.5*(p==n) for p in positive for n in negative)/(len(positive)*len(negative))
            if positive and negative else None)
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    (out/'analysis.md').write_text('\n'.join(report)+'\n')
    if cfg.get('primary_comparison'):
        from scipy.stats import binomtest
        spec=cfg['primary_comparison']
        paired=[r for r in rows if 'outcomes' in r and r['key'][1]==spec['mode'] and r['key'][2]==spec['condition']]
        wins=sum(r['outcomes'][spec['policy']]['accuracy']>r['outcomes'][spec['baseline']]['accuracy'] for r in paired)
        losses=sum(r['outcomes'][spec['policy']]['accuracy']<r['outcomes'][spec['baseline']]['accuracy'] for r in paired)
        complete=len(paired)==split_count(cfg,'evaluation') and len({r['key'][0] for r in paired})==len(paired)
        pvalue=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.
        delta=(wins-losses)/len(paired) if paired else None
        primary={'specification':spec,'n_tasks':len(paired),'complete':complete,'wins':wins,'losses':losses,
                 'ties':len(paired)-wins-losses,'accuracy_delta':delta,'two_sided_exact_mcnemar_p':pvalue,
                 'positive_primary_result':bool(complete and delta>0 and pvalue<spec['alpha']),
                 'note':'One prespecified comparison. Partial-run p-values are descriptive only; do not stop early for significance. Other comparisons are exploratory.'}
        (out/'primary-comparison.json').write_text(json.dumps(primary,indent=2))
        with (out/'analysis.md').open('a') as f:
            f.write(f'\nPrimary comparison: {spec["policy"]} vs {spec["baseline"]} in {spec["condition"]}. '
                    f'n={len(paired)}, wins={wins}, losses={losses}, exact two-sided p={pvalue:.6g}, complete={complete}.\n')


def benchmark(model,cfg,out):
    task=next(tasks('calibration',1))
    timings={}
    for mode in cfg['modes']:
        samples=[]
        for i in range(3):
            t=time.perf_counter()
            gen_before=model.tokens['generated']
            error=None
            try:
                peers=[{'sender':j,'id':j,'text':'Check the arithmetic and verify option A. '*4} for j in range(i)]
                text=model.generate(task,peers,mode,100+i)
            except RuntimeError as e:
                text=''
                error=str(e)
            s=time.perf_counter()
            raw_generation_tokens=model.tokens['generated']-gen_before
            try:
                model.score(task,peers,mode)
                score=time.perf_counter()-s
            except RuntimeError as e:
                score=None
                print(str(e),flush=True)
            samples.append({'generate_seconds':s-t,'score_seconds':score,
                            'generated_tokens':model.token_count(text),
                            'raw_generation_tokens':raw_generation_tokens,
                            'score_reasoning_tokens':model.tokens['generated']-gen_before-raw_generation_tokens,
                            'error':error})

            out.write_text(json.dumps({'kind':'timing_only_not_experiment',
                'config':cfg,'timings':dict(timings,**{mode:samples})},indent=2))
            print(mode,samples[-1],flush=True)
        timings[mode]=samples
    data={'kind':'timing_only_not_experiment','config':cfg,'timings':timings,
          'versions':model.versions,'calls':model.calls,'tokens':model.tokens}
    out.write_text(json.dumps(data,indent=2))
    print(json.dumps(data,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','benchmark','run','analyze','export'])
    p.add_argument('--config',type=Path,default=ROOT/'config.json')
    p.add_argument('--out',type=Path,default=ROOT/'runs'/'pilot')
    p.add_argument('--approve-run',action='store_true',help='Explicit full experiment opt-in')
    args=p.parse_args();cfg=json.loads(args.config.read_text())
    validate_config(cfg)
    if args.command=='prepare':
        from huggingface_hub import snapshot_download
        path=Path(snapshot_download(cfg['model'],revision=cfg['revision'],allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model']))
        cfg['revision']=path.name
        args.config.write_text(json.dumps(cfg,indent=2)+'\n')
        print('Pinned checkpoint',path.name);return
    if args.command=='analyze':analyze(args.out,cfg);return
    if args.command=='export':
        dest=ROOT/'morca-export.zip'
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
            for f in ROOT.rglob('*'):
                if f.is_file() and f.suffix!='.zip' and not {'__pycache__','.git','.lake','.venv'}.intersection(f.parts):
                    z.write(f,f.relative_to(ROOT))
        print(dest);return
    if args.command=='run' and not args.approve_run:
        p.error('Full experiment requires --approve-run; benchmark is a separate command.')
    model=Qwen(cfg)
    if args.command=='benchmark':benchmark(model,cfg,ROOT/'benchmark.json')
    else:run(model,cfg,args.out)


def validate_config(cfg):
    for k in ('lambda','kappa','B','B_R','uptake_threshold'):
        if not math.isfinite(cfg[k]) or cfg[k]<0 or (k in ('B','B_R') and cfg[k]==0):
            raise ValueError(f'Invalid {k}')
    for k in ('tasks_per_split','threads','max_new_tokens','bootstrap_samples'):
        if not isinstance(cfg[k],int) or cfg[k]<1:raise ValueError(f'Invalid {k}')
    if 'max_thinking_tokens' in cfg and (not isinstance(cfg['max_thinking_tokens'],int) or cfg['max_thinking_tokens']<1):
        raise ValueError('Invalid max_thinking_tokens')
    if not cfg['seeds'] or len(set(cfg['seeds']))!=len(cfg['seeds']):raise ValueError('Seeds must be unique and nonempty')
    if not cfg['modes'] or any(x not in ('thinking','nonthinking') for x in cfg['modes']):raise ValueError('Invalid modes')
    for split in ('calibration','evaluation'):
        n=split_count(cfg,split)
        if not isinstance(n,int) or not 1<=n<=4802:raise ValueError('Invalid split size')
    for key,allowed in [('conditions',CONDITIONS),('policies',POLICIES)]:
        values=cfg.get(key,allowed)
        if not values or len(values)!=len(set(values)) or any(x not in allowed for x in values):raise ValueError(f'Invalid {key}')
    if not {'none','morca'}.issubset(cfg.get('policies',POLICIES)):raise ValueError('Analysis requires none and morca policies')
    if cfg.get('primary_comparison'):
        s=cfg['primary_comparison']
        if len(cfg['seeds'])!=1 or s['condition'] not in cfg.get('conditions',CONDITIONS) or s['mode'] not in cfg['modes']:
            raise ValueError('Exact primary comparison requires one seed and a selected mode/condition')
        if not 0<s['alpha']<1 or any(s[k] not in cfg.get('policies',POLICIES) for k in ('policy','baseline')):
            raise ValueError('Invalid primary specification')


def split_count(cfg,split):
    return cfg.get(split+'_tasks',cfg['tasks_per_split'])


if __name__=='__main__':main()
