import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experiment as ex
from topology import beta1, sensitivity, returns


class FakeModel:

    versions={'fixture':True}
    def __init__(self):self.calls={'score':0,'generate':0,'embedding':0}
    def generate(self,task,inbox,mode,seed):
        self.calls['generate']+=1
        return 'A ' + ' '.join(e['text'] for e in inbox)[-100:]
    def score(self,task,inbox,mode):
        self.calls['score']+=1
        count=sum(e['text'].count('A') for e in inbox)
        a=np.array([min(.9,.25+count*.1),.25,.25,.25]);a/=a.sum()
        return {'logp':float(np.log(a[task['truth']])),'probs':a.tolist(),'prediction':int(a.argmax())}
    def neutral(self,text):return 'neutral '*len(text.split())
    def token_count(self,text):return len(text.split())
    def embedding(self,text):return np.array([1.,0.])


class MathTests(unittest.TestCase):
    def test_path_homology(self):
        self.assertEqual(beta1([]),0)
        self.assertEqual(beta1([(0,1),(1,2)]),0)
        self.assertEqual(beta1([(0,1),(1,2),(2,0)]),1)
        self.assertEqual(beta1([(0,1),(1,2),(0,2)]),0)
        self.assertEqual(beta1([(0,1),(1,0)]),0)
        self.assertEqual(beta1([(i,j) for i in range(3) for j in range(3) if i!=j]),0)
    def test_event_deletion_reaggregates(self):
        es=[{'sender':a,'receiver':b,'uptake':.6} for a,b in [(0,1),(1,2),(2,0)]]
        self.assertAlmostEqual(sensitivity(es,0),.6)
        es.append(copy.deepcopy(es[0]))
        self.assertEqual(sensitivity(es,0),0)
    def test_time_order(self):
        es=[{'sender':a,'receiver':b,'uptake':1.,'send_time':2*i,'receive_time':2*i+1}
            for i,(a,b) in enumerate([(0,1),(1,2),(2,0)])]
        self.assertEqual(returns(es,.01),[[0,1,2]])
        es.reverse()
        self.assertEqual(returns(es,.01),[])
    def test_gates(self):
        cfg={'lambda':3.,'kappa':1.}
        self.assertEqual(ex.weights([dict(H=0,P=1,E=1,R=0)],'morca',cfg),[1.])
        w=ex.weights([dict(H=.5,P=1,E=1,R=1)],'morca',cfg)[0]
        self.assertAlmostEqual(w,1/5.5)
        self.assertLessEqual(w**3,1/8)
    def test_split_disjoint(self):
        a={(t['a'],t['b']) for t in ex.tasks('calibration',12)}
        b={(t['a'],t['b']) for t in ex.tasks('evaluation',12)}
        self.assertFalse(a & b)
    def test_protected_route(self):
        e={'id':0,'sender':0,'receiver':1,'text':'A','protected':True}
        routed=ex.route(FakeModel(),[e],[dict(H=1,P=1,E=1,R=1)],'lexical',{'lambda':1,'kappa':1},('test',))
        self.assertTrue(routed[0]['accepted'])

        duplicate=dict(e,id=1)
        routed=ex.route(FakeModel(),[e,duplicate],[{},{}],'lexical',{},('test',))
        self.assertEqual(routed[1]['weight'],0)
        self.assertTrue(routed[1]['accepted'])
    def test_online_drop_changes_parents(self):
        task=next(ex.tasks('calibration',1))
        decisions=[{'weight':0.,'coin':.5,'accepted':False} for _ in ex.schedule('ring')]
        events=ex.network(FakeModel(),task,'ring','nonthinking',17,decisions=decisions)
        self.assertTrue(all(not e['parents'] for e in events))
    def test_no_config_nan(self):
        cfg=json.loads((ex.ROOT/'config.json').read_text());cfg['lambda']=float('nan')
        with self.assertRaises(ValueError):ex.validate_config(cfg)
    def test_partial_record_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'records.jsonl';p.write_text('{"ok":1}\n{"cut":')
            self.assertEqual(ex.read_records(p,recover=True),[{'ok':1}])
            self.assertEqual(p.read_text(),'{"ok":1}\n')
            self.assertTrue(p.with_suffix('.incomplete').exists())


class IntegrationTests(unittest.TestCase):
    def test_reduced_design_and_exact_primary(self):
        cfg=json.loads((ex.ROOT/'config.json').read_text())
        cfg.update(calibration_tasks=1,evaluation_tasks=6,bootstrap_samples=20)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            ex.run(FakeModel(),cfg,out)
            rows=ex.read_records(out/'records.jsonl')
            self.assertEqual(len(rows),28)
            self.assertEqual(sum('outcomes' in r for r in rows),24)
            self.assertTrue(all(set(r['outcomes'])=={'none','morca','random_budget'} for r in rows if 'outcomes' in r))
            primary=json.loads((out/'primary-comparison.json').read_text())
            self.assertTrue(primary['complete'])
            self.assertEqual(primary['n_tasks'],6)

            for row in rows:
                if 'outcomes' in row and row['key'][2]=='mutual_correction':
                    row['outcomes']['morca']['accuracy']=1
                    row['outcomes']['none']['accuracy']=0
            (out/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            ex.analyze(out,cfg)
            primary=json.loads((out/'primary-comparison.json').read_text())
            self.assertAlmostEqual(primary['two_sided_exact_mcnemar_p'],.03125)
            self.assertTrue(primary['positive_primary_result'])
            cfg['evaluation_tasks']=64
            ex.analyze(out,cfg)
            self.assertFalse(json.loads((out/'primary-comparison.json').read_text())['positive_primary_result'])

    def test_fixture_pipeline_export_analysis_resume(self):
        cfg=json.loads((ex.ROOT/'config.json').read_text())
        cfg.update(tasks_per_split=1,calibration_tasks=1,evaluation_tasks=1,seeds=[17],modes=['nonthinking'],bootstrap_samples=20,conditions=ex.CONDITIONS,policies=ex.POLICIES)
        cfg.pop('primary_comparison',None)
        m=FakeModel()
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            ex.run(m,cfg,out)
            rows=[json.loads(x) for x in (out/'records.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),18)
            self.assertTrue((out/'analysis.md').exists())
            self.assertEqual(len(json.loads((out/'summary.json').read_text())),len(ex.POLICIES))
            before=copy.deepcopy(m.calls)
            ex.run(m,cfg,out)
            self.assertEqual(m.calls,before)
            for row in rows:
                for v in row.get('outcomes',{}).values():
                    self.assertEqual(v['protected_drops'],0)
            cfg['lambda']=99
            with self.assertRaises(ValueError):ex.run(m,cfg,out)


if __name__=='__main__':unittest.main()
