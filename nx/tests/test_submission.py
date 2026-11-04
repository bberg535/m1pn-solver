#!/usr/bin/env python3
"""Check generated Slurm commands and guards, without submitting anything."""
import json, os, pathlib, subprocess, tempfile, unittest
NX=pathlib.Path(__file__).resolve().parents[1]

class SubmissionTests(unittest.TestCase):
    def test_dry_run_and_resource_guards(self):
        with tempfile.TemporaryDirectory(dir=NX/'tmp') as tmp:
            base=pathlib.Path(tmp);campaign=base/'campaign';campaign.mkdir()
            (campaign/'tasks.json').write_text(json.dumps([{'id':k} for k in range(48)]))
            (campaign/'manifest.json').write_text(json.dumps({'threads':16,'plan':{'execution':{'max_concurrent_tasks':2}}}))
            fake=base/'bin';fake.mkdir();marker=base/'SUBMITTED'
            (fake/'sbatch').write_text('#!/bin/bash\ntouch "'+str(marker)+'"\nexit 99\n')
            (fake/'sbatch').chmod(0o755)
            env=dict(os.environ,PATH=str(fake)+os.pathsep+os.environ['PATH'])
            def invoke(*args):
                return subprocess.run(['bash',str(NX/'scripts/submit_nx.sh'),'--dry-run',*map(str,args)],capture_output=True,text=True,env=env)
            good=invoke('run',campaign,2)
            self.assertEqual(good.returncode,0,good.stderr)
            self.assertIn('--array=0-47%2',good.stdout)
            self.assertIn('--cpus-per-task=16',good.stdout)
            self.assertIn(str(NX/'batches/run.sbatch'),good.stdout)
            self.assertFalse(marker.exists(),'dry-run invoked sbatch')
            self.assertNotEqual(invoke('run',campaign,3).returncode,0)
            self.assertNotEqual(invoke('run',campaign,2,'48').returncode,0)
            collect=invoke('collect',campaign,12345)
            self.assertEqual(collect.returncode,0,collect.stderr)
            self.assertIn('--dependency=afterany:12345',collect.stdout)
            self.assertNotEqual(invoke('prepare','production',campaign).returncode,0)
            escaped=invoke('prepare','production',NX.parent/'tmp/forbidden_submit_test')
            self.assertNotEqual(escaped.returncode,0)
            self.assertFalse(marker.exists())

    def test_compute_script_rejects_login_execution(self):
        env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
        p=subprocess.run(['bash',str(NX/'scripts/build_nx.sh')],env=env,capture_output=True,text=True)
        self.assertEqual(p.returncode,2)
        self.assertIn('allocation',p.stderr)

    def test_mass_correction_wrapper_rejects_biharmonic_campaign(self):
        with tempfile.TemporaryDirectory(dir=NX/'tmp') as tmp:
            campaign=pathlib.Path(tmp)
            (campaign/'manifest.json').write_text(json.dumps({'plan':{'method':{
                'model':'r2','mass_mode':'lumped_afc','mass_corrections':1,
                'integrator':'ssprk3','stabilization':'biharmonic-corrected',
                'biharmonic_gamma':.004,'kv_gamma':0}}}))
            p=subprocess.run(['bash',str(NX/'scripts/submit_mass_correction.sh'),
                              '--dry-run','collect',str(campaign)],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('Expected standard B',p.stderr)

    def test_complete_launcher_dry_run_never_submits(self):
        with tempfile.TemporaryDirectory(dir=NX/'tmp') as tmp:
            base=pathlib.Path(tmp);fake=base/'bin';fake.mkdir();marker=base/'SUBMITTED'
            (fake/'sbatch').write_text('#!/bin/bash\ntouch "'+str(marker)+'"\nexit 99\n')
            (fake/'sbatch').chmod(0o755)
            env=dict(os.environ,PATH=str(fake)+os.pathsep+os.environ['PATH'])
            p=subprocess.run(['python3',str(NX/'scripts/start_mass_correction.py'),'--dry-run'],
                             capture_output=True,text=True,env=env)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(len(p.stdout.strip().splitlines()),3)
            self.assertIn('--array=0-47%2',p.stdout)
            self.assertIn('--dependency=afterok:PREPARE_JOB_ID',p.stdout)
            self.assertIn('afterany:ARRAY_JOB_ID',p.stdout)
            self.assertFalse(marker.exists(),'dry-run invoked sbatch')

if __name__=='__main__':unittest.main()
