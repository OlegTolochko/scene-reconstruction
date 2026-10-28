"""Resume the frozen five-scene benchmark; reuse only completed inference runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import socket

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/3dfront_benchmark'

def execute(command, log, cwd=ROOT, environment=None):
    print('Running: '+' '.join(map(str,command)),flush=True)
    with log.open('w') as handle:
        result=subprocess.run(list(map(str,command)),cwd=cwd,env=environment,stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Return code {result.returncode}: {log}')

def main(start=1, export=True, end=5):
    if not 1 <= start <= end <= 5:
        raise ValueError('Expected 1 <= start <= end <= 5')
    print(f'Benchmark batch starts on {socket.gethostname()}; scenes {start}–{end}',flush=True)
    protocol=json.loads((ROOT/'data/3dfront/benchmark/protocol.json').read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for record in protocol['scenes'][start-1:end]:
        sid=record['id']
        result=OUT/sid;result.mkdir(parents=True,exist_ok=True)
        output=Path(record['output'])
        progress_path=result/'progress.json'
        history=json.loads(progress_path.read_text()) if progress_path.exists() else {'scene':record,'events':[]}
        def event(stage,status,detail=''):
            history['events'].append({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':stage,'status':status,'detail':detail})
            progress_path.write_text(json.dumps(history,indent=2)+'\n')
            print(f'{sid}: {stage}: {status} {detail}',flush=True)
        try:
            manifest_path=output/'run_manifest.json'
            manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            complete=manifest.get('inference_completed') or manifest.get('status')=='complete'
            if not (complete and (output/'mesh.ply').exists()):
                event('inference','running')
                execute([sys.executable,'-m','experiments.3dfront_validation.inference.run_genrecon','--data',record['input'],
                    '--output',output,'--inference-only'],result/'launcher.log')
                event('inference','complete')
            else:
                event('inference','reused','Verified successful run manifest and raw mesh')
            preview=output/('scene_preview_reference_frame.glb' if sid=='scene_01' else 'preview/scene.glb')
            if export and not preview.exists():
                event('preview','running')
                env=os.environ.copy()
                env.update({'CUDA_HOME':sys.prefix,'PATH':str(Path(sys.prefix)/'bin')+':'+env['PATH'],
                    'CPATH':str(Path(sys.prefix)/'targets/x86_64-linux/include'),
                    'PYTHONUNBUFFERED':'1','PYTORCH_NO_CUDA_MEMORY_CACHING':'1'})
                command=[sys.executable,'chunked_to_glb.py','--inputs',output/'to_glb_inputs.pt',
                    '--chunk_inputs',output/'chunk_inputs.pt','--output_dir',output/'preview',
                    '--skip_remesh','--no_chunk_cache','--simplify_threshold','250000','--texture_size','2048']
                try:
                    execute(command,result/'preview_export.log',ROOT/'papers/GenRecon',env)
                    updated=json.loads(manifest_path.read_text())
                    updated.update({'status':'complete','preview_export_completed':True,
                        'preview_path':str(preview),'preview_export_command':list(map(str,command)),
                        'preview_export_options':{'skip_remesh':True,'simplify_threshold_per_chunk':250000,'texture_size':2048}})
                    manifest_path.write_text(json.dumps(updated,indent=2)+'\n')
                    event('preview','complete')
                except Exception as error:
                    event('preview','failed',str(error))
            event('evaluation','running')
            command=[sys.executable,'-m','experiments.3dfront_validation.evaluation.evaluate_scene','--data',record['data'],
                '--inputs',record['input'],'--prediction',output/'mesh.ply','--output',result,
                '--samples',protocol['surface_samples_per_mesh'],'--grid',protocol['ray_grid']]
            if preview.exists():command.extend(['--preview',preview])
            env=os.environ.copy();env.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
            execute(command,result/'evaluation.log',environment=env)
            event('evaluation','complete')
        except Exception as error:
            event('scene','failed',str(error))
    execute([sys.executable,'-m','experiments.3dfront_validation.evaluation.summarize_benchmark'],OUT/'summary.log')
    summary=json.loads((OUT/'summary.json').read_text())
    selected_ids={r['id'] for r in protocol['scenes'][start-1:end]}
    completed_ids={r['scene'] for r in summary['scenes'] if r['camera_gate_passed']}
    if not selected_ids <= completed_ids:
        raise RuntimeError('Requested scene batch incomplete; inspect per-scene progress and summary.json')
    if export:
        for sid in selected_ids:
            metrics=json.loads((OUT/sid/'metrics.json').read_text())
            if metrics['preview_comparison'] is None:
                raise RuntimeError(f'{sid}: requested preview comparison is missing')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',type=int,default=1)
    parser.add_argument('--end',type=int,default=5)
    parser.add_argument('--no-export',action='store_true')
    args=parser.parse_args()
    main(args.start,not args.no_export,args.end)
