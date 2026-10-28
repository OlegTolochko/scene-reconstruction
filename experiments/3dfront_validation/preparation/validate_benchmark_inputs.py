"""Check frozen source hashes, camera-depth agreement, and held-out isolation."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import trimesh
from ..evaluation.evaluate_scene import Y_TO_Z, make_query, raycast, save_json
from .validate_room import read_depth, valid_depth

ROOT=Path(__file__).resolve().parents[3]

def main():
    protocol=json.loads((ROOT/'data/3dfront/benchmark/protocol.json').read_text())
    out=ROOT/'outputs/3dfront_benchmark';out.mkdir(parents=True,exist_ok=True)
    reports=[]
    figure,axes=plt.subplots(5,11,figsize=(22,10))
    for row,record in enumerate(protocol['scenes']):
        data=Path(record['data']);inputs=Path(record['input'])
        manifest=json.loads((data/'manifest.json').read_text())
        for item in manifest['files']:
            if hashlib.sha256((data/item['path']).read_bytes()).hexdigest()!=item['sha256']:
                raise RuntimeError('Source hash mismatch')
        selected=json.loads((inputs/'input_manifest.json').read_text())
        ids={v['index'] for v in selected['views']}
        if {p.stem for p in (inputs/'iphone/rgb').glob('*.png')}!=ids:
            raise RuntimeError('Conditioning directory differs from frozen input IDs')
        render=data/f"3D-FRONT-TEST-RENDER/{record['room']}"
        meta=json.loads((render/'meta.json').read_text())
        mesh=trimesh.load_scene(data/f"3D-FRONT-TEST-SCENE/{record['room']}.glb",process=False).to_mesh()
        mesh.apply_transform(Y_TO_Z);query=make_query(mesh)
        checks=[]
        for col,camera in enumerate(meta['locations']):
            index=camera['index'];depth=read_depth(render/f'depth_{index}.exr')
            h,w=depth.shape
            xx,yy=np.meshgrid(np.linspace(0,w-1,64).round().astype(int),np.linspace(0,h-1,64).round().astype(int))
            target=depth[yy.ravel(),xx.ravel()]
            pred=raycast(query,np.asarray(camera['transform_matrix']),meta['camera_angle_x'],xx.ravel(),yy.ravel(),w,h)
            both=valid_depth(target)&np.isfinite(pred);union=valid_depth(target)|np.isfinite(pred)
            iou=both.sum()/max(1,union.sum());rel=np.median(np.abs(pred[both]-target[both])/target[both]) if both.any() else np.inf
            passed=both.sum()>=20 and iou>=.95 and rel<.01
            checks.append({'index':index,'paired_pixels':int(both.sum()),'iou':float(iou),
                           'median_relative_error':float(rel),'passed':bool(passed)})
            ax=axes[row,col];ax.imshow(Image.open(render/f'render_{index}.webp'))
            ax.set_title(f"{record['id']} / {index}\n{'INPUT' if index in ids else 'HELD OUT'}",fontsize=8,
                         color='black' if index in ids else 'darkred');ax.axis('off')
        reports.append({'scene':record,'checks':checks,'input_ids':sorted(ids),
            'heldout_ids':[c['index'] for c in meta['locations'] if c['index'] not in ids],
            'passed':all(c['passed'] for c in checks)})
        print(record['id'],reports[-1]['passed'],flush=True)
    figure.tight_layout();figure.savefig(out/'frozen_camera_split.png',dpi=130);plt.close(figure)
    save_json(out/'input_validation.json',reports)
    if not all(r['passed'] for r in reports):raise RuntimeError('Input gate failed')
    # Portable tracked protocol omits machine-specific artifact paths.
    portable={k:v for k,v in protocol.items() if k!='scenes'}
    portable['scenes']=[{'id':r['id'],'room':r['room']} for r in protocol['scenes']]
    save_json(ROOT/'experiments/3dfront_validation/configs/BENCHMARK_PROTOCOL.json',portable)

if __name__=='__main__':main()
