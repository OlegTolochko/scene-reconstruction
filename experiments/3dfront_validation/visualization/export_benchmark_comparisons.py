"""Create side-by-side inspection GLBs for completed benchmark evaluations."""
import json
from pathlib import Path
import shutil
from .export_comparison import main as export

ROOT=Path(__file__).resolve().parents[3]

def main():
    protocol=json.loads((ROOT/'data/3dfront/benchmark/protocol.json').read_text())
    for scene in protocol['scenes']:
        result=ROOT/'outputs/3dfront_benchmark'/scene['id']
        destination=result/'comparison_textured.glb'
        if not (result/'metrics.json').exists() or destination.exists():
            continue
        source=Path(scene['output'])
        if scene['id']=='scene_01':
            shutil.copyfile(source/'comparison_textured_preview.glb',destination)
        else:
            export(source/'preview/scene.glb',destination,Path(scene['data']))
        print(destination,flush=True)

if __name__=='__main__':main()
