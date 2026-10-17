"""Hash the complete predeclared evaluation input cohort before inference."""
import json
from common import RUN, PROTOCOL, write, sha
manifest=json.loads((RUN/'datasets/manifest.json').read_text())
validation=json.loads((RUN/'results/dataset_validation.json').read_text())
assert validation['passed'] and validation['scenes']==len(manifest['scenes'])
inputs={s['scene_id']:{n:sha(RUN/'datasets/inputs'/s['scene_id']/n) for n in ['left.png','right.png','cameras.json','task.json']} for s in manifest['scenes']}
record=dict(status='frozen_before_inference',protocol_sha256=sha(RUN/'protocols/protocol.json'),
    dataset_manifest_sha256=sha(RUN/'datasets/manifest.json'),inputs=inputs,
    code_sha256={p.name:sha(p) for p in sorted((RUN/'code').glob('*.py'))},models_retrained=False)
dest=RUN/'protocols/input_freeze.json'
if dest.exists():
    old=json.loads(dest.read_text());assert old['inputs']==inputs and old['protocol_sha256']==record['protocol_sha256']
else: write(dest,record)
print('INPUT_COHORT_FROZEN',len(inputs),flush=True)
