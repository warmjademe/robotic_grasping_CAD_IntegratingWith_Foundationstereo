"""Bind each immutable task to its newly generated cloud and candidate hashes."""
import json
from common import RUN,write,sha
path=RUN/'inputs/manifest.json';manifest=json.loads(path.read_text())
for task in manifest['tasks']:
    cloud=RUN/'results/perception'/task['perception']/task['episode_id']/'cloud.npz'
    grasps=RUN/'results/grasps'/task['task_id']/'grasps.npz'
    if cloud.exists(): task.update(cloud=str(cloud.relative_to(RUN)),cloud_sha256=sha(cloud))
    if grasps.exists(): task.update(grasps=str(grasps.relative_to(RUN)),grasps_sha256=sha(grasps))
write(path,manifest)
print('MANIFEST_READY',sum('grasps' in t for t in manifest['tasks']),flush=True)
