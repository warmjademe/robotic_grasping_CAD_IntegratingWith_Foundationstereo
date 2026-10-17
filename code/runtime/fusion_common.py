import json
from common import RUN,PROJECT,RUNTIME,sha,write,gpu_lock
ROOT=RUN
SOURCE=RUN
def read(path): return json.loads(path.read_text())
