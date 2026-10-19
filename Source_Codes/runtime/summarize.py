"""Physical dispatch counts only; independently audited outcomes are aggregated separately."""
import json
from common import RUN,write
def measured():
    rows={}
    for p in (RUN/'results/physics').glob('**/result.json'):
        for r in json.loads(p.read_text())['rows']:
            key=(r['platform'],r['profile'],r['task_id'])
            assert key not in rows, key
            rows[key]=r
    return rows
if __name__=='__main__':
    rows=measured()
    write(RUN/'results/preflight_counts.json',dict(kind='main_study_dispatch',physical_rows=len(rows),formal_outcomes=0))
