"""Recompute lift/hold from retained traces; bind conclusions to raw evidence."""
import argparse
import sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from project_common import read,sha,write


def audit(run,calibration_only=False):
    protocol=read(run/'protocols/protocol.json');cfg=protocol['physics']
    sys.path.insert(0,str(run/'code'))
    from scenes import calibration
    calibrations={c['task_id']:c for c in calibration()}
    tasks={t['task_id']:t for t in read(run/'inputs/manifest.json')['tasks']}
    files=sorted((run/'calibration').glob('**/result.json'))
    if not calibration_only:files+=sorted((run/'results/physics').glob('**/result.json'))
    checked=[];keys=set()
    for file in files:
        report=read(file);platform=report['platform'];is_isaac=platform=='franka_isaac'
        assert platform in protocol['platforms']
        assert report['status']=='completed' and not report.get('object_attached',False)
        assert not report.get('object_pose_written_after_initialization',False)
        assert report['protocol_sha256']==sha(run/'protocols/protocol.json')
        script='run_isaac.py' if is_isaac else 'run_mujoco.py'
        assert report['code_sha256']==sha(run/'code'/script)
        tracepath=file.parent/'trace.npz';digest=sha(tracepath)
        with np.load(tracepath) as trace:
            phases=trace['phases'];times=trace['time_s'];hold=phases=='hold';active=phases!='settle'
            assert hold.sum()==cfg['trace_hz']*cfg['hold_seconds']
            continuous=bool(np.allclose(np.diff(times),1/cfg['trace_hz'],atol=1e-5))
            for raw in report['rows']:
                tid=raw['task_id'];key=(platform,tid);assert key not in keys;keys.add(key)
                is_cal=tid in calibrations
                if is_cal:
                    case=calibrations[tid];meshfile=Path(case['mesh']);selection=case['selection']
                else:
                    task=tasks[tid]
                    assert platform in task['platforms'], 'Task dispatched to an unregistered platform'
                    meshfile=run/'datasets/assets'/task['object_id']/'mesh_m.ply'
                    plan=read(run/'results/plans'/(tid+'.json'))
                    assert plan['status']=='sampled_path_found';selection=plan['selected']
                vertices=trimesh.load(meshfile,force='mesh').vertices
                if is_isaac:
                    index=list(trace['case_ids']).index(tid);poses=trace['object_pose'][hold,index]
                else:poses=trace['object_pose'][hold]
                assert np.isfinite(poses).all()
                rotations=Rotation.from_quat(poses[:,[4,5,6,3]]).as_matrix()
                bottom=np.min(np.einsum('tv,pv->tp',rotations[:,2,:],vertices),axis=1)+poses[:,2]
                minimum=float(bottom.min());lift=bool(minimum>=cfg['lift_bottom_clearance_m'])
                assert abs(minimum-raw['min_hold_bottom_clearance_m'])<1e-6
                assert lift==raw['physics_lift_success']
                ep=raw['approach_position_error_mm'];er=raw['approach_rotation_error_deg']
                if is_isaac:
                    H=np.array(selection['hand_pose']);H[:3,3]+=trace['origins'][index]
                    hand=list(trace['body_names']).index('panda_hand')
                    actual=trace['bodies'][np.flatnonzero(phases=='approach_hold')[-1],index,hand]
                    new_ep=float(np.linalg.norm(actual[:3]-H[:3,3])*1000)
                    new_er=float(np.rad2deg((Rotation.from_quat(actual[[4,5,6,3]]).inv()*Rotation.from_matrix(H[:3,:3])).magnitude()))
                    assert abs(new_ep-ep)<1e-5 and abs(new_er-er)<1e-5
                else:
                    recorded=np.max(trace['contact_resultants'][active],axis=0)
                    fullrate=np.array([raw['max_arm_contact_force_n'],raw['max_finger_non_target_force_n']])
                    assert np.all(recorded<=fullrate+1e-6)
                reached=bool(ep<=cfg['endpoint_position_m']*1000 and er<=cfg['endpoint_rotation_degrees'])
                collision=raw['prohibited_contact_steps']==0
                peak=max(raw['max_arm_contact_force_n'],raw['max_finger_non_target_force_n'])
                assert collision==(peak<=cfg['prohibited_contact_resultant_n'])
                assert reached==raw['approach_endpoint_reached'] and collision==raw['collision_free']
                success=bool(lift and reached and collision)
                assert success==raw['execution_success']
                if success:assert continuous,'Discontinuous successful trajectory'
                if is_cal:assert success==case['expected']
                checked.append(dict(task_id=tid,platform=platform,calibration=is_cal,success=success,
                    hold_frames=int(hold.sum()),minimum_hold_clearance_m=minimum,
                    continuous=continuous,isaac_endpoint_recomputed=is_isaac,
                    result=str(file.relative_to(run)),result_sha256=sha(file),trace_sha256=digest))
    calibration_count=len(protocol['platforms'])*len(protocol['calibration'])
    assert {(x['platform'],x['task_id']) for x in checked if x['calibration']}=={
        (p,c) for p in protocol['platforms'] for c in protocol['calibration']}, 'Each selected engine requires all calibration cases'
    result=dict(passed=True,calibration_only=calibration_only,physical_records=len(checked),
        calibration_records=calibration_count,rows=checked,code_sha256=sha(__file__),
        scope='Lift and duration recomputed from 50 Hz traces; Isaac endpoint recomputed from body poses. MuJoCo endpoint and 250 Hz contact maxima checked against recorded engine results; sampled traces cannot exclude between-sample contact.')
    write(run/'results'/('calibration_trace_audit.json' if calibration_only else 'physical_trace_audit.json'),result)
    print('PHYSICAL_RECORD_AUDIT',len(checked),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--calibration-only',action='store_true');a=p.parse_args();audit(a.run,a.calibration_only)
