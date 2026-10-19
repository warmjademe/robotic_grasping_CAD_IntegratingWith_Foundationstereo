"""Offline rendering/calibration audit. Ground truth is evaluation-only."""
import json
import numpy as np
from PIL import Image, ImageDraw
from common import RUN, transform, write, sha

def main():
    rows=[]; previews=[]
    for inp in sorted((RUN/'datasets/inputs').iterdir()):
        if not inp.is_dir():continue
        task=json.loads((inp/'task.json').read_text())
        c=json.loads((inp/'calibration.json').read_text())
        L,R=[np.array(c['cameras'][s]['world_from_camera']) for s in ['left','right']]
        K=np.array(c['cameras']['left']['K'])
        relative=np.linalg.inv(L)@R
        assert np.allclose(relative[:3,:3],np.eye(3),atol=1e-7)
        assert np.allclose(relative[:3,3],[c['baseline_m'],0,0],atol=1e-7)
        scene=json.loads((RUN/'datasets/evaluation'/inp.name/'scene.json').read_text())
        target=next(x for x in scene['instances'] if x['role']=='target')
        with np.load(RUN/'datasets/assets'/task['object_id']/'cad_samples.npz') as d:
            world=transform(d['points_m'],np.array(target['world_from_object']))
        pc=[transform(world,np.linalg.inv(T)) for T in [L,R]]
        pixels=[p[:,:2]/p[:,2,None]*K[[0,1],[0,1]]+K[:2,2] for p in pc]
        vertical=float(np.abs(pixels[0][:,1]-pixels[1][:,1]).max())
        reconstructed=K[0,0]*c['baseline_m']/(pixels[0][:,0]-pixels[1][:,0])
        depth_error=float(np.abs(reconstructed-pc[0][:,2]).max())
        assert vertical<1e-6 and depth_error<1e-6
        images=[Image.open(inp/f'{s}.png').convert('RGB') for s in ['left','right']]
        assert all(im.size==(c['width'],c['height']) and np.asarray(im).std()>5 for im in images)
        assert sha(inp/'left.png')!=sha(inp/'right.png')
        depth=np.load(RUN/'datasets/evaluation'/inp.name/'left_depth_gt.npy')
        uv=np.rint(pixels[0]).astype(int)
        inside=(uv[:,0]>=0)&(uv[:,0]<c['width'])&(uv[:,1]>=0)&(uv[:,1]<c['height'])
        agreement=np.abs(depth[uv[inside,1],uv[inside,0]]-pc[0][inside,2])<.002
        # Dense CAD samples must include visible surfaces that coincide with rendered depth.
        assert agreement.sum()>100,(inp.name,int(agreement.sum()))
        rows.append(dict(scene_id=inp.name,epipolar_max_px=vertical,
            disparity_depth_identity_error_m=depth_error,visible_cad_samples_agreeing_2mm=int(agreement.sum()),
            left_sha256=sha(inp/'left.png'),right_sha256=sha(inp/'right.png')))
        if len(previews)<6:
            panel=Image.new('RGB',(640,268),'white');draw=ImageDraw.Draw(panel)
            draw.text((8,5),task['object_id']+' : left / right',fill='black')
            for j,im in enumerate(images):panel.paste(im.resize((320,240)),(320*j,28))
            previews.append(panel)
    assert len(rows)==len(json.loads((RUN/'datasets/manifest.json').read_text())['scenes'])
    out=RUN/'results';out.mkdir(exist_ok=True)
    sheet=Image.new('RGB',(1280,804),'white')
    for i,p in enumerate(previews):sheet.paste(p,((i%2)*640,(i//2)*268))
    sheet.save(out/'stereo_dataset_preview.png')
    write(out/'dataset_validation.json',dict(passed=True,scenes=len(rows),rows=rows,
        role='Offline evaluation of rendering and calibration; never model input'))
    print('DATASET_VALIDATION_PASS',len(rows),flush=True)

if __name__=='__main__':main()
