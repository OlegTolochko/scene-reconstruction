"""Area-weighted surface and held-out depth evaluation in source coordinates.

Uses CPU Open3D triangle distance/ray queries. No registration, rescaling,
prediction cropping, or preview simplification is applied to raw geometry.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image
import trimesh

from ..preparation.validate_room import read_depth, valid_depth

Y_TO_Z = np.array([[1,0,0,0], [0,0,-1,0], [0,1,0,0], [0,0,0,1]], dtype=float)
THRESHOLDS = (.005, .01, .02)

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def number(value):
    return float(value) if np.isfinite(value) else None

def average(values):
    return float(np.mean(values)) if len(values) else None

def distance_stats(distance, diagonal):
    if not len(distance):
        return {'count': 0, 'mean_fraction_diagonal': None, 'median_fraction_diagonal': None,
                'p95_fraction_diagonal': None, 'within_threshold': {str(t): None for t in THRESHOLDS}}
    return {'count': len(distance), 'mean_fraction_diagonal': float(distance.mean()/diagonal),
        'median_fraction_diagonal': float(np.median(distance)/diagonal),
        'p95_fraction_diagonal': float(np.quantile(distance,.95)/diagonal),
        'within_threshold': {str(t): float((distance <= t*diagonal).mean()) for t in THRESHOLDS}}

def make_query(mesh):
    # RaycastingScene uses a CPU acceleration structure; no GPU memory is needed.
    tensor = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(tensor)
    return scene

def distances(query, points):
    parts = []
    for batch in np.array_split(points, max(1, int(np.ceil(len(points)/25000)))):
        if len(batch):
            parts.append(query.compute_distance(o3d.core.Tensor(batch.astype(np.float32)), nthreads=4).numpy())
    return np.concatenate(parts) if parts else np.empty(0)

def samples(mesh, count, rng):
    faces = rng.choice(len(mesh.faces), count, p=mesh.area_faces/mesh.area)
    uv = rng.random((count,2))
    uv[uv.sum(axis=1)>1] = 1-uv[uv.sum(axis=1)>1]
    tri = mesh.vertices[mesh.faces[faces]]
    return tri[:,0] + uv[:,0,None]*(tri[:,1]-tri[:,0]) + uv[:,1,None]*(tri[:,2]-tri[:,0]), faces

def raycast(query, c2w, fov, xs, ys, width, height):
    focal = width/(2*np.tan(fov/2))
    local = np.stack([(xs+.5-width/2)/focal, -(ys+.5-height/2)/focal, -np.ones_like(xs)],axis=-1)
    directions = local @ c2w[:3,:3].T
    rays = np.concatenate([np.broadcast_to(c2w[:3,3], directions.shape), directions],axis=1)
    return query.cast_rays(o3d.core.Tensor(rays.astype(np.float32)), nthreads=4)['t_hit'].numpy()

def classify_projection(points, cameras, depths, fov, diagonal):
    """Visible if projected depth agrees with input reference depth.

    Visibility tolerance max(0.2% GT diagonal, 0.5% reference depth).
    Unknown includes outside every image and background; neither is called occluded.
    Free-space flags use a stricter 1% diagonal/1% depth tolerance.
    """
    visible = np.zeros(len(points),dtype=bool)
    behind = visible.copy()
    in_front = visible.copy()
    background = visible.copy()
    for camera, depth in zip(cameras, depths):
        h,w = depth.shape
        focal = w/(2*np.tan(fov/2))
        c2w = np.asarray(camera['transform_matrix'])
        local = (points-c2w[:3,3]) @ c2w[:3,:3]
        z = -local[:,2]
        with np.errstate(divide='ignore',invalid='ignore'):
            px = focal*local[:,0]/z+w/2
            py = -focal*local[:,1]/z+h/2
        inside = (z>0)&(px>=0)&(px<w)&(py>=0)&(py<h)
        selected = np.flatnonzero(inside)
        x,y = px[inside].astype(int), py[inside].astype(int)
        ref = depth[y,x]
        ok = valid_depth(ref)
        ids = selected[ok]
        tolerance = np.maximum(.002*diagonal,.005*ref[ok])
        delta = z[ids]-ref[ok]
        visible[ids] |= np.abs(delta)<=tolerance
        behind[ids] |= delta>tolerance
        in_front[ids] |= delta < -np.maximum(.01*diagonal,.01*ref[ok])
        background[selected[~ok]] = True
    return visible, behind & ~visible, ~(visible|behind), in_front, background

def error_artifacts(output, gt_points, pred_points, recall_distance, precision_distance, diagonal):
    figure, axes = plt.subplots(2,2,figsize=(12,10))
    for row,(points,values,title) in enumerate([
            (gt_points,recall_distance,'Reference: distance to prediction (missing/detail errors)'),
            (pred_points,precision_distance,'Prediction: distance to reference (extra/displaced geometry)')]):
        colors = (plt.get_cmap('turbo')(np.clip(values/(.02*diagonal),0,1))*255).astype(np.uint8)
        # Export y-up, matching inspection GLBs.
        y_up = trimesh.transform_points(points,np.linalg.inv(Y_TO_Z))
        trimesh.points.PointCloud(y_up,colors=colors).export(output/('reference_errors.ply' if row==0 else 'prediction_errors.ply'))
        pick = np.arange(0,len(points),max(1,len(points)//20000))
        for col,(x,y,name) in enumerate([(0,1,'top: X/Y'),(0,2,'front: X/Z')]):
            ax = axes[row,col]
            im = ax.scatter(points[pick,x],points[pick,y],c=values[pick]/diagonal,
                            vmin=0,vmax=.02,cmap='turbo',s=.5,rasterized=True)
            ax.set_aspect('equal'); ax.set_title(title+'\n'+name)
            ax.set_xlabel('Source coordinate'); ax.set_ylabel('Source coordinate')
    figure.colorbar(im,ax=axes.ravel().tolist(),label='Distance / reference diagonal (clipped at 0.02)')
    figure.savefig(output/'surface_errors.png',dpi=140,bbox_inches='tight'); plt.close(figure)

def evaluate(data, inputs, prediction, output, sample_count=100000, grid=192, preview=None):
    began = time.monotonic()
    output.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((data/'manifest.json').read_text())
    input_manifest = json.loads((inputs/'input_manifest.json').read_text())
    if input_manifest['room'] != manifest['room']:
        raise RuntimeError('Input and reference room differ')
    render = data/f"3D-FRONT-TEST-RENDER/{manifest['room']}"
    gt_path = data/f"3D-FRONT-TEST-SCENE/{manifest['room']}.glb"
    original = trimesh.load_scene(gt_path,process=False)
    objects, labels, rows = [], [], []
    for i,node in enumerate(original.graph.nodes_geometry):
        transform,name = original.graph[node]
        obj = original.geometry[name].copy()
        obj.apply_transform(Y_TO_Z @ transform)
        objects.append(obj)
        labels.extend([i]*len(obj.faces))
        rows.append({'object_id':i,'node':str(node),'geometry':str(name),'area':float(obj.area),
                     'extents_source_units':obj.extents.tolist(),'faces':len(obj.faces)})
    gt = trimesh.util.concatenate(objects)
    labels = np.asarray(labels)
    diagonal = np.linalg.norm(gt.extents)
    pred = trimesh.load(prediction,process=False,force='mesh')
    # Main input is native inference PLY in Blender z-up. GLBs already use y-up.
    if prediction.suffix.lower() == '.glb':
        pred.apply_transform(Y_TO_Z)
    if not np.isfinite(pred.vertices).all() or not len(pred.faces):
        raise RuntimeError('Invalid prediction')
    print(f'Loaded {len(pred.faces):,} prediction triangles; building CPU queries',flush=True)
    gt_query, pred_query = make_query(gt), make_query(pred)
    rng = np.random.default_rng(42)
    gt_points, gt_faces = samples(gt,sample_count,rng)
    pred_points,_ = samples(pred,sample_count,rng)
    recall_dist = distances(pred_query,gt_points)
    precision_dist = distances(gt_query,pred_points)
    precision = distance_stats(precision_dist,diagonal)
    recall = distance_stats(recall_dist,diagonal)
    fscores = {}
    for t in THRESHOLDS:
        p,r = precision['within_threshold'][str(t)],recall['within_threshold'][str(t)]
        fscores[str(t)] = {'precision':p,'recall':r,'fscore':2*p*r/(p+r) if p+r else 0.}
    meta = json.loads((render/'meta.json').read_text())
    input_ids = {v['index'] for v in input_manifest['views']}
    cameras = [c for c in meta['locations'] if c['index'] in input_ids]
    heldout = [c for c in meta['locations'] if c['index'] not in input_ids]
    if len(cameras)!=8 or not heldout:
        raise RuntimeError('Expected eight input cameras and nonempty held-out split')
    depths = [read_depth(render/f"depth_{c['index']}.exr") for c in cameras]
    visible,occluded,unknown,_,_ = classify_projection(gt_points,cameras,depths,meta['camera_angle_x'],diagonal)
    _,_,_,front,background = classify_projection(pred_points,cameras,depths,meta['camera_angle_x'],diagonal)
    visibility = {name:distance_stats(recall_dist[mask],diagonal) for name,mask in
                  [('visible',visible),('occluded',occluded),('unclassified',unknown)]}
    model_cameras_path = prediction.parent/'cameras.json'
    if not model_cameras_path.exists():
        raise RuntimeError('Missing actual model camera record')
    model_camera_record = json.loads(model_cameras_path.read_text())
    model_ids = {Path(c['img_path']).stem for c in model_camera_record['scene']}
    if not model_ids or not model_ids <= input_ids:
        raise RuntimeError('Actual conditioning cameras must be a nonempty subset of supplied inputs')
    effective = [(c,d) for c,d in zip(cameras,depths) if c['index'] in model_ids]
    effective_visible,effective_occluded,effective_unknown,_,_ = classify_projection(
        gt_points,[c for c,d in effective],[d for c,d in effective],meta['camera_angle_x'],diagonal)
    effective_visibility = {name:distance_stats(recall_dist[mask],diagonal) for name,mask in
        [('visible',effective_visible),('occluded',effective_occluded),('unclassified',effective_unknown)]}
    reconstruction_log = prediction.parent/'reconstruction.log'
    fallback_lines = [line for line in reconstruction_log.read_text().splitlines()
                      if '[get_images]' in line and 'falling back' in line] if reconstruction_log.exists() else []
    chunk_path = prediction.parent/'chunk_transforms.json'
    chunk_count = len(json.loads(chunk_path.read_text())['chunks']) if chunk_path.exists() else None
    sample_object = labels[gt_faces]
    for i,row in enumerate(rows):
        ids = sample_object==i
        row.update(distance_stats(recall_dist[ids],diagonal))
        row['visible_sample_fraction'] = average(visible[ids])
        row['visible_recall_at_1pct'] = average(recall_dist[ids & visible]<=.01*diagonal)
        row['occluded_recall_at_1pct'] = average(recall_dist[ids & occluded]<=.01*diagonal)
    out_of_bounds = np.any((pred_points<gt.bounds[0]-.01*diagonal)|(pred_points>gt.bounds[1]+.01*diagonal),axis=1)
    architectural_context = None
    context_parts = []
    context_paths = [gt_path.with_suffix('')/name for name in ('ceil.glb','wall.glb','floor.glb','others.glb')]
    context_paths = [p for p in context_paths if p.exists()]
    for context_path in context_paths:
        part = trimesh.load_scene(context_path,process=False).to_mesh()
        part.apply_transform(Y_TO_Z)
        if len(part.faces):context_parts.append(part)
    if context_parts:
        shell_distance = distances(make_query(trimesh.util.concatenate(context_parts)),pred_points)
        full_reference = trimesh.util.concatenate([gt,*context_parts])
        full_points,_ = samples(full_reference,sample_count,np.random.default_rng(43))
        full_precision = distance_stats(np.minimum(shell_distance,precision_dist),diagonal)
        full_recall = distance_stats(distances(pred_query,full_points),diagonal)
        full_thresholds = {}
        for t in THRESHOLDS:
            p,r = full_precision['within_threshold'][str(t)],full_recall['within_threshold'][str(t)]
            full_thresholds[str(t)] = {'precision':p,'recall':r,'fscore':2*p*r/(p+r) if p+r else 0.}
        architectural_context = {
            'scope':'Additional released assets in supplied coordinates, absent from primary furniture renders/reference; others.glb has no reliable semantic labels. Context only.',
            'asset_files':[p.name for p in context_paths],
            'all_released_assets_geometry':{'reference_faces':len(full_reference.faces),
                'normalization':'Same furniture-reference diagonal D as primary metrics; full-reference sampling seed 43',
                'prediction_to_reference':full_precision,'reference_to_prediction':full_recall,
                'threshold_metrics':full_thresholds},
            'prediction_near_additional_assets_and_far_from_furniture_at_1pct_D':float(((shell_distance<=.01*diagonal)&(precision_dist>.01*diagonal)).mean()),
            'prediction_near_furniture_or_additional_assets_at_1pct_D':float((np.minimum(shell_distance,precision_dist)<=.01*diagonal).mean())}
    per_view = []
    validation = []
    total = {'gt_valid':0,'pred_valid':0,'intersection':0,'union':0,'missing':0,'extra':0}
    abs_sum = rel_sum = squared_sum = 0.
    correct = {str(t):0 for t in (.01,.05,.1)}
    figure,axes = plt.subplots(len(heldout),6,figsize=(20,3.8*len(heldout)),squeeze=False)
    # Validate the camera/depth convention against reference geometry for ALL views.
    for camera in meta['locations']:
        index = camera['index']
        depth = read_depth(render/f'depth_{index}.exr')
        h,w = depth.shape
        axis_x = np.linspace(0,w-1,grid).round().astype(int)
        axis_y = np.linspace(0,h-1,grid).round().astype(int)
        xx,yy = np.meshgrid(axis_x,axis_y)
        xs,ys = xx.ravel(),yy.ravel()
        c2w = np.asarray(camera['transform_matrix'])
        target = depth[ys,xs]
        gt_valid = valid_depth(target)
        reference_cast = raycast(gt_query,c2w,meta['camera_angle_x'],xs,ys,w,h)
        reference_valid = np.isfinite(reference_cast)
        both = gt_valid&reference_valid
        consistency = {'index':index,'paired_pixels':int(both.sum()),
            'silhouette_iou':float(both.sum()/max(1,(gt_valid|reference_valid).sum())),
            'median_relative_depth_error':number(np.median(np.abs(reference_cast[both]-target[both])/target[both])) if both.any() else None}
        validation.append(consistency)
        if index in input_ids:
            continue
        predicted = raycast(pred_query,c2w,meta['camera_angle_x'],xs,ys,w,h)
        pred_valid = np.isfinite(predicted)
        both,union = gt_valid&pred_valid, gt_valid|pred_valid
        missing,extra = gt_valid&~pred_valid,pred_valid&~gt_valid
        err = np.abs(predicted[both]-target[both])
        rel = err/target[both]
        counts = {'gt_valid':int(gt_valid.sum()),'pred_valid':int(pred_valid.sum()),
            'intersection':int(both.sum()),'union':int(union.sum()),
            'missing':int(missing.sum()),'extra':int(extra.sum())}
        for key,value in counts.items():
            total[key]+=value
        abs_sum+=err.sum(); rel_sum+=rel.sum(); squared_sum+=(err**2).sum()
        for threshold in correct:
            correct[threshold]+=int((rel<float(threshold)).sum())
        per_view.append({'index':index,**counts,
            'silhouette_iou':float(both.sum()/max(1,union.sum())),
            'depth_coverage':float(both.sum()/max(1,gt_valid.sum())),
            'missing_fraction_gt':float(missing.sum()/max(1,gt_valid.sum())),
            'extra_fraction_pred':float(extra.sum()/max(1,pred_valid.sum())),
            'depth_mae_fraction_diagonal':average(err/diagonal),
            'depth_absrel':average(rel),'depth_rmse_fraction_diagonal':number(np.sqrt(np.mean(err**2))/diagonal) if len(err) else None,
            'correct_depth_fraction_all_gt':{t:float((rel<float(t)).sum()/max(1,gt_valid.sum())) for t in correct}})
        row = len(per_view)-1
        rgba = np.asarray(Image.open(render/f'render_{index}.webp').convert('RGBA'))[yy,xx]
        error_image = np.full(len(xs),np.nan); error_image[both]=rel
        masks = np.zeros((len(xs),3)); masks[missing]=[1,0,0];masks[extra]=[0,.6,1];masks[both]=[.3,.3,.3]
        axes[row,0].imshow(rgba); axes[row,0].set_title(f'Held-out {index}: reference RGB')
        for col,values,title in [(1,np.where(gt_valid,target,np.nan),'Reference depth'),
                                 (2,np.where(pred_valid,predicted,np.nan),'Predicted depth')]:
            axes[row,col].imshow(values.reshape(grid,grid),vmin=target[gt_valid].min(),vmax=target[gt_valid].max(),cmap='viridis')
            axes[row,col].set_title(title)
        axes[row,3].imshow(error_image.reshape(grid,grid),vmin=0,vmax=.1,cmap='turbo')
        axes[row,3].set_title('Absolute relative error (0–10%)')
        axes[row,4].imshow(masks.reshape(grid,grid,3)); axes[row,4].set_title('Missing red / extra blue')
        axes[row,5].axis('off'); axes[row,5].text(.02,.7,
            f"IoU: {per_view[-1]['silhouette_iou']:.3f}\nCoverage: {per_view[-1]['depth_coverage']:.3f}\nAbsRel: {average(rel) if len(rel) else float('nan'):.3f}",fontsize=13)
    for ax in axes.ravel(): ax.axis('off')
    figure.tight_layout();figure.savefig(output/'heldout_views.png',dpi=130);plt.close(figure)
    aligned = all(v['paired_pixels']>=20 and v['silhouette_iou']>=.95 and
                  v['median_relative_depth_error'] is not None and v['median_relative_depth_error']<.01 for v in validation)
    heldout_metrics = {**total,'silhouette_iou':total['intersection']/max(1,total['union']),
        'depth_coverage':total['intersection']/max(1,total['gt_valid']),
        'missing_fraction_gt':total['missing']/max(1,total['gt_valid']),
        'extra_fraction_pred':total['extra']/max(1,total['pred_valid']),
        'depth_mae_fraction_diagonal':float(abs_sum/total['intersection']/diagonal) if total['intersection'] else None,
        'depth_absrel':float(rel_sum/total['intersection']) if total['intersection'] else None,
        'depth_rmse_fraction_diagonal':float(np.sqrt(squared_sum/total['intersection'])/diagonal) if total['intersection'] else None,
        'correct_depth_fraction_all_gt':{t:n/max(1,total['gt_valid']) for t,n in correct.items()},'per_view':per_view}
    error_artifacts(output,gt_points,pred_points,recall_dist,precision_dist,diagonal)
    preview_metrics = None
    if preview and preview.exists():
        preview_mesh = trimesh.load(preview,force='mesh',process=False)
        preview_mesh.apply_transform(Y_TO_Z)
        preview_query = make_query(preview_mesh)
        preview_points,_ = samples(preview_mesh,sample_count,np.random.default_rng(42))
        p_dist = distances(gt_query,preview_points)
        r_dist = distances(preview_query,gt_points)
        preview_metrics = {'faces':len(preview_mesh.faces),'surface_area':float(preview_mesh.area),
            'prediction_to_reference':distance_stats(p_dist,diagonal),
            'reference_to_prediction':distance_stats(r_dist,diagonal),
            'raw_to_preview':distance_stats(distances(preview_query,pred_points),diagonal),
            'preview_to_raw':distance_stats(distances(pred_query,preview_points),diagonal)}
    failures = []
    if model_ids != input_ids:
        failures.append(f"Native camera filtering dropped {len(input_ids-model_ids)} of eight supplied RGB views: {', '.join(sorted(input_ids-model_ids))}. Depth support still uses the eight supplied views.")
    if fallback_lines:
        failures.append(f'Native image selector reported {len(fallback_lines)} camera-selection fallback warnings; inspect reconstruction.log and cameras.json.')
    if not aligned: failures.append('Reference camera/depth consistency gate failed; quality metrics require investigation.')
    primary = fscores['0.01']
    if primary['precision']<.9: failures.append(f"Unmatched furniture-target geometry: {1-primary['precision']:.1%} of sampled predicted surface lies farther than 1% diagonal from primary furniture reference; inspect supplementary-asset comparison before calling it hallucination.")
    if primary['recall']<.9: failures.append(f"Incomplete/detail loss: {1-primary['recall']:.1%} of sampled reference furniture surface lies farther than 1% diagonal from prediction.")
    if heldout_metrics['missing_fraction_gt']>.05: failures.append(f"Held-out holes: {heldout_metrics['missing_fraction_gt']:.1%} of reference foreground has no predicted depth.")
    if heldout_metrics['extra_fraction_pred']>.05: failures.append(f"Held-out extra silhouettes: {heldout_metrics['extra_fraction_pred']:.1%} of predicted foreground lies on reference background.")
    for row in rows:
        if row['count']>=30 and row['within_threshold']['0.01']<.8:
            failures.append(f"Object {row['object_id']} ({row['node']}): recall@1% diagonal {row['within_threshold']['0.01']:.1%} ({row['count']} area samples).")
    report = {'room':manifest['room'],'dataset_revision':manifest['revision'],
        'prediction':str(prediction),'prediction_sha256':hashlib.sha256(prediction.read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256(gt_path.read_bytes()).hexdigest(),
        'input_ids':sorted(input_ids),'heldout_ids':[c['index'] for c in heldout],
        'effective_conditioning_ids':sorted(model_ids),'dropped_conditioning_ids':sorted(input_ids-model_ids),
        'chunk_count':chunk_count,'camera_selection_fallbacks':fallback_lines,
        'native_camera_selection_policy':'Chunk cube clipped projected bounding-box area >=0.4 of image; closest-camera fallback. See papers/GenRecon/genrecon/datasets/components.py:237 and inference/get_images.py:127.',
        'settings':{'seed':42,'area_samples_per_mesh':sample_count,'ray_grid':grid,
            'distance_method':'area-uniform samples to nearest triangle surface, exact CPU query; untrimmed',
            'thresholds_fraction_gt_diagonal':THRESHOLDS,'alignment':'fixed supplied coordinates; no ICP, scale fitting or cropping',
            'visibility_tolerance':'max(0.002 * GT diagonal, 0.005 * reference depth)',
            'free_space_tolerance':'max(0.01 * GT diagonal, 0.01 * reference depth)',
            'failure_flags':'descriptive screening: precision/recall <0.9; missing/extra silhouette >0.05; object recall <0.8 with >=30 samples'},
        'metric_scale_verified':False,'gt_diagonal_source_units':float(diagonal),
        'reference_faces':len(gt.faces),'prediction_faces':len(pred.faces),
        'reference_surface_area':float(gt.area),'prediction_surface_area':float(pred.area),
        'prediction_to_reference':precision,'reference_to_prediction':recall,'threshold_metrics':fscores,
        'symmetric_mean_distance_fraction_diagonal':.5*(precision['mean_fraction_diagonal']+recall['mean_fraction_diagonal']),
        'visibility_stratified_recall':visibility,
        'visibility_stratified_recall_effective_conditioning':effective_visibility,
        'predicted_surface_input_free_space_violation_fraction':float(front.mean()),
        'predicted_surface_input_background_projection_fraction':float(background.mean()),
        'predicted_surface_outside_expanded_gt_bounds_fraction':float(out_of_bounds.mean()),
        'architectural_context':architectural_context,
        'heldout':heldout_metrics,'per_object':rows,'camera_consistency':validation,'camera_consistency_passed':aligned,
        'preview_comparison':preview_metrics,'failure_flags':failures,'elapsed_seconds':time.monotonic()-began}
    save_json(output/'metrics.json',report)
    with (output/'objects.csv').open('w') as handle:
        fields=['object_id','node','area','count','visible_sample_fraction','recall_005','recall_01','recall_02','mean_distance_fraction_diagonal','p95_distance_fraction_diagonal']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({**{k:row[k] for k in fields[:5]},
                'recall_005':row['within_threshold']['0.005'],'recall_01':row['within_threshold']['0.01'],'recall_02':row['within_threshold']['0.02'],
                'mean_distance_fraction_diagonal':row['mean_fraction_diagonal'],'p95_distance_fraction_diagonal':row['p95_fraction_diagonal']})
    print(json.dumps({'room':manifest['room'],'camera_consistency_passed':aligned,'threshold_metrics':fscores,
                      'heldout':{k:v for k,v in heldout_metrics.items() if k!='per_view'},'elapsed_seconds':report['elapsed_seconds']},indent=2),flush=True)
    if not aligned:
        raise RuntimeError('Reference alignment gate failed; inspect metrics before reporting benchmark scores')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--prediction',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preview',type=Path)
    parser.add_argument('--samples',type=int,default=100000)
    parser.add_argument('--grid',type=int,default=192)
    args=parser.parse_args()
    evaluate(args.data,args.inputs,args.prediction,args.output,args.samples,args.grid,args.preview)
