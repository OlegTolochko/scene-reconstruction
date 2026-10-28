"""Area-weighted surface sampling and nearest-triangle metrics."""
import numpy as np

from ..common import RESOLUTION


def query(mesh):
    import open3d as o3d
    q = o3d.t.geometry.RaycastingScene(nthreads=4)
    q.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, np.uint32))))
    return q


def distances(q, points):
    import open3d as o3d
    return np.concatenate([q.compute_distance(o3d.core.Tensor(np.asarray(p, np.float32)), nthreads=4).numpy()
                           for p in np.array_split(points, max(1, int(np.ceil(len(points) / 25000))))])


def sample(mesh, seed, n=100000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(mesh.faces), size=n, p=mesh.area_faces / mesh.area)
    uv = rng.random((n, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    t = mesh.triangles[idx]
    return t[:, 0] + uv[:, :1] * (t[:, 1] - t[:, 0]) + uv[:, 1:] * (t[:, 2] - t[:, 0])


def stats(d, edge, diagonal):
    return dict(mean_source_units=float(d.mean()), median_source_units=float(np.median(d)),
                p95_source_units=float(np.quantile(d, .95)), max_sampled_source_units=float(d.max()),
                mean_voxels=float(d.mean() / (edge / RESOLUTION)),
                p95_voxels=float(np.quantile(d, .95) / (edge / RESOLUTION)),
                mean_fraction_diagonal=float(d.mean() / diagonal))


def compare(reference, prediction, edge):
    r_points, p_points = sample(reference, 42), sample(prediction, 43)
    rq, pq = query(reference), query(prediction)
    p_dist, r_dist = distances(rq, p_points), distances(pq, r_points)
    diagonal = float(np.linalg.norm(reference.extents))
    scores = {}
    thresholds = {f'{v}_voxels': v * edge / RESOLUTION for v in (1, 2, 4)}
    thresholds.update({f'{v}_diagonal': v * diagonal for v in (.005, .01, .02)})
    for label, threshold in thresholds.items():
        precision, recall = float((p_dist <= threshold).mean()), float((r_dist <= threshold).mean())
        scores[label] = dict(threshold_source_units=threshold, precision=precision, recall=recall,
                             fscore=2 * precision * recall / max(precision + recall, 1e-20))
    result = dict(prediction_to_reference=stats(p_dist, edge, diagonal),
                  reference_to_prediction=stats(r_dist, edge, diagonal), scores=scores,
                  symmetric_mean_source_units=float((p_dist.mean() + r_dist.mean()) / 2),
                  reference_faces=len(reference.faces), prediction_faces=len(prediction.faces),
                  prediction_area_over_reference=float(prediction.area / reference.area),
                  bounds_difference_source_units=(prediction.bounds - reference.bounds).tolist(),
                  samples_per_surface=100000, seeds=[42, 43], alignment='none',
                  method='area weighted samples to exact nearest triangles; unsigned distances')
    return result, rq, pq


