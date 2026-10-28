"""Export same-frame overlays and translated surface-error comparisons."""
import numpy as np
import trimesh

from ..common import RESOLUTION
from ..evaluation.surface import compare, distances


def colored(mesh, q, max_distance):
    import matplotlib
    d = distances(q, mesh.vertices)
    result = mesh.copy()
    result.visual = trimesh.visual.ColorVisuals(mesh=result,
        vertex_colors=(matplotlib.colormaps['turbo'](np.clip(d / max_distance, 0, 1)) * 255).astype(np.uint8))
    return result


def comparisons(dest, reference, decoded, edge):
    result, rq, pq = compare(reference, decoded, edge)
    r = colored(reference, pq, 4 * edge / RESOLUTION)
    p = colored(decoded, rq, 4 * edge / RESOLUTION)
    gap = max(reference.extents[0], decoded.extents[0]) * .15
    shift = float(reference.bounds[1, 0] - decoded.bounds[0, 0] + gap)
    display = np.eye(4)
    display[0, 3] = shift
    side = trimesh.Scene()
    side.add_geometry(r, node_name='ORIGINAL_reference_to_decoded_error')
    side.add_geometry(p, transform=display, node_name='DECODED_decoded_to_reference_error')
    side.metadata.update(experiment='Ground-truth latent reconstruction baseline',
        left='original GT', right='raw-latent decoded geometry',
        display_translation_x=shift, error_colormap='turbo: dark blue=0, red=4+ voxels',
        color_saturation_distance_source_units=4 * edge / RESOLUTION,
        geometry_resolution=RESOLUTION, alignment='none', metric_scale_verified=False)
    side.export(dest / 'comparison_surface_error.glb')
    if not (dest / 'decoded_pbr.glb').exists():
        plain = trimesh.Scene()
        plain.add_geometry(reference.copy(), node_name='ORIGINAL')
        plain.add_geometry(decoded.copy(), transform=display, node_name='DECODED_shape_only')
        plain.metadata.update(left='original geometry', right='exact decoded shape; no decoded texture channels',
                              display_translation_x=shift, error_colors=False, alignment='none')
        plain.export(dest / 'comparison.glb')
    overlay = trimesh.Scene()
    r = reference.copy()
    p = decoded.copy()
    r.visual = trimesh.visual.ColorVisuals(mesh=r, vertex_colors=[40, 150, 240, 160])
    p.visual = trimesh.visual.ColorVisuals(mesh=p, vertex_colors=[245, 125, 35, 160])
    overlay.add_geometry(r, node_name='ORIGINAL_blue')
    overlay.add_geometry(p, node_name='DECODED_orange')
    overlay.metadata.update(alignment='none', display_translation=[0, 0, 0], original='blue', decoded='orange')
    overlay.export(dest / 'overlay.glb')
    for path, expected in [(dest / 'comparison_surface_error.glb', side), (dest / 'overlay.glb', overlay)]:
        restored = trimesh.load_scene(path, process=False)
        expected_faces = len(reference.faces) + len(decoded.faces)
        actual_faces = sum(len(restored.geometry[restored.graph[n][1]].faces) for n in restored.graph.nodes_geometry)
        assert actual_faces == expected_faces
        assert np.allclose(restored.bounds, expected.bounds, atol=1e-6)
    result['comparison_export_roundtrip_verified'] = True
    return result


