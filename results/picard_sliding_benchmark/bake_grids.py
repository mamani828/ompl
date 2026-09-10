from pathlib import Path
import sys
sys.path.insert(0, '/ompl/ur5_experiments')
from ur5_nav import SimSession, make_env
from ur5_nav.sdf import bake, scene_primitives, open_mount_hole
out = Path('/ompl/results/picard_sliding_benchmark')
for source in sorted(Path('/ompl/ur5_experiments/out').glob('*.problem')):
    data = dict(line.split(maxsplit=1) for line in source.read_text().splitlines() if line and not line.startswith('#') and not line.startswith('goal '))
    sim = SimSession(gui=False)
    env = make_env(data['env'], sim.client, seed=int(data['seed']))
    primitives = open_mount_hole(scene_primitives(sim.client, [sim.plane_id] + env.obstacles), env.mount_body, half_width=0.30)
    grid, _ = bake(sim.client, None, voxel=float(data['voxel']), primitives=primitives)
    grid.save(str(out / data['grid']))
    (out / source.name).write_text(source.read_text())
    print('baked', source.stem, flush=True)
    sim.close()
