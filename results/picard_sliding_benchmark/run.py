import csv, json, re, subprocess, time
from pathlib import Path
root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
configs = [('sequential',0,8,1), ('picard8_w1',2,8,1), ('picard8_w4',2,8,4), ('picard16_w4',2,16,4)]
rows = []
begin=time.time()
for scene in ['empty','pillars','clutter','shelf','corridor','wall_gap']:
  for seed in range(5):
    for name,iters,window,workers in configs[seed%4:]+configs[:seed%4]:
      cmd=[str(root/'build/demos/demo_UR5PyBulletScene'),str(out/(scene+'.problem')),'2','','1','-1','-1','-1','0','0','-1','1.5','0.10','-1',str(seed*100),'%d'%iters,str(window),str(workers)]
      run=subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
      (out/f'{scene}_{seed}_{name}.log').write_text(run.stdout)
      if run.returncode not in (0, 3): raise RuntimeError((cmd,run.returncode,run.stdout[-1000:]))
      goal=None
      for line in run.stdout.splitlines():
        m=re.match(r'^(\S+)\s+rrtc\s+(yes|no)\s+',line)
        if m: goal=m[1]
        parts=line.split()
        if len(parts)>12 and parts[0]=='cbf' and parts[1] in ('yes','no'):
          row=dict(scene=scene,seed=seed,config=name,goal=goal,solved=int(parts[1]=='yes'),seconds=float(parts[2]),unsafe=int(parts[5]),filter_calls=int(parts[7]),selfcoll=int(parts[10]))
          rows.append(row)
        if line.startswith('    parallel Picard:'):
          m=re.search(r'(\d+)/(\d+) proposals accepted, (\d+) fallbacks.*?(\d+) map \+ (\d+) verify filters',line)
          if m: rows[-1].update(zip(['accepted','attempts','fallbacks','map_calls','verify_calls'],map(int,m.groups())))
      (out/'runs.json').write_text(json.dumps(rows,indent=2))
    print(scene,'seed',seed,'done,',round(time.time()-begin,1),'seconds',flush=True)
with (out/'runs.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}))
  writer.writeheader();writer.writerows(rows)
print('completed',len(rows),'goal runs',flush=True)
