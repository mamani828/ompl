import re, subprocess, sys, statistics, csv
from pathlib import Path
root = Path('/home/ax-ml-ubu24/ompl')
out  = root/'results/picard_sliding_benchmark'
demo = str(root/'build/demos/demo_UR5PyBulletScene')
# name, picardIterations, window, workers, adaptive
configs = [('sequential',0,8,1,1), ('picard_fixed',2,8,1,0), ('picard_adaptive',2,8,1,1)]
scenes  = ['empty','pillars','clutter','shelf','corridor']
rows=[]
for scene in scenes:
    for seed in range(3):
        for name,iters,window,workers,adapt in configs[seed%3:]+configs[:seed%3]:
            cmd=[demo,str(out/(scene+'.problem')),'2','','3','-1','-1','-1','0','0','-1',
                 '1.5','0.10','-1',str(seed*100),str(iters),str(window),str(workers),
                 '1','1','40','3',str(adapt)]
            r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
            if r.returncode not in (0,3):
                print('FAIL',scene,seed,name,r.returncode); print(r.stdout[-2000:]); sys.exit(1)
            goal=None
            for line in r.stdout.splitlines():
                m=re.match(r'^(\S+)\s+rrtc\s+(yes|no)\s',line)
                if m: goal=m[1]
                p=line.split()
                if len(p)>12 and p[0]=='cbf' and p[1] in ('yes','no'):
                    rows.append(dict(scene=scene,seed=seed,config=name,goal=goal,
                                     solved=int(p[1]=='yes'),seconds=float(p[2]),
                                     unsafe=int(p[5]),calls=int(p[7])))
        print(scene,seed,'done',flush=True)
with open('/tmp/claude-1000/-home-ax-ml-ubu24-ompl/63204d42-c153-4e02-a96c-a20322937cac/scratchpad/ab.csv','w') as f:
    w=csv.DictWriter(f,fieldnames=['scene','seed','config','goal','solved','seconds','unsafe','calls'])
    w.writeheader(); w.writerows(rows)
print('rows',len(rows))
