import collections, json, statistics, math, csv
from pathlib import Path
out=Path(__file__).resolve().parent
rows=json.loads((out/'runs.json').read_text())
configs=['sequential','picard8_w1','picard8_w4','picard16_w4']
bykey=collections.defaultdict(dict)
for r in rows: bykey[r['scene'],r['seed'],r['goal']][r['config']]=r
matched=[x for x in bykey.values() if len(x)==4 and all(r['solved'] for r in x.values())]
print('Total rows:',len(rows),'all-config solved pairs:',len(matched))
print('scene,config,solved,attempts,matched_n,matched_median_ms,paired_geomean_ratio,median_filter_calls')
summary=[]
for scene in ['empty','pillars','clutter','shelf','corridor','wall_gap','ALL']:
 for c in configs:
  a=[r for r in rows if r['config']==c and (scene=='ALL' or r['scene']==scene)]
  m=[x for x in matched if scene=='ALL' or x[c]['scene']==scene]
  if not a: continue
  times=[x[c]['seconds']*1000 for x in m]
  ratios=[x[c]['seconds']/x['sequential']['seconds'] for x in m]
  row=dict(scene=scene,config=c,solved=sum(r['solved'] for r in a),attempts=len(a),matched_n=len(m),matched_median_ms=statistics.median(times) if m else None,paired_geomean_ratio=math.exp(statistics.mean(map(math.log,ratios))) if m else None,median_filter_calls=statistics.median([x[c]['filter_calls'] for x in m]) if m else None)
  summary.append(row)
  print(','.join(str(v) for v in row.values()))
print('unsafe states',sum(r['unsafe'] for r in rows),'self overlaps',sum(r['selfcoll'] for r in rows))
for c in configs[1:]:
 a=[r for r in rows if r['config']==c]
 totals={k:sum(r.get(k,0) for r in a) for k in ['accepted','attempts','fallbacks','map_calls','verify_calls']}
 print(c,totals)
with (out/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader();w.writerows(summary)
