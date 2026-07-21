import json
p=r'C:\Users\ll\AppData\Local\Temp\claude\E--github-opencv-screen-detector\8a2c7dd6-f924-4c3b-822c-5e3c797ab266\tasks\wn15ra3vh.output'
d=json.loads(open(p,encoding='utf-8').read())
r=d['result']
if isinstance(r,str): r=json.loads(r)
s=r['synthesis']
out=[]
out.append('=== RECOMMENDED DEFAULT CONFIG ===')
out.append(s['recommended_default_config'])
out.append('\n=== EXPERIMENT MATRIX (by priority) ===')
for e in sorted(s['experiment_matrix'], key=lambda x:x['priority']):
    out.append(f"P{e['priority']} [{e['id']}] {e['name']}\n    hyp: {e['hypothesis']}\n    change: {e['change']}")
out.append('\n=== RISKS ===')
for x in s.get('risks',[]): out.append('- '+x)
open('experiment/cnn_fft_dwt_ablation/synth.txt','w',encoding='utf-8').write('\n'.join(out))
print("written", len('\n'.join(out)))
