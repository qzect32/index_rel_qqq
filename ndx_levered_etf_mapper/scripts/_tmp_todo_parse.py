import re, pathlib
p=pathlib.Path(__file__).resolve().parents[1]/'TODO_STATUS_SCAFFOLDS.md'
text=p.read_text(encoding='utf-8',errors='ignore').splitlines()
items=[]
cur=None
for ln in text:
    m=re.match(r'^##\s+(\d+)\.\s+(.*)$',ln)
    if m:
        if cur: items.append(cur)
        cur={'i':int(m.group(1)),'title':m.group(2).strip(),'status':'IN-PROGRESS'}
        continue
    if cur and ln.strip().startswith('- STATUS:'):
        cur['status']=ln.split(':',1)[1].strip().upper()
if cur: items.append(cur)
open_items=[it for it in items if it['status']!='DONE']
print('total',len(items),'open',len(open_items))
print('first5',open_items[:5])
