import re, importlib
parts = {
 22:'part5',23:'part5',24:'part6',25:'part6',26:'part6',27:'part6',28:'part6',
 29:'part7',30:'part7',31:'part7',32:'part7',
 33:'part8',34:'part8',35:'part8',36:'part8',37:'part8',
 38:'part9',39:'part9',40:'part9',41:'part9',42:'part9'}
mods = {p: importlib.import_module(p) for p in set(parts.values())}
for n in range(22,43):
    mod = mods[parts[n]]
    L = getattr(mod, f'LESSON_{n:02d}')
    en = L['en']; zh = L['zh']
    refs_en = sorted(set(int(x) for x in re.findall(r'Lesson (\d+)', en)))
    # zh: handle 第 N 课 and 第N课 and ranges 第N–M课
    zh_refs = set()
    for m in re.findall(r'第\s*(\d+)\s*[–—-]\s*(\d+)\s*课', zh):
        for v in range(int(m[0]), int(m[1])+1): zh_refs.add(v)
    for m in re.findall(r'第\s*(\d+)\s*课', zh):
        zh_refs.add(int(m))
    refs_zh = sorted(zh_refs)
    mark = "  *** REF SET DIFFERS ***" if set(refs_en) != set(refs_zh) else ""
    print(f"L{n}: en={refs_en} zh={refs_zh}{mark}")
