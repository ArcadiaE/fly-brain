# -*- coding: utf-8 -*-
import zipfile, re, sys, io, glob, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def text(p):
    z = zipfile.ZipFile(p)
    xml = z.read('word/document.xml').decode('utf-8', 'ignore')
    xml = re.sub(r'</w:p>', '\n', xml)
    xml = re.sub(r'<[^>]+>', '', xml)
    return xml

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '博士申请')
which = sys.argv[1] if len(sys.argv) > 1 else '规划'
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 25000
files = sorted(glob.glob(os.path.join(base, '**', '*.docx'), recursive=True))
for f in files:
    if '.bak' in f:
        continue
    if which in os.path.basename(f):
        print('==== FILE:', f)
        print(text(f)[:limit])
