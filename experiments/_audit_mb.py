# -*- coding: utf-8 -*-
"""Audit mushroom-body cell types in FlyWire v783 annotations."""
import pandas as pd, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ann = pd.read_csv(r'F:\Claude项目\fly-brain\data\annotations\Supplemental_file1_neuron_annotations.tsv',
                  sep='\t', low_memory=False)
print('columns:', list(ann.columns)[:20])
print('total annotated:', len(ann))

ct = ann['cell_type'].astype(str)
kc  = ann[ct.str.match(r'KC', na=False)]
mbon = ann[ct.str.match(r'MBON', na=False)]
dan = ann[ct.str.match(r'(PAM|PPL)', na=False)]
other_mb = ann[ct.str.match(r'(APL|DPM)', na=False)]

for name, df in [('Kenyon cells (KC)', kc), ('MBON', mbon), ('DAN (PAM+PPL)', dan), ('APL/DPM', other_mb)]:
    print(f'\n{name}: {len(df)}')
    if len(df):
        print('  cell types:', df['cell_type'].nunique())
        print('  top:', df['cell_type'].value_counts().head(8).to_dict())

# check against model neuron list
comp = pd.read_csv(r'F:\Claude项目\fly-brain\data\2025_Completeness_783.csv')
print('\nmodel neurons:', len(comp), 'cols:', list(comp.columns)[:10])
