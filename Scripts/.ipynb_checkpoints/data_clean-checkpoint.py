import GEOparse
import pandas as pd
gse = GEOparse.get_GEO(filepath="../Data/GSE7390_family.soft.gz",silent=True,)
gsms = {}
genes = {}
gpls = {}
for gsm_name, gsm in gse.gsms.items():
    gsms[gsm_name] = gsm.metadata['characteristics_ch1']
    genes[gsm_name] = gsm.table

for gpl_name, gpl in gse.gpls.items():
    gpls[gpl_name] = gpl.table
df = []
for name in gsms.keys():
    d = dict(zip(genes[name]['ID_REF'],genes[name]['VALUE']))
    d.update(dict([x.split(':') for x in gsms[name]]))
    df += [d]
    
df = pd.DataFrame(df)
df.to_csv("../Data/cleaned_data.csv")
df['e.os'].value_counts()
gene = pd.read_csv("../Data/gene.csv")
relevant_genes = gene["Gene"].values
selected = df[list(relevant_genes) + ['samplename', 'id', 'filename', 'hospital', 'age','size', 'Surgery_type', 'Histtype', 'Angioinv', 'Lymp_infil', 'node','grade', 'er', 't.rfs', 'e.rfs', 't.os', 'e.os', 't.dmfs', 'e.dmfs','t.tdm', 'e.tdm', 'risksg', 'NPI', 'risknpi', 'AOL_os_10y', 'risk_AOL','veridex_risk']]
selected.to_csv("../Data/selected.csv")