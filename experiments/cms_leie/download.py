#!/usr/bin/env python3
"""Gate 1, step 1: download the CMS 'Medicare Physician & Other Practitioners - by Provider' CSVs (one per calendar
year, URLs from CMS's DCAT catalog https://data.cms.gov/data.json, saved in out/by_provider_sources.json) and the
HHS-OIG LEIE UPDATED.csv; record URL, bytes, SHA-256, UTC timestamp, HTTP Last-Modified, rows, columns in
out/download_manifest.json; convert each CMS CSV to parquet (pyarrow, all columns read as strings so nothing is
re-interpreted) and delete the CSV to save disk. Resumable: years already in the manifest are skipped.
Usage: python3 download.py leie 2013 2014 ...   (any subset; 'leie' for the exclusion list)"""
import os, sys, json, hashlib, subprocess, datetime, time
import pyarrow as pa, pyarrow.csv as pacsv, pyarrow.parquet as pq
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'data'); OUT=os.path.join(HERE,'out')
MAN=os.path.join(OUT,'download_manifest.json')
man=json.load(open(MAN)) if os.path.exists(MAN) else {}
src={str(r['year']):r for r in json.load(open(os.path.join(OUT,'by_provider_sources.json')))}
LEIE_URL='https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv'
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()
def fetch(url,dest):
    t0=time.time()
    hdr=subprocess.run(['curl','-sS','-L','--max-time','900','-D','-','-o',dest,url],capture_output=True,text=True)
    if hdr.returncode!=0: raise RuntimeError(hdr.stderr)
    lm=[l.split(':',1)[1].strip() for l in hdr.stdout.splitlines() if l.lower().startswith('last-modified:')]
    return {'url':url,'bytes':os.path.getsize(dest),'sha256':sha256(dest),'downloaded_utc':datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'http_last_modified':lm[-1] if lm else None,'seconds':round(time.time()-t0,1)}
for key in sys.argv[1:]:
    if key in man: print('skip (in manifest):',key); continue
    if key=='leie':
        dest=os.path.join(DATA,'LEIE_UPDATED.csv'); rec=fetch(LEIE_URL,dest)
        tbl=pacsv.read_csv(dest,convert_options=pacsv.ConvertOptions(column_types={c:pa.string() for c in
            ['LASTNAME','FIRSTNAME','MIDNAME','BUSNAME','GENERAL','SPECIALTY','UPIN','NPI','DOB','ADDRESS','CITY','STATE','ZIP','EXCLTYPE','EXCLDATE','REINDATE','WAIVERDATE','WVRSTATE']}))
        rec.update({'rows':tbl.num_rows,'columns':tbl.column_names,'file':'LEIE_UPDATED.csv'})
        man[key]=rec; print('LEIE',rec['rows'],'rows',rec['bytes'],'bytes',rec['sha256'][:12],rec['http_last_modified'])
    else:
        r=src[key]; name=r['csv_url'].split('/')[-1]; dest=os.path.join(DATA,name); rec=fetch(r['csv_url'],dest)
        # read everything as string: no numeric inference, blanks stay null
        hdr=pacsv.read_csv(dest,read_options=pacsv.ReadOptions(skip_rows=0),parse_options=pacsv.ParseOptions(),convert_options=pacsv.ConvertOptions()).schema.names if False else None
        first=open(dest,encoding='utf-8-sig').readline().rstrip('\n').split(',')
        cols=[c.strip('"') for c in first]
        tbl=pacsv.read_csv(dest,read_options=pacsv.ReadOptions(encoding='utf-8'),convert_options=pacsv.ConvertOptions(column_types={c:pa.string() for c in cols},strings_can_be_null=True,null_values=['']))
        pqp=os.path.join(DATA,f'by_provider_{key}.parquet'); pq.write_table(tbl,pqp,compression='zstd')
        rec.update({'rows':tbl.num_rows,'columns':tbl.column_names,'file':name,'parquet':os.path.basename(pqp),'parquet_bytes':os.path.getsize(pqp),'catalog_modified':r['modified'],'catalog_title':r['title']})
        os.remove(dest); man[key]=rec
        print(key,rec['rows'],'rows',len(rec['columns']),'cols',rec['bytes'],'bytes',rec['sha256'][:12],'parquet',rec['parquet_bytes'],'b',rec['seconds'],'s')
    json.dump(man,open(MAN,'w'),indent=1)
