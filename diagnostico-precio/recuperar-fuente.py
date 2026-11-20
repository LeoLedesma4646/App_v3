"""Descarga pública, sin credenciales; conserva fuente y huellas verificables."""
from pathlib import Path
import hashlib,json,zipfile,io
from datetime import datetime,timezone
import requests

OUT=Path(__file__).resolve().parent/'fuentes-publicas'
OUT.mkdir(exist_ok=True)
DATASET='beatafaron/fmcg-daily-sales-data-to-2022-2024'
COMMIT='190925c87ae60b7ff22b26a8d643f900aae8b398'
EXPECTED_ZIP_SHA256='c2ad83e0964513932fbe5ee4f05669567adeed95ac901dd0b3c678bda716ca21'
JOBS=[
 ('dataset-metadata.json',f'https://www.kaggle.com/api/v1/datasets/list?search={DATASET.split("/")[1]}'),
 ('dataset-v16.zip',f'https://www.kaggle.com/api/v1/datasets/download/{DATASET}'),
 ('notebook-autora.json','https://www.kaggle.com/api/v1/kernels/pull/beatafaron/fmcg-forecasting-ml-real-demand-signal-2'),
 ('notebook-eda-autora.json','https://www.kaggle.com/api/v1/kernels/pull/beatafaron/1-eda-key-business-insights-fmcg-2022-24'),
 ('github-repo.json','https://api.github.com/repos/BeataFaron/fmcg_forecasting_repo'),
 ('github-tree.json',f'https://api.github.com/repos/BeataFaron/fmcg_forecasting_repo/git/trees/{COMMIT}?recursive=1'),
 ('github-job-semanal.ipynb',f'https://raw.githubusercontent.com/BeataFaron/fmcg_forecasting_repo/{COMMIT}/Databricks_version/05_weekly_prediction_job.ipynb'),
 ('github-commits.json','https://api.github.com/repos/BeataFaron/fmcg_forecasting_repo/commits?per_page=30'),
]

def main():
    records=[]
    for name,url in JOBS:
        destination=OUT/name
        if destination.exists():
            if name=='dataset-v16.zip' and hashlib.sha256(destination.read_bytes()).hexdigest()!=EXPECTED_ZIP_SHA256:raise ValueError('El archivo no corresponde a la fuente v16 verificada.')
            records.append({'name':name,'url':url,'status':'conservado','bytes':destination.stat().st_size,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()});continue
        try:
            with requests.get(url,timeout=(10,60),stream=True) as response:
                response.raise_for_status()
                chunks=[];size=0
                for chunk in response.iter_content(65536):
                    size+=len(chunk)
                    if size>50_000_000:raise ValueError('Fuente supera el límite de 50 MB.')
                    chunks.append(chunk)
                raw=b''.join(chunks)
                if name=='dataset-v16.zip' and hashlib.sha256(raw).hexdigest()!=EXPECTED_ZIP_SHA256:raise ValueError('La descarga no corresponde a la versión v16 verificada. No se guarda ni se sustituye la base.')
                if name.endswith('.zip'):
                    if not zipfile.is_zipfile(io.BytesIO(raw)):raise ValueError('Respuesta no es un ZIP.')
                    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                        members=[{'name':m.filename,'bytes':m.file_size} for m in archive.infolist()]
                else:
                    json.loads(raw);members=None
                destination.write_bytes(raw)
                records.append({'name':name,'url':url,'status':response.status_code,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'members':members})
        except (requests.RequestException,ValueError) as exc:
            records.append({'name':name,'url':url,'status':'no recuperado','error':str(exc)})
        print(json.dumps(records[-1],ensure_ascii=False),flush=True)
    receipt={'documented_at':datetime.now(timezone.utc).isoformat(),'authentication':'ninguna; solo GET público','dataset':DATASET,'verified_dataset_version':16,'repository_commit':COMMIT,'files':records,'scope':'Documenta archivos recuperados/conservados; documented_at no representa la fecha de descarga de cada archivo ni frescura de los datos.'}
    (OUT/'recepcion.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')

if __name__=='__main__':main()
