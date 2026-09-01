import json, shutil, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import BASE_CORPUS_PATH, DEMO_CORPUS_PATH, POISONED_DOCS_PATH, ensure_project_dirs
from retrieval.hybrid_retriever import HybridRetriever
def main():
 ensure_project_dirs()
 rows=[json.loads(x) for x in BASE_CORPUS_PATH.read_text().splitlines() if x.strip()] if BASE_CORPUS_PATH.exists() else [{'doc_id':i,'text':f'General reference passage {i} about science and geography.'} for i in range(50)]
 pois=[json.loads(x) for x in POISONED_DOCS_PATH.read_text().splitlines() if x.strip()]
 for poison in pois:
  poison.setdefault('title', f"Adversarial report: {poison['target_query']}")
  poison.setdefault('source_type', poison.get('attack_type', 'adversarial report'))
 DEMO_CORPUS_PATH.write_text('\n'.join(json.dumps(x) for x in rows+pois)+'\n'); print('Injected',len(pois),'documents into',DEMO_CORPUS_PATH)
 retriever=HybridRetriever(DEMO_CORPUS_PATH,force_rebuild=True,write_integrity_manifest=True)
 print('Poison retrieval report:')
 for poison in pois:
  ranked=retriever.retrieve(poison['target_query'],top_k=5)
  rank=next((i for i,d in enumerate(ranked,1) if d['doc_id']==poison['doc_id']),None)
  print(f"  doc={poison['doc_id']} rank={rank or 'not-top-5'} query={poison['target_query']}")
if __name__=='__main__': main()
