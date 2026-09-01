import streamlit as st
from pipeline.secure_rag import secure_rag_answer
st.title('Secure RAG Poisoning Detection — Phase 1')
q=st.text_input('Query','Where is the Eiffel Tower located?'); defense=st.toggle('Defense ON',True)
if st.button('Run'):
 out=secure_rag_answer(q,defense); st.subheader('Answer'); st.write(out['answer']); st.subheader('Retrieved documents')
 for d in out['kept_docs']+out['filtered_docs']: st.write({'doc_id':d['doc_id'],'poison_probability':out['scores'][str(d['doc_id'])],'status':'kept' if d in out['kept_docs'] else 'filtered'}); st.caption(d['text'])
