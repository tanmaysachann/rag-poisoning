"""Create polished, detailed PDF knowledge reports for the demo corpus."""
from pathlib import Path
import json, hashlib
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'output'/'pdf'; OUT.mkdir(parents=True,exist_ok=True)
POISON=ROOT/'data'/'poisoned_docs.jsonl'
def styles():
 s=getSampleStyleSheet(); return {
  'title':ParagraphStyle('title',parent=s['Title'],fontName='Helvetica-Bold',fontSize=25,leading=30,textColor=colors.HexColor('#12333a'),spaceAfter=10),
  'sub':ParagraphStyle('sub',parent=s['Normal'],fontSize=10,leading=14,textColor=colors.HexColor('#4e6870')),
  'h':ParagraphStyle('h',parent=s['Heading2'],fontName='Helvetica-Bold',fontSize=13,leading=17,textColor=colors.HexColor('#0c6973'),spaceBefore=14,spaceAfter=6),
  'body':ParagraphStyle('body',parent=s['BodyText'],fontSize=10.2,leading=15,textColor=colors.HexColor('#26343a'),spaceAfter=7),
  'callout':ParagraphStyle('callout',parent=s['BodyText'],fontSize=9.8,leading=14,textColor=colors.HexColor('#6b3d19'),backColor=colors.HexColor('#fff4e5'),borderPadding=8),
  'small':ParagraphStyle('small',parent=s['BodyText'],fontSize=8,leading=11,textColor=colors.HexColor('#61747a')),
 }
def footer(canvas,doc):
 canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#d9e3e4')); canvas.line(18*mm,15*mm,192*mm,15*mm); canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#71858a')); canvas.drawString(18*mm,10*mm,'SENTINEL RAG / CONTROLLED REVIEW-1 CORPUS'); canvas.drawRightString(192*mm,10*mm,f'PAGE {doc.page}'); canvas.restoreState()
def story(doc, poisoned=False):
 st=styles(); text=doc['text']; title=doc['target_query']; attack=doc.get('attack_type','clean reference')
 parts=[x.strip() for x in text.split('. ') if x.strip()]
 out=[Paragraph('SENTINEL RAG  /  KNOWLEDGE REPORT',st['small']),Spacer(1,5),Paragraph(title,st['title']),Paragraph('Evidence-oriented reference document for the Review-1 secure retrieval demonstration',st['sub']),Spacer(1,14)]
 meta=[[Paragraph('<b>DOCUMENT ID</b><br/>%s'%doc['doc_id'],st['body']),Paragraph('<b>CLASSIFICATION</b><br/>%s'%('ADVERSARIAL VARIANT' if poisoned else 'CLEAN REFERENCE'),st['body']),Paragraph('<b>TOPIC</b><br/>%s'%attack.title(),st['body'])]]
 t=Table(meta,colWidths=[48*mm,62*mm,62*mm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#edf5f5')),('BOX',(0,0),(-1,-1),.6,colors.HexColor('#c8dddd')),('INNERGRID',(0,0),(-1,-1),.4,colors.HexColor('#d5e5e5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9)])); out += [t,Spacer(1,12),Paragraph('Executive summary',st['h']),Paragraph('This report provides a self-contained explanation of the topic in a form suitable for retrieval by a question-answering system. It is intentionally written as a realistic reference passage rather than a one-line fact, so retrieval and security checks can be evaluated on natural document structure.',st['body']),Paragraph('Detailed reference',st['h'])]
 for p in parts: out.append(Paragraph(p+'.',st['body']))
 if poisoned:
  out += [Spacer(1,5),Paragraph('Security annotation: injected payload',st['h']),Paragraph('This adversarial variant preserves substantial topical context to satisfy the retrievability condition, then inserts a confident false claim or instruction. The inserted sentence is the generation-influence payload that the detector is expected to quarantine.',st['callout']),Paragraph('<b>Injected claim:</b> '+doc['injected_claim'],st['body']),Paragraph('<b>Operations represented:</b> '+', '.join(doc['operations_applied']),st['body'])]
 digest=hashlib.sha256(text.encode()).hexdigest(); out += [Spacer(1,10),Paragraph('Provenance and integrity',st['h']),Paragraph('Document SHA-256: '+digest,st['small']),Paragraph('This file is part of a controlled academic corpus. It is not a live external source and should be treated as untrusted context by the secure RAG pipeline.',st['small'])]
 return out
def make(path,doc,poisoned): SimpleDocTemplate(str(path),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=14*mm,bottomMargin=18*mm,title=doc['target_query']).build(story(doc,poisoned),onFirstPage=footer,onLaterPages=footer)
def main():
 docs=[json.loads(x) for x in POISON.read_text(encoding='utf8').splitlines() if x.strip()]
 for d in docs: make(OUT/f"doc_{d['doc_id']}_report.pdf",d,True)
 combined=[]
 for i,d in enumerate(docs):
  combined += story(d,True)
  if i<len(docs)-1: combined.append(PageBreak())
 SimpleDocTemplate(str(OUT/'sentinel_rag_poisoned_corpus_dossier.pdf'),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=14*mm,bottomMargin=18*mm,title='Sentinel RAG Poisoned Corpus Dossier').build(combined,onFirstPage=footer,onLaterPages=footer)
 print(f'Created {len(docs)+1} detailed PDF reports in {OUT}')
if __name__=='__main__': main()
