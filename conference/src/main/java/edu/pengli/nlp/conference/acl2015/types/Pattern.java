package edu.pengli.nlp.conference.acl2015.types;

import java.io.Serializable;

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.util.CoreMap;

//pattern should not extends from tuple, change pattern will affect tuple
public class Pattern implements Serializable{
	
	Argument arg1;
	Predicate rel;
	Argument arg2;
	Tuple t;
	
	//deep copy
	public Pattern(Argument argument1, Predicate pre, Argument argument2, Tuple t){
		this.t = t;
		arg1 = new Argument();
		for(IndexedWord iw : argument1){
			IndexedWord niw = new IndexedWord();
			niw.setDocID(iw.docID());
			niw.setIndex(iw.index());
			niw.setSentIndex(iw.sentIndex());
			niw.setOriginalText(iw.originalText());
			niw.setNER(iw.ner());
			niw.setTag(iw.tag());
			arg1.add(niw);
		}
		IndexedWord headArg1 = new IndexedWord();
		headArg1.setDocID(argument1.getHead().docID());
		headArg1.setIndex(argument1.getHead().index());
		headArg1.setSentIndex(argument1.getHead().sentIndex());
		headArg1.setOriginalText(argument1.getHead().originalText());
		headArg1.setNER(argument1.getHead().ner());
		headArg1.setTag(argument1.getHead().tag());
		arg1.setHead(headArg1);
		
		rel = new Predicate();
		for(IndexedWord iw : pre){
			IndexedWord niw = new IndexedWord();
			niw.setDocID(iw.docID());
			niw.setIndex(iw.index());
			niw.setSentIndex(iw.sentIndex());
			niw.setOriginalText(iw.originalText());
			niw.setNER(iw.ner());
			niw.setTag(iw.tag());
			rel.add(niw);
		}
		IndexedWord relHead = new IndexedWord();
		relHead.setDocID(pre.getHead().docID());
		relHead.setIndex(pre.getHead().index());
		relHead.setSentIndex(pre.getHead().sentIndex());
		relHead.setOriginalText(pre.getHead().originalText());
		relHead.setNER(pre.getHead().ner());
		relHead.setTag(pre.getHead().tag());
		rel.setHead(relHead);
		
		
		arg2 = new Argument();
		for(IndexedWord iw : argument2){
			IndexedWord niw = new IndexedWord();
			niw.setDocID(iw.docID());
			niw.setIndex(iw.index());
			niw.setSentIndex(iw.sentIndex());
			niw.setOriginalText(iw.originalText());
			niw.setNER(iw.ner());
			niw.setTag(iw.tag());
			arg2.add(niw);
		}
		IndexedWord headArg2 = new IndexedWord();
		headArg2.setDocID(argument2.getHead().docID());
		headArg2.setIndex(argument2.getHead().index());
		headArg2.setSentIndex(argument2.getHead().sentIndex());
		headArg2.setOriginalText(argument2.getHead().originalText());
		headArg2.setNER(argument2.getHead().ner());
		headArg2.setTag(argument2.getHead().tag());
		arg2.setHead(headArg2);
	}
	
	public Tuple getTuple(){
		return t;
	}
	
	public Argument getArg1() {
		return arg1;
	}
	
	public Predicate getRel() {
		return rel;
	}
	
	public Argument getArg2() {
		return arg2;
	}
				
	//for fusion and clustering
	public String toString(){

		StringBuilder sb = new StringBuilder();

		IndexedWord headArg1 = arg1.getHead();
		for(IndexedWord w : arg1){
			if(w.index() == headArg1.index()){
				sb.append(w.ner().toUpperCase()+" ");
			}else
				sb.append(w.originalText()+" ");
		}
		
		sb.append(rel.originaltext()+" ");
		
		
		IndexedWord headArg2 = arg2.getHead();
		for(IndexedWord w : arg2){
			if(w.index() == headArg2.index()){
				sb.append(w.ner().toUpperCase()+" ");
			}else
				sb.append(w.originalText()+" ");
		}
		
		return sb.toString().trim();
	}
		
}
