package edu.pengli.nlp.conference.acl2015.types;

import java.io.Serializable;

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.util.CoreMap;

public class Pattern extends Tuple implements Serializable{
	
	public Pattern(Argument arg1, Predicate rel, Argument arg2,  CoreMap sent){
		super(arg1, rel, arg2, sent);
	}
		
	//for clustering
	public String toGeneralizedForm(){
		return arg1.getHead().ner().toUpperCase()+" "+rel.getHead().originalText()+" "
		+arg2.getHead().ner().toUpperCase();
	}
	
	//for fusion
	public String toSpecificForm(){

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
	
	public String toString(){
		try {		
			throw new NoSuchMethodException();
		} catch (NoSuchMethodException e) {
			// TODO Auto-generated catch block
			System.out.println("For debug");
		}finally{
			
			return originaltext();
		}
	}
	
}
