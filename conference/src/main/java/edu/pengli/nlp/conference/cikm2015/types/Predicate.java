package edu.pengli.nlp.conference.cikm2015.types;

import java.util.ArrayList;

import edu.stanford.nlp.ling.IndexedWord;

public class Predicate extends ArrayList<IndexedWord>{
	
	IndexedWord head = null;
	
	public Predicate(){
		super();
	}
	
	public void setHead(IndexedWord head){
		this.head = head;
	}
	
	public IndexedWord getHead(){
		return head;
	}
	
	public String originaltext(){
		
		StringBuilder sb = new StringBuilder();
		for(IndexedWord tok : this){
			sb.append(tok.originalText()+" ");
		}
		return sb.toString().trim();	
	}

	
	public String lemmatext(){
		
		StringBuilder sb = new StringBuilder();
		for(IndexedWord tok : this){
			sb.append(tok.lemma()+" ");
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
