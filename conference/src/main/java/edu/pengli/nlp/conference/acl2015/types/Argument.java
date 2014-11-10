package edu.pengli.nlp.conference.acl2015.types;

import java.util.ArrayList;

import edu.stanford.nlp.ling.IndexedWord;

public class Argument extends ArrayList<IndexedWord>{
	
	IndexedWord head = null;
	
	public Argument(){
		super();
	}
	
	public void setHead(IndexedWord head){
		this.head = head;
	}
	
	public IndexedWord getHead(){
		return head;
	}
	
	public String toString(){
		
		StringBuilder sb = new StringBuilder();
		for(IndexedWord tok : this){
			sb.append(tok.originalText()+" ");
		}
		return sb.toString().trim();	
	}

}
