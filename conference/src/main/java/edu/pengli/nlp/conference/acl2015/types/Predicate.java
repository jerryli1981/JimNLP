package edu.pengli.nlp.conference.acl2015.types;

import java.util.ArrayList;

import edu.stanford.nlp.ling.CoreLabel;

public class Predicate extends ArrayList<CoreLabel>{
	
	public Predicate(){
		super();
	}
	
	public String toString(){
		
		StringBuilder sb = new StringBuilder();
		for(CoreLabel tok : this){
			sb.append(tok.lemma()+" ");
		}
		return sb.toString().trim();	
	}
	
}
