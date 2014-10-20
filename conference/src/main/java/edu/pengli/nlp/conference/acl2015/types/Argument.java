package edu.pengli.nlp.conference.acl2015.types;

import java.util.ArrayList;

import edu.stanford.nlp.ling.CoreLabel;

public class Argument extends ArrayList<CoreLabel>{
	
	
	public Argument(){
		super();
	}
	
	public String toString(){
		
		StringBuilder sb = new StringBuilder();
		for(CoreLabel tok : this){
			sb.append(tok.originalText()+" ");
		}
		return sb.toString().trim();	
	}

}
