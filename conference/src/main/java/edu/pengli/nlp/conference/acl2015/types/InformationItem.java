package edu.pengli.nlp.conference.acl2015.types;

import java.util.ArrayList;

import edu.stanford.nlp.ling.IndexedWord;
//this is pattern
public class InformationItem {
	
	private IndexedWord subject;
	private IndexedWord predicate;
	private ArrayList<IndexedWord> object;
	
	public InformationItem(IndexedWord subject, IndexedWord predicate, 
			ArrayList<IndexedWord> object){
		this.subject = subject;
		this.predicate = predicate;
		this.object = object;
	}
	
	public IndexedWord getSubject(){
		return subject;
	}
	public IndexedWord getPredicate(){
		return predicate;
	}
	public ArrayList<IndexedWord> getObject(){
		return object;
	}
	
	public String toString(){
		if(object != null)
		{
			StringBuffer objectMention = new StringBuffer();;
			for(IndexedWord n : object){
				objectMention.append(n.originalText()+" ");
			}
	
			return "S:"+subject.originalText()+"<----->"+"P:"+predicate.originalText()+"<----->"+"O:"+objectMention.toString().trim();
		}else{
			
			return "S:"+subject.originalText()+"<----->"+"P:"+predicate.originalText();
		}
			
		
	}

}
