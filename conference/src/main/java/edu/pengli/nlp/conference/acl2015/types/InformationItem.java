package edu.pengli.nlp.conference.acl2015.types;

import edu.stanford.nlp.trees.TreeGraphNode;

public class InformationItem {
	
	private TreeGraphNode subject;
	private TreeGraphNode predicate;
	private TreeGraphNode object;
	
	public InformationItem(TreeGraphNode subject, TreeGraphNode predicate, TreeGraphNode object){
		this.subject = subject;
		this.predicate = predicate;
		this.object = object;
	}
	
	public TreeGraphNode getSubject(){
		return subject;
	}
	public TreeGraphNode getPredicate(){
		return predicate;
	}
	public TreeGraphNode getObject(){
		return object;
	}
	
	public String toString(){
		if(object != null)
		return subject.toString()+":"+predicate.toString()+":"+object.toString();
		else
			return subject.toString()+":"+predicate.toString();
		
	}

}
