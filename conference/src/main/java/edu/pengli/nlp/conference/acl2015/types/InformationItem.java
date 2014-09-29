package edu.pengli.nlp.conference.acl2015.types;

import java.util.ArrayList;

import edu.stanford.nlp.trees.TreeGraphNode;

public class InformationItem {
	
	private TreeGraphNode subject;
	private TreeGraphNode predicate;
	private ArrayList<TreeGraphNode> object;
	
	public InformationItem(TreeGraphNode subject, TreeGraphNode predicate, 
			ArrayList<TreeGraphNode> object){
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
	public ArrayList<TreeGraphNode> getObject(){
		return object;
	}
	
	public String toString(){
		if(object != null)
		{
			StringBuffer objectMention = new StringBuffer();;
			for(TreeGraphNode n : object){
				objectMention.append(n.nodeString()+" ");
			}
	
			return subject.nodeString()+"<----->"+predicate.nodeString()+"<----->"+objectMention.toString().trim();
		}else{
			return subject.nodeString()+"<----->"+predicate.nodeString();
		}
			
		
	}

}
