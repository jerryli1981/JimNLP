package edu.pengli.nlp.conference.acl2015.types;

import java.io.Serializable;

import edu.stanford.nlp.util.CoreMap;

public class Pattern implements Serializable{
	
	String arg1;
	String predicate;
	String arg2;
	CoreMap sent;
	Tuple t;
	
	public Pattern(String arg1, String predicate, String arg2, CoreMap sent, Tuple t){
		this.arg1 = arg1;
		this.predicate = predicate;
		this.arg2 = arg2;
		this.sent = sent;
		this.t = t;
	}
	
	public String getArg1(){
		return arg1;
	}
	
	public String getRel(){
		return predicate;
	}
	
	public String getArg2(){
		return arg2;
	}
	
	public CoreMap getCoreMap(){
		return sent;
	}
	
	public Tuple getTuple(){
		return t;
	}
	
	public String toString(){
		//Pattern p = new Pattern(arg1Head.get(0).ner(), 
		//t.getRel().toString(), arg2Head.get(0).ner(), sent);
		return arg1.toUpperCase()+" "+predicate+" "+arg2.toUpperCase();
	}
	
	public int hashCode() {

		return this.getArg1().hashCode()+
				this.getRel().hashCode()+
				this.getArg2().hashCode();

	}

	public boolean equals(Object compare) {

		if (compare instanceof Pattern) {
			Pattern obj = (Pattern) compare;
			if (this.getArg1().equals(obj.getArg1()) &&
					this.getRel().equals(obj.getRel()) &&
					this.getArg2().equals(this.getArg2()))
				return true;
		}
		return false;
	}

}
