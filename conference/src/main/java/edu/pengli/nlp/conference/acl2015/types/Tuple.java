package edu.pengli.nlp.conference.acl2015.types;

import java.io.Serializable;
import java.util.ArrayList;

import edu.stanford.nlp.ling.CoreLabel;

public class Tuple implements Serializable{

	double confidence;
	Argument arg1;
	Predicate rel;
	Argument arg2;

	public Tuple(double confidence, Argument arg1, Predicate rel,
			Argument arg2) {
		this.arg1 = arg1;
		this.rel = rel;
		this.arg2 = arg2;
		this.confidence = confidence;
	}

	public double getConfidence() {
		return confidence;
	}

	public Argument getArg1() {
		return arg1;
	}
	
	public void setArg1(Argument argument){
		this.arg1 = argument;
	}

	public Predicate getRel() {
		return rel;
	}
	
	public void setRel(Predicate relation){
		this.rel = relation;
	}

	public Argument getArg2() {
		return arg2;
	}
	
	public void setArg2(Argument argument){
		this.arg2 = argument;
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		StringBuilder arg1Mention = new StringBuilder();
		for(CoreLabel tok : arg1){
			arg1Mention.append(tok.originalText()+" ");
		}
		
		StringBuilder arg2Mention = new StringBuilder();
		for(CoreLabel tok : arg2){
			arg2Mention.append(tok.originalText()+" ");
		}
		
		StringBuilder relMention = new StringBuilder();
		for(CoreLabel tok : rel){
			relMention.append(tok.originalText()+" ");
		}
		
		sb.append("["+arg1Mention.toString().trim()+"]").append('\t').
		append("["+relMention.toString().trim()+"]").append('\t').
		append("["+arg2Mention.toString().trim()+"]");

		return sb.toString();
	}
	
	public int hashCode() {

		return this.getArg1().hashCode()+
				this.getRel().hashCode()+
				this.getArg2().hashCode();

	}

	public boolean equals(Object compare) {

		if (compare instanceof Tuple) {
			Tuple obj = (Tuple) compare;
			if (this.getArg1().equals(obj.getArg1()) &&
					this.getRel().equals(obj.getRel()) &&
					this.getArg2().equals(this.getArg2()))
				return true;
		}
		return false;
	}
}
