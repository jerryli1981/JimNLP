package edu.pengli.nlp.conference.acl2015.types;

import java.io.Serializable;

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.util.CoreMap;

public class Tuple implements Serializable{

	double confidence;
	Argument arg1;
	Predicate rel;
	Argument arg2;
	CoreMap annotatedSentence;

	public Tuple(double confidence, Argument arg1, Predicate rel,
			Argument arg2, CoreMap sent) {
		this.arg1 = arg1;
		this.rel = rel;
		this.arg2 = arg2;
		this.confidence = confidence;
		this.annotatedSentence = sent;
	}
	
	public Tuple(Argument arg1, Predicate rel,
			Argument arg2, CoreMap sent) {
		this.arg1 = arg1;
		this.rel = rel;
		this.arg2 = arg2;
		this.annotatedSentence = sent;
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
	
	public CoreMap getAnnotatedSentence(){
		return annotatedSentence;
	}
	
	public String originaltext() {
		StringBuilder sb = new StringBuilder();
	
		sb.append("["+arg1.originaltext()+"]").append('\t').
		append("["+rel.originaltext()+"]").append('\t').
		append("["+arg2.originaltext()+"]");

		return sb.toString();
	}
	
	public String getSentenceRepresentation() {
		StringBuilder sb = new StringBuilder();
	
		sb.append(arg1.originaltext()+" ").
		append(rel.originaltext()+" ").append(arg2.originaltext());

		return sb.toString();
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
