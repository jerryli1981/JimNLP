package edu.pengli.nlp.platform.types;

import java.io.Serializable;


public class Token implements Serializable{

	private static final long serialVersionUID = -6262838125694448417L;
	protected String mention;
	protected String pos;

	public Token(Instance inst) {
		this.mention = inst.getData().toString();
	}

	public Token(String mention) {
		this.mention = mention;
	}

	public Token(String mention, String pos) {
		this.mention = mention;
		this.pos = pos;
	}

	public String getMention() {
		return mention;
	}

	public String getPOS() {
		return pos;
	}

	public void setMention(String mention) {
		this.mention = mention;
	}

	public int hashCode(){
		
		return mention.hashCode();
		
	}
	public boolean equals(Object compare) {

		if (compare instanceof Token) {
			Token obj = (Token) compare;
			if (this.mention.equals(obj.getMention()))
				return true;
		}
		return false;
	}
	
	public String toString(){
		return mention;
	}
}
