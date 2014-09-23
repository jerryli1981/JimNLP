package edu.pengli.nlp.platform.types;

import java.io.Serializable;
import java.util.ArrayList;


public class TokenSequence extends ArrayList<Token> implements Sequence, Serializable{


	public TokenSequence () {
		super();
	}

	public TokenSequence (int capacity) {
		super (capacity);
	}
		
	public String toString() {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < this.size(); i++) {
			String tt = get(i).toString();
			tt.replaceAll("\n","");
			if (i > 0){
				sb.append(" ");
			}
			sb.append(tt);
		}
		return sb.toString();
	}
	
	public void add(String string) {
		add(new Token(string));
	}

}

