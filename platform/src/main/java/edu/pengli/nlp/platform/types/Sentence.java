package edu.pengli.nlp.platform.types;

import java.io.Serializable;

import edu.stanford.nlp.trees.Tree;

public class Sentence extends TokenSequence implements Serializable {

	private static final long serialVersionUID = -7286059045547631449L;

	protected Tree parserTree;
	
	public Sentence(){
		super();
	}

	public Sentence(String tokenizedSentence) {
		super();
		String[] toks = tokenizedSentence.split(" ");
		for (int i = 0; i < toks.length; i++) {
			String tokMention = toks[i];
			this.add(tokMention);
		}
	}

	public void setParserTree(Tree parserTree) {
		this.parserTree = parserTree;
	}

	public Tree getParserTree() {
		return parserTree;
	}


	public int hashCode() {

		return this.hashCode();

	}

	public boolean equals(Object compare) {

		if (compare instanceof Feature) {
			Sentence obj = (Sentence) compare;
			if (this.equals(obj))
				return true;
		}
		return false;
	}

}
