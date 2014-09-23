package edu.pengli.nlp.platform.types;

public class Label {
	
	Object entry;
	LabelAlphabet dictionary;
	int index;

	protected Label() {
		throw new IllegalStateException(
				"Label objects can only be created by their Alphabet.");
	}

	/**
	 * You should never call this directly. New Label objects are created
	 * on-demand by calling LabelAlphabet.lookupIndex(obj).
	 */
	Label(Object entry, LabelAlphabet dict, int index) {
		this.entry = entry;
		this.dictionary = dict;
		assert (dict.lookupIndex(entry, false) == index);
		this.index = index;
	}

	public LabelAlphabet getLabelAlphabet() {
		return (LabelAlphabet) dictionary;
	}

	public int getIndex() {
		return index;
	}

	public Alphabet getAlphabet() {
		return dictionary;
	}

	public Alphabet[] getAlphabets() {
		return new Alphabet[] { dictionary };
	}

	public Object getEntry() {
		return entry;
	}

	public String toString() {
		return entry.toString();
	}

}
