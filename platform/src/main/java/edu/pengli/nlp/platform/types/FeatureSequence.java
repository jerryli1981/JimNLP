package edu.pengli.nlp.platform.types;

import java.io.Serializable;


public class FeatureSequence implements Sequence, AlphabetCarrying, Serializable {

	Alphabet dictionary;
	int[] features;
	int length;

	public FeatureSequence(Alphabet dict, int capacity) {
		dictionary = dict;
		features = new int[capacity];
		length = 0;
	}

	public void add(Object key) {
		features[length++] = dictionary.lookupIndex(key);

	}

	public final int size() {
		return length;
	}
	
	public final int getIndexAtPosition (int pos)
	{
		return features[pos];
	}


	public Object get(int pos) {
		return dictionary.lookupObject(features[pos]);
	}

	public int[] toFeatureIndexSequence() {
		int[] feats = new int[length];
		System.arraycopy(features, 0, feats, 0, length);
		return feats;
	}

	public int[] toSortedFeatureIndexSequence() {
		int[] feats = this.toFeatureIndexSequence();
		java.util.Arrays.sort(feats);
		return feats;
	}

	public Alphabet getAlphabet() {
		// TODO Auto-generated method stub
		return dictionary;
	}
}
