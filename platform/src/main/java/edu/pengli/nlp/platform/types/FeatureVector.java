package edu.pengli.nlp.platform.types;

import java.io.Serializable;

public class FeatureVector extends SparseVector implements Serializable, AlphabetCarrying{
	
	Alphabet dictionary;
	
	public FeatureVector (FeatureSequence fs, boolean binary)
	{
		super (fs.toSortedFeatureIndexSequence(), false, false, true, binary);
		this.dictionary = (Alphabet) fs.getAlphabet();
	}
	
	public FeatureVector(int[] idx, double[] vals){
		super(idx, vals);
	}
	

	public Alphabet getAlphabet() {
		// TODO Auto-generated method stub
		return dictionary;
	}

}
