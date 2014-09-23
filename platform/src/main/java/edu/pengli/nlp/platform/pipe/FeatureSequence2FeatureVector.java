package edu.pengli.nlp.platform.pipe;

import edu.pengli.nlp.platform.types.FeatureSequence;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;

public class FeatureSequence2FeatureVector extends Pipe {

	boolean binary;

	public FeatureSequence2FeatureVector(boolean binary) {
		this.binary = binary;
	}

	public FeatureSequence2FeatureVector() {
		this(false);
	}

	public Instance pipe(Instance carrier) {

		FeatureSequence fs = (FeatureSequence) carrier.getData();
		carrier.setData(new FeatureVector(fs, binary));
		return carrier;

	}

}
