package edu.pengli.nlp.platform.pipe;


import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.FeatureCounter;
import edu.pengli.nlp.platform.types.FeatureSequence;
import edu.pengli.nlp.platform.types.Instance;
import gnu.trove.map.hash.TIntIntHashMap;

//this is count how many docs contains features
public class FeatureDocFreqPipe extends Pipe {
	
	FeatureCounter counter;
	int numInstances;

	public FeatureDocFreqPipe() {
		super(new Alphabet(), null);
		counter = new FeatureCounter(super.getDataAlphabet());
		numInstances = 0;
	}
		
	public FeatureDocFreqPipe(Alphabet dataAlphabet, Alphabet targetAlphabet) {
		super(dataAlphabet, targetAlphabet);

		counter = new FeatureCounter(dataAlphabet);
		numInstances = 0;
	}

	public Instance pipe(Instance instance) {
		
		TIntIntHashMap localCounter = new TIntIntHashMap();

		if (instance.getData() instanceof FeatureSequence) {
				
			FeatureSequence features = (FeatureSequence) instance.getData();

			for (int position = 0; position < features.size(); position++) {
				localCounter.adjustOrPutValue(features.getIndexAtPosition(position), 1, 1);
			}

		}
		else {
			throw new IllegalArgumentException("Looking for a FeatureSequence, found a " + 
											   instance.getData().getClass());
		}

		for (int feature: localCounter.keys()) {
			counter.increment(feature);
		}

		numInstances++;

		return instance;
	}
	
	public FeatureCounter getFeatureDocCounter(){
		return counter;
	}
}
