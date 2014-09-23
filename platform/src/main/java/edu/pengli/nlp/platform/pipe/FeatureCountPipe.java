package edu.pengli.nlp.platform.pipe;

import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.FeatureCounter;
import edu.pengli.nlp.platform.types.FeatureSequence;
import edu.pengli.nlp.platform.types.Instance;

public class FeatureCountPipe extends Pipe {


	FeatureCounter counter;

	public FeatureCountPipe() {
		super(new Alphabet(), null);

		counter = new FeatureCounter(this.getDataAlphabet());
	}
		
	public FeatureCountPipe(Alphabet dataAlphabet, Alphabet targetAlphabet) {
		super(dataAlphabet, targetAlphabet);

		counter = new FeatureCounter(dataAlphabet);
	}

	public Instance pipe(Instance instance) {
			
		if (instance.getData() instanceof FeatureSequence) {
				
			FeatureSequence features = (FeatureSequence) instance.getData();

			for (int position = 0; position < features.size(); position++) {
				counter.increment(features.getIndexAtPosition(position));
			}

		}
		else {
			throw new IllegalArgumentException("Looking for a FeatureSequence, found a " + 
											   instance.getData().getClass());
		}

		return instance;
	}
	
	public FeatureCounter getFeatureCounter(){
		return counter;
	}

}
