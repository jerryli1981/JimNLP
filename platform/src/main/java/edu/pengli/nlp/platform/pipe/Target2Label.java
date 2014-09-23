package edu.pengli.nlp.platform.pipe;

import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.Label;
import edu.pengli.nlp.platform.types.LabelAlphabet;

public class Target2Label extends Pipe {

	public Target2Label(Alphabet dataAlphabet, LabelAlphabet labelAlphabet) {
		super(dataAlphabet, labelAlphabet);
	}

	public Target2Label() {
		this(null, new LabelAlphabet());
	}

	public Target2Label(LabelAlphabet labelAlphabet) {
		this(null, labelAlphabet);
	}

	public Instance pipe(Instance carrier) {
		if (carrier.getTarget() != null) {
			if (carrier.getTarget() instanceof Label)
				throw new IllegalArgumentException("Already a label.");
			LabelAlphabet ldict = (LabelAlphabet) getTargetAlphabet();
			carrier.setTarget(ldict.lookupLabel(carrier.getTarget()));
		}
		return carrier;
	}

}
