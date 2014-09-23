package edu.pengli.nlp.conference.cikm2012.pipe;

import edu.pengli.nlp.platform.pipe.CharSequenceRemoveHTML;
import edu.pengli.nlp.platform.types.Instance;

public class CharSequenceCleanNews extends CharSequenceRemoveHTML {

	public Instance pipe(Instance carrier) {
		String content = carrier.getData().toString();
		
		carrier.setData((CharSequence) content.trim());
		carrier.setSource((CharSequence) content.trim());
		return carrier;
	}

}
