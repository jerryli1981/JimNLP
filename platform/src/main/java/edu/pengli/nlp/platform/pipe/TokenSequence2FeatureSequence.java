package edu.pengli.nlp.platform.pipe;

import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.Feature;
import edu.pengli.nlp.platform.types.FeatureSequence;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.TokenSequence;

public class TokenSequence2FeatureSequence extends Pipe {

	public TokenSequence2FeatureSequence (Alphabet dataDict)
	{
		super (dataDict, null);
	}

	public TokenSequence2FeatureSequence ()
	{
		super(new Alphabet(), null);
	}

    public Instance pipe(Instance carrier) {

		TokenSequence ts = (TokenSequence) carrier.getData();
		FeatureSequence ret =new FeatureSequence (getDataAlphabet(), ts.size());
		for (int i = 0; i < ts.size(); i++) {
			Feature feat = new Feature("wordMention", ts.get(i).getMention());
			ret.add(feat.getValue());
		}
		carrier.setData(ret);
		return carrier;
		
    }

}
