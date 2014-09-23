package edu.pengli.nlp.platform.pipe;

import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.Token;
import edu.pengli.nlp.platform.types.TokenSequence;


public class TokenSequenceLowercase extends Pipe{
	
	public Instance pipe (Instance carrier)
	{
		TokenSequence ts = (TokenSequence) carrier.getData();
		for (int i = 0; i < ts.size(); i++) {
			Token t = ts.get(i);
			t.setMention(t.getMention().toLowerCase());
		}
		return carrier;
	}
}
