package edu.pengli.nlp.platform.pipe;

import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.Token;
import edu.pengli.nlp.platform.types.TokenSequence;
import edu.pengli.nlp.platform.util.CharSequenceLexer;

public class CharSequence2TokenSequence extends Pipe {

	CharSequenceLexer lexer;

	public CharSequence2TokenSequence() {
		lexer = new CharSequenceLexer("[\\p{L}\\p{N}_]+");
	}

	public Instance pipe(Instance carrier) {

		CharSequence string = (CharSequence) carrier.getData();
		TokenSequence ts = new TokenSequence();
		lexer.setCharSequence(string);
		while (lexer.hasNext()) {
			lexer.next();
			String mention = string.toString().substring(
					lexer.getStartOffset(), lexer.getEndOffset());
			Token tok = new Token(mention);
			ts.add(tok);
		}
		carrier.setData(ts);
		carrier.setSource(string);

		return carrier;

	}
	
}
