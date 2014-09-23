package edu.pengli.nlp.platform.pipe;

import java.util.regex.Pattern;

import org.htmlparser.Parser;
import org.htmlparser.util.ParserException;

import edu.pengli.nlp.platform.types.Instance;

public abstract class CharSequenceRemoveHTML extends Pipe {

	protected Parser parser;
	protected Pattern p;

	public CharSequenceRemoveHTML() {
		parser = new Parser();
		try {
			parser.setEncoding("UTF-8");
		} catch (ParserException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public CharSequenceRemoveHTML(String regex) {
		p = Pattern.compile(regex);
	}

	public abstract Instance pipe(Instance carrier);
}
