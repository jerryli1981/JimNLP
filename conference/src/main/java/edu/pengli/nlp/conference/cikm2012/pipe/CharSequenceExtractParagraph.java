package edu.pengli.nlp.conference.cikm2012.pipe;


import org.htmlparser.Node;
import org.htmlparser.NodeFilter;
import org.htmlparser.Parser;
import org.htmlparser.filters.NodeClassFilter;
import org.htmlparser.tags.ParagraphTag;
import org.htmlparser.util.NodeList;
import org.htmlparser.util.ParserException;

import edu.pengli.nlp.platform.pipe.CharSequenceRemoveHTML;
import edu.pengli.nlp.platform.types.Instance;

public class CharSequenceExtractParagraph extends CharSequenceRemoveHTML {

	public Instance pipe(Instance carrier) {
		String content = carrier.getData().toString();  	
		StringBuffer paragraph = new StringBuffer();
		parser = Parser.createParser(content, "UTF-8");
		NodeFilter[] nflist = new NodeFilter[2];
		nflist[0] = new NodeClassFilter(ParagraphTag.class);
		NodeList nodelist;
		try {
			nodelist = parser.parse(nflist[0]);
			Node[] nodes = nodelist.toNodeArray();
			for (int j = 0; j < nodes.length; j++) {
				if (nodes[j] instanceof ParagraphTag) {
					paragraph.append(nodes[j].toPlainTextString());
					paragraph.append("\n");

				}
			}
		} catch (ParserException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		String cont = paragraph.toString();
		carrier.setData((CharSequence)cont.trim());
		return carrier;
	}

}
