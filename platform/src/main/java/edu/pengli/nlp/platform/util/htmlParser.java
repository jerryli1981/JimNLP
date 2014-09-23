package edu.pengli.nlp.platform.util;

import java.io.IOException;

import org.htmlparser.filters.*;
import org.htmlparser.*;
import org.htmlparser.tags.*;
import org.htmlparser.util.*;

public class htmlParser {

	public static String getTitle(String content) {
		String title = null;
		Parser parser;
		parser = Parser.createParser(content, "UFT-8");
		NodeFilter[] nflist = new NodeFilter[2];
		nflist[0] = new NodeClassFilter(ParagraphTag.class);
		nflist[1] = new NodeClassFilter(TitleTag.class);
		OrFilter moreFilter = new OrFilter(nflist);
		NodeList nodelist;
		try {
			nodelist = parser.extractAllNodesThatMatch(moreFilter);
			Node[] nodes = nodelist.toNodeArray();

			for (int j = 0; j < nodes.length; j++) {
				if (nodes[j] instanceof TitleTag) {
					String regex = " - Wikipedia, the free encyclopedia";
					String ss = nodes[j].toPlainTextString();
					title = ss.replaceAll(regex, "");
					title = title.replaceAll("\\(.*\\)", "").trim();

				}

			}
		} catch (ParserException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return title;
	}

	public static String getParagraph(String sPage){

		StringBuffer paragraph = new StringBuffer();
		Parser parser;
		parser = Parser.createParser(sPage, "UTF-8");
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

		return paragraph.toString();
	}

	public static String getTable(String sPage) throws ParserException,
			IOException {

		Parser parser;
		parser = Parser.createParser(sPage, "UTF-8");
		NodeFilter[] nflist = new NodeFilter[1];

		nflist[0] = new NodeClassFilter(TableTag.class);

		NodeList nodelist = parser.parse(nflist[0]);
		Node[] nodes = nodelist.toNodeArray();
		if (nodes.length == 0) {
			return null;
		} else {

			return nodes[0].toHtml();
		}

	}

}
