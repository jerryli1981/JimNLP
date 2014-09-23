package edu.pengli.nlp.conference.cikm2012.io;

import java.io.File;
import java.io.PrintWriter;

import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.htmlCrawler;
import edu.pengli.nlp.platform.util.htmlParser;

public class GenerateTestDocumentsFromGoogleNews {
	
	public static void main(String[] args) throws Exception{
		String topicID = "5";
		File parentDir = new File("/home/peng/Develop/Workspace/NLP/data/EMNLP2012/Topics/Google News/"+topicID);
		String url = "http://articles.latimes.com/2011/jan/25/world/la-fg-mexico-clinton-20110125";
		String content = htmlCrawler.fetchHTML(url);
		PrintWriter out = FileOperation.getPrintWriter(parentDir, "latimes");
		String title = htmlParser.getTitle(content);
		out.println("<TITLE>"+title+"</TITLE>");
		out.println();
		out.println("<HEADLINE>"+"</HEADLINE>");
		out.println();
		String par = htmlParser.getParagraph(content);
		String[] line = par.split("\n");
		for(int i=0; i<line.length; i++){
			String l = line[i];
			if(l.length()==0 || l.split(" ").length <=3)continue;
			out.println("<P>"+line[i]+"</P>");
			out.println();
		}
		out.close();
		System.out.println("done");
	}

}
