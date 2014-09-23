package edu.pengli.nlp.conference.cikm2012.io;

import java.io.File;
import java.io.PrintWriter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.pengli.nlp.platform.util.FileOperation;

public class ParserHTML {

	public static void main(String[] args) {
		String topic = "Dick Clark die";
		File parentDir = new File(
				"/home/peng/Develop/Workspace/NLP/data/EMNLP2012/HTML");
		String fileName = topic+".html";
		String content = FileOperation.readContentFromFile(parentDir, fileName);
		
		Pattern p = Pattern
				.compile("(<p class=\"js-tweet-text\">).*(</p>)");
		Pattern pp = Pattern.compile("(<span class=\"username js-action-profile-name\">).*(</span>)");

		Matcher m = p.matcher(content);
		Matcher mm = pp.matcher(content);
		
		PrintWriter out = FileOperation
				.getPrintWriter(
						new File("/home/peng/Develop/Workspace/NLP/data/EMNLP2012/Topics/Twitter"),
						topic+".extractfromhtml");
		
		int i=0;
		Pattern href = Pattern.compile("(<a data-ultimate-url).*?(</a>)");
		while(m.find() && mm.find()){
			String username = mm.group();
			username = username.replaceAll("<.*?>", "");
			username = username.replaceAll("@", "");
			String tweet = m.group();	
			Matcher mmm = href.matcher(tweet);
			String link = null;
			String tmp = null;
			if(mmm.find()){
			    tmp = mmm.group();
				link = "http://" + tmp.replaceAll("<.*?>", "");
				tweet = tweet.replace(tmp, link);
			}
			
			tweet= tweet.replaceAll("<.*?>", "");
			tweet = tweet.replaceAll("#Wikinews", "");

			out.println("<UserName>" + username + "</UserName>" + "<RelTweet>"
					+ tweet+ "</RelTweet>");
			i++;
		}
		System.out.println(i);
		out.close();
		
	}

}
