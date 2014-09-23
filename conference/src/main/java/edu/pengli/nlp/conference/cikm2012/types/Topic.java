package edu.pengli.nlp.conference.cikm2012.types;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

//here topic means query
public class Topic {

	public String id;
	public String title;
	public String querytweettime;

	private static Pattern idPattern = Pattern.compile("<num>.*?</num>");
	private static Pattern titlePattern = Pattern.compile("<title>.*?</title>");
	private static Pattern querytweettimePattern = Pattern
			.compile("<querytweettime>.*?</querytweettime>");

	public Topic(String description) {
		Matcher idM = idPattern.matcher(description);
		if (idM.find())
			id = idM.group().replaceAll("<.*?>", "").trim();

		Matcher titleM = titlePattern.matcher(description);
		if (titleM.find())
			title = titleM.group().replaceAll("<.*?>", "").trim();

		Matcher querytweettimeM = querytweettimePattern.matcher(description);
		if (querytweettimeM.find())
			querytweettime = querytweettimeM.group().replaceAll("<.*?>", "").trim();
		
	}
	
	public String getID(){
		return id;
	}
	public String getTitle(){
		return title;
	}

}
