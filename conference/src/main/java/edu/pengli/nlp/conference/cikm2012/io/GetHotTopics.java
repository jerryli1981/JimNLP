package edu.pengli.nlp.conference.cikm2012.io;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.api.client.http.HttpRequestInitializer;
import com.google.api.client.http.HttpTransport;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.jackson2.*;
import com.google.api.services.customsearch.Customsearch;
import com.google.api.services.customsearch.model.Search;
import com.google.api.services.customsearch.model.Search.SearchInformation;

import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.RankMap;

public class GetHotTopics {

	public static void main(String[] args) throws IOException {
		
	//	String apiKey = "AIzaSyCmoQxCtBfpDZbkGAXVkwNymfx2_PN6V9w";
		String apiKey = "AIzaSyDTe2n6tJ3boIiNmuJDteqSgOgwaunBtxA";
		String cx = "009063622200028507080:jcothwteflu";

		// Set up the HTTP transport and JSON factory
		HttpTransport httpTransport = new NetHttpTransport();
		JsonFactory jsonFactory = new JacksonFactory();
		
		HttpRequestInitializer httpRequestInitializer = null;
		

		Customsearch cs = new Customsearch(httpTransport, jsonFactory, httpRequestInitializer);
		
		String dir = "../data/EMNLP2012";
		String fileName = "wikinews.list";
		BufferedReader in = FileOperation.getBufferedReader(new File(dir), fileName);
		String input = null;
		String regex = "http://.*?\\s?<\\/T>";
		Pattern p = Pattern.compile(regex);
		
		HashMap<String, Long> map = new HashMap<String, Long>();
		int i=0;
		while((input=in.readLine())!=null && i <=50){
			Matcher m = p.matcher(input);
			String link = null;
			if(m.find()){
				link = m.group().replaceAll("<\\/T>", "");
			}
			String topic = input.replace(link, "").trim();
			
			com.google.api.services.customsearch.Customsearch.Cse.List list = cs.cse().list
					(topic);
			list.setCx(cx);
			list.setKey(apiKey);
			Search results = list.execute();
			SearchInformation si = results.getSearchInformation();
			Long totalResults = si.getTotalResults();
			map.put(topic, totalResults);
			System.out.println(i++);
		}

		PrintWriter out = FileOperation
				.getPrintWriter(
						new File("../data/EMNLP2012/"), "rankedList");
		
		LinkedHashMap ranked = RankMap.sortHashMapByValues(map, false);
		Iterator iter = ranked.keySet().iterator();
		while(iter.hasNext()){
			String t = (String) iter.next();
			Long r = (Long) ranked.get(t);
			out.println(t+"    "+r);
		}
		out.close();
	}

}
