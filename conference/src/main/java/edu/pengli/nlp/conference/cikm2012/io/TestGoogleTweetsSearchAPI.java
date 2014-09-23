package edu.pengli.nlp.conference.cikm2012.io;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import com.google.api.client.http.HttpRequestInitializer;
import com.google.api.client.http.HttpTransport;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.jackson2.JacksonFactory;
import com.google.api.services.customsearch.Customsearch;
import com.google.api.services.customsearch.model.Result;
import com.google.api.services.customsearch.model.Search;
import com.google.api.services.customsearch.model.Search.SearchInformation;

import edu.pengli.nlp.platform.util.FileOperation;

public class TestGoogleTweetsSearchAPI {
	
	public static void main(String[] args) throws IOException {
	    String apiKey = "AIzaSyCmoQxCtBfpDZbkGAXVkwNymfx2_PN6V9w";
		String cx = "009063622200028507080:jcothwteflu";

		// Set up the HTTP transport and JSON factory
		HttpTransport httpTransport = new NetHttpTransport();
		JsonFactory jsonFactory = new JacksonFactory();
		HttpRequestInitializer httpRequestInitializer = null;
		

		Customsearch cs = new Customsearch(httpTransport, jsonFactory, httpRequestInitializer);

		String topic = "Dick Clark";
		com.google.api.services.customsearch.Customsearch.Cse.List list = cs.cse().list
				(topic);
		PrintWriter out = FileOperation
				.getPrintWriter(
						new File("/home/peng/Develop/Workspace/NLP/data/EMNLP2012/Topics/Twitter"),
						topic+".gapi");
		list.setCx(cx);
		list.setKey(apiKey);
		for(int i=1; i<100; i+=10){
			list.setStart((long) i);
			Search results = list.execute();
			List<Result> items = results.getItems();	
			for(Result re : items){
				String id = re.getCacheId();
				String snippet = re.getHtmlSnippet().replaceAll("<.*?>", "");
				out.println("<UserName>" + id+ "</UserName>" + "<RelTweet>"
						+ snippet + "</RelTweet>");

			}
		}
		out.close();
		System.out.println("done");

	}
}
