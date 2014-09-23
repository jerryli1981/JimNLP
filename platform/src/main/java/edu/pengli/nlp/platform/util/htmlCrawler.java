package edu.pengli.nlp.platform.util;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.URLConnection;



public class htmlCrawler { 
	
	public static String fetchHTML(String urlPath) throws Exception {
		
		URL url = new URL(urlPath);
		URLConnection conn = url.openConnection();
		conn.setDoOutput(true);
		InputStream inputStream = conn.getInputStream();
		InputStreamReader isr = new InputStreamReader(inputStream, "UTF-8");
		BufferedReader in = new BufferedReader(isr);
		StringBuffer sb = new StringBuffer();
		String inputLine;

		while ((inputLine = in.readLine()) != null) {
			sb.append(inputLine);
			sb.append("\n");
		}

		String strContent = sb.toString();

		return strContent;

	}

}
