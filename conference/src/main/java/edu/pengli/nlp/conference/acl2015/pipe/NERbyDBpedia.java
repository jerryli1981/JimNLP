package edu.pengli.nlp.conference.acl2015.pipe;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.util.HashMap;
import java.util.LinkedHashMap;

import org.apache.commons.httpclient.DefaultHttpMethodRetryHandler;
import org.apache.commons.httpclient.Header;
import org.apache.commons.httpclient.HttpClient;
import org.apache.commons.httpclient.HttpException;
import org.apache.commons.httpclient.HttpMethod;
import org.apache.commons.httpclient.HttpStatus;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.commons.httpclient.params.HttpMethodParams;
import org.apache.log4j.Logger;

import edu.pengli.nlp.platform.util.RankMap;

public class NERbyDBpedia {
	
	private final static String API_URL = "http://spotlight.dbpedia.org/";
	private static final double CONFIDENCE = 0.0;
	private static final int SUPPORT = 0;
	private static HttpClient client;
	private Logger LOG;
	
	public NERbyDBpedia(){
		LOG = Logger.getLogger(this.getClass());
		client = new HttpClient();
	}
	
	private String labelVoting(String mention) {

		String[] toks = mention.split(",");
		HashMap<String, Integer> counts = new HashMap<String, Integer>();
		for (String s : toks) {
			if (s.startsWith("DBpedia")) {
				String label = s.replaceAll("DBpedia:", "");
				label = label.toLowerCase();
				if (!counts.containsKey(label)) {
					counts.put(label, 1);
				} else {
					int c = counts.get(label);
					counts.put(label, ++c);
				}

			} else if (s.startsWith("Freebase")) {
				String tmp = s.replaceAll("Freebase:/", "");
				String[] ls = tmp.split("/");
				for (String label : ls) {
					label = label.toLowerCase();
					if (!counts.containsKey(label)) {
						counts.put(label, 1);
					} else {
						int c = counts.get(label);
						counts.put(label, ++c);
					}
				}
			}
		}
		LinkedHashMap map = RankMap.sortHashMapByValues(counts, false);
		String ret = (String) map.keySet().iterator().next();
		return ret;
	}
	
	private String request(HttpMethod method) {

		String response = null;

		// Provide custom retry handler is necessary
		method.getParams().setParameter(HttpMethodParams.RETRY_HANDLER,
				new DefaultHttpMethodRetryHandler(3, false));

		try {
			// Execute the method.
			int statusCode = client.executeMethod(method);

			if (statusCode != HttpStatus.SC_OK) {
				LOG.error("Method failed: " + method.getStatusLine());
			}

			// Read the response body.
			byte[] responseBody = method.getResponseBody(); // TODO Going to
															// buffer response
															// body of large or
															// unknown size.
															// Using
															// getResponseBodyAsStream
															// instead is
															// recommended.

			// Deal with the response.
			// Use caution: ensure correct character encoding and is not binary
			// data
			response = new String(responseBody);

		} catch (HttpException e) {
			LOG.error("Fatal protocol violation: " + e.getMessage());
			System.out.println("Fatal protocol violation");
			System.exit(0);

		} catch (IOException e) {
			LOG.error("Fatal transport error: " + e.getMessage());
			LOG.error(method.getQueryString());
			System.out.println("Fatal transport error");
			System.exit(0);
		} finally {
			// Release the connection.
			method.releaseConnection();
		}
		return response;

	}

	private String NameEntityRecognition(String sentMention) {
		String spotlightResponse = null;

		GetMethod getMethod;

		try {
			getMethod = new GetMethod(API_URL + "rest/annotate/?"
					+ "confidence=" + CONFIDENCE + "&support=" + SUPPORT
					+ "&text=" + URLEncoder.encode(sentMention, "utf-8"));

			getMethod
					.addRequestHeader(new Header("Accept", "application/json"));

			spotlightResponse = request(getMethod);
			assert spotlightResponse != null;

		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return spotlightResponse;

	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
