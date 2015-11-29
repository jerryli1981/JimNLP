package edu.pengli.nlp.conference.cikm2015.pipe;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;

import org.apache.http.HttpException;
import org.apache.http.HttpResponse;
import org.apache.http.NameValuePair;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.utils.URLEncodedUtils;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.message.BasicNameValuePair;
import org.apache.http.util.EntityUtils;
import org.apache.log4j.Logger;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import edu.pengli.nlp.platform.util.RankMap;

//currently doesn't work
public class DBpediaTagger {

	private static HttpClient client;
	private static Logger LOG;

	public DBpediaTagger() {
		LOG = Logger.getLogger(this.getClass());
		client = new DefaultHttpClient();
	}

	private static String labelVoting(ArrayList<String> toks) {

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
		Iterator iter = map.keySet().iterator();
		StringBuilder xx = new StringBuilder();
		while(iter.hasNext()){
			xx.append((String)iter.next() + " ");
		}
		
//		String ret = (String) map.keySet().iterator().next();
		return xx.toString();
	}

	public static String NameEntityRecognition(String sentMention) throws ClientProtocolException, IOException{
		
		String API_URL = "http://spotlight.dbpedia.org/";
		String CONFIDENCE = "0.0";
		String SUPPORT = "0";
		
		List<NameValuePair> params = new ArrayList<NameValuePair>();
//		params.add(new BasicNameValuePair("support", SUPPORT));
//		params.add(new BasicNameValuePair("confidence", CONFIDENCE));
		params.add(new BasicNameValuePair("text", sentMention));
		URLEncodedUtils.format(params, "UTF-8");

		String url = API_URL + "rest/annotate/" + "?"
				+ URLEncodedUtils.format(params, "UTF-8");
		
		HttpResponse httpResponse = client.execute(new HttpGet(url));
			

		JSONArray entities = null;
		
		ArrayList<String> labels = new ArrayList<String>();

		try {
			
			JSONObject resultJSON = new JSONObject(EntityUtils.toString(httpResponse
					.getEntity()));
			entities = resultJSON.getJSONArray("Resources");
			for (int i = 0; i < entities.length(); i++) {
				JSONObject entity = entities.getJSONObject(i);
				String mention = entity.getString("@types");
				if(mention.equals(""))
					return null;
				String[] tokens = mention.split(",");
				for(String l : tokens){
					labels.add(l);
				}
			}
			
		} catch (JSONException e) {
			//e.printStackTrace();
		}

		String label = labelVoting(labels);
		return label;

	}

	public static void main(String[] args) throws ClientProtocolException, IOException {
		DBpediaTagger obj = new DBpediaTagger();
		System.out.println(obj.NameEntityRecognition("Helen"));

	}

}
