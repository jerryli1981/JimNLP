package edu.pengli.nlp.conference.acl2015.pipe;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.List;

import org.apache.http.HttpResponse;
import org.apache.http.NameValuePair;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.utils.URLEncodedUtils;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.message.BasicNameValuePair;
import org.apache.http.util.EntityUtils;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class FreebaseTagger {
	
	String api_key;
	String service_url;

	public FreebaseTagger() {
		
		api_key = "AIzaSyDTe2n6tJ3boIiNmuJDteqSgOgwaunBtxA";
		service_url = "https://www.googleapis.com/freebase/v1/search";

	}

	public ArrayList<String> search(String query) throws IOException,
			JSONException {

		List<NameValuePair> params = new ArrayList<NameValuePair>();
		params.add(new BasicNameValuePair("query", query));
		params.add(new BasicNameValuePair("key", api_key));
		URLEncodedUtils.format(params, "UTF-8");

		String url = service_url + "?"
				+ URLEncodedUtils.format(params, "UTF-8");

		HttpClient httpclient = new DefaultHttpClient();

		HttpResponse httpResponse = httpclient.execute(new HttpGet(url));
		JSONObject response = new JSONObject(EntityUtils.toString(httpResponse
				.getEntity()));
		JSONArray results = (JSONArray) response.get("result");
		ArrayList<String> labels = new ArrayList<String>();
		for (int i = 0; i < results.length(); i++) {
			JSONObject entity = results.getJSONObject(i);
			if(!entity.toString().contains("\"notable\":"))
					continue;
			JSONObject notable = entity.getJSONObject("notable");
			String label = notable.getString("name");
			labels.add(label);
		}
		
		return labels;

	}

	public static void main(String[] args) throws IOException, JSONException {
		String query = "person";
		FreebaseTagger ft = new FreebaseTagger();
		System.out.println(ft.search(query));

	}


}
