package edu.pengli.nlp.conference.cikm2012.pipe;

import edu.pengli.nlp.platform.pipe.CharSequenceRemoveHTML;
import edu.pengli.nlp.platform.types.Instance;

public class CharSequenceCleanTweets extends CharSequenceRemoveHTML {

	public Instance pipe(Instance carrier) {
		Instance user = (Instance) carrier.getData();
		String content = user.getData().toString();
		content = content.replaceAll("<.*?>", "");
		content = content.replaceAll("\\[.*?\\]", "");
		content = content.replaceAll("\\(.*?\\)", "");
		content = content.replaceAll("@.*?\\s", " "); // remove @usename
		content = content.replaceAll(
				"(https?)://[a-zA-Z0-9\\-\\.]+\\.[a-zA-Z]{2,3}(/\\S*)?", "");
		content = content.replaceAll("[ \\t\\x0B\\f\\r]+", " "); //should not use \\s, because \\s contain \\n
		content = content.replaceAll("[^(\\p{Print}|(\\n))]", ""); // remove foreign
															// language like 아
															// 무셔워 Καμικάζι σε
															// αεροδρόμιο とり
															// えず友達は大丈夫みたい
		
		content = content.replaceAll("#", "");
		content = content.replaceAll("RT", "");
		content = content.replaceAll("&39;", "'");
		content = content.replaceAll("&quot;", "\"");
		content = content.replaceAll("&amp;", "");
		content = content.replaceAll("&middot", "");
		
		carrier.setData((CharSequence)content.trim());
		carrier.setSource((CharSequence)content.trim());
		return carrier;
	}

}
