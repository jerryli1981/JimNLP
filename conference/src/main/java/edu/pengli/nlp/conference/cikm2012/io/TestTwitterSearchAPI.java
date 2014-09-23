package edu.pengli.nlp.conference.cikm2012.io;

import java.io.File;
import java.io.PrintWriter;
import java.util.List;

import edu.pengli.nlp.platform.util.FileOperation;
import twitter4j.Query;
import twitter4j.Query.ResultType;
import twitter4j.QueryResult;
import twitter4j.Status;
import twitter4j.TwitterResponse;
import twitter4j.Twitter;
import twitter4j.TwitterException;
import twitter4j.TwitterFactory;

public class TestTwitterSearchAPI {

	public static void main(String[] args) throws TwitterException {

		String topic = "Dick Clark";
		Twitter twitter = new TwitterFactory().getInstance();
		Query query = new Query(topic);
		query.setLang("en");
        query.setResultType(ResultType.mixed); //Example Values: mixed, recent, popular
        
//		query.setSinceId(170957701206114305L); //170957701206114305 (02-18)
		PrintWriter out = FileOperation
				.getPrintWriter(
						new File("/home/peng/Develop/Workspace/NLP/data/EMNLP2012/Topics/Twitter"),
						topic+".tapi");
		int counts = 0;
		try {
			QueryResult result = twitter.search(query);
			List<Status> tweets = result.getTweets();
			counts +=tweets.size();
			for (Status tweet : tweets) {

				out.println("<UserName>" + tweet.getInReplyToScreenName() + "</UserName>" + "<RelTweet>"
						+ tweet.getText() + "</RelTweet>");
			}
		} catch (TwitterException te) {
			te.printStackTrace();
			System.out.println("Failed to search tweets: " + te.getMessage());
			System.exit(-1);
		}
		out.close();
		System.out.println(counts);
		System.out.println("done");
	}
}
