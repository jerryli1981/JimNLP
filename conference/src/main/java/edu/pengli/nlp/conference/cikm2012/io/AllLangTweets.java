package edu.pengli.nlp.conference.cikm2012.io;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Random;

import twitter4j.Query;
import twitter4j.QueryResult;
import twitter4j.Status;
import twitter4j.Twitter;
import twitter4j.TwitterException;
import twitter4j.TwitterFactory;

//written by kareem
public class AllLangTweets {
	private static long SLEEP_TIME = 1000;
	private static int RPP = 1000;
	private static String LANG = "en";
	//private static String TWEETERS_FILE = "resources/tweeters.filtered.txt";
	
    public static void main(String[] args) throws IOException, TwitterException, InterruptedException {
    	System.out.println("Starting up...");
        BufferedWriter bw = new BufferedWriter(new FileWriter(String.format("tweets_%s.txt", LANG), true));

        // The factory instance is re-useable and thread safe.
        Twitter twitter = new TwitterFactory().getInstance();

        // Read the users list, if available:
        
        
        // Setup the query
        Query query = new Query(String.format("lang:%s", LANG));
        query.setSinceId(139256640917614592L);
        long queryCounter = 0;
        long tweetCounter = 0;
        long beginning = System.currentTimeMillis();
        System.out.println("Started at: " + (new Date()));
        Random r = new Random(System.currentTimeMillis());
        while (true) {
            try {
                queryCounter++;
                QueryResult result = twitter.search(query);
                List<Status> tweets = result.getTweets();
                Collections.reverse(tweets); // We need to sort each result set to maintain a total ordering that's ascending

                // Setup the next query:
                query.setSinceId(result.getMaxId());
                
                for (Status tweet : tweets) {
                    tweetCounter++;
                    bw.write(tweet.getInReplyToScreenName() + "\t"
                    	   + tweet.getId() + "\t"
                           + tweet.getId() + "\t"
                           + tweet.getGeoLocation() + "\t"
                           + tweet.getCreatedAt() + "\t"
                           + tweet.getText().replaceAll("\n", " ").replaceAll("\r", " ") + "\t"
                           + "\r\n");
                    bw.flush();
                } 
                
                if(tweetCounter%(r.nextInt(100)+1)==0) {
                    long since = System.currentTimeMillis() - beginning;
                    System.out.println("Status:");
                    System.out.println("    Queries: " + queryCounter + " queries at " + queryCounter/(since/1000) + "/sec");
                    System.out.println("    Tweets: " + tweetCounter + " tweets at " + tweetCounter/(since/1000) + "/sec");
                }
                Thread.sleep(SLEEP_TIME);
            } catch (TwitterException ex) {
                // do nothing
                System.out.println(ex.toString());
                Thread.sleep(5* Math.max(ex.getRetryAfter(), SLEEP_TIME));
            }
        }
//        bw.close();
//    	System.out.println("Completed!");
    }
}
