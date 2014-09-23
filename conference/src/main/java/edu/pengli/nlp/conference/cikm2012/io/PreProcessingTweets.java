package edu.pengli.nlp.conference.cikm2012.io;

import java.io.File;
import java.io.PrintWriter;

import edu.pengli.nlp.conference.cikm2012.pipe.CharSequenceCleanTweets;
import edu.pengli.nlp.conference.cikm2012.pipe.iterator.TweetsUserIterator;
import edu.pengli.nlp.conference.cikm2012.types.TweetCorpus;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.SentenceTokenization;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.util.FileOperation;

public class PreProcessingTweets {

	public static void main(String[] args) {
		String[] topics = { "Marie_Colvin", "Poland_rail_crash",
				"Russian_presidential_election", "features_of_ipad3",
				"Syrian_uprising", "Dick_Clark", "Mexican_Drug_War",
				"Obama_same_sex_marriage_donation", "Russian_jet_crash",
				"Yulia_Tymoshenko_hunger_strike" };

		for (int t = 0; t < topics.length; t++) {
			String topic = topics[t];

			String twiDir = "../data/EMNLP2012/Topics/Twitter";
			TweetsUserIterator tUserIter = new TweetsUserIterator(twiDir,
					String.valueOf(topic));

			// one tweet as one sentence, so do not need sentence detection.
			PipeLine pipeLine = new PipeLine();
			pipeLine.addPipe(new CharSequenceCleanTweets());
			pipeLine.addPipe(new SentenceTokenization());
			TweetCorpus tc = new TweetCorpus(tUserIter, pipeLine);
			
			PrintWriter out = FileOperation.getPrintWriter(new File(twiDir+"/cleaned"), topic+".clean");

			for (Instance u : tc) {

				String path = twiDir + "/" + topic;
				String content = (String) u.getData();
				if(content.equals("") || content.equals(" "))continue;
                String user = (String) u.getName();
                user = user.replace("_T", "");
				out.println("<UserName>"+user+"</UserName><RelTweet>"+content+"</RelTweet>");
			}
			out.close();
		}
		System.out.println("done");

	}

}
