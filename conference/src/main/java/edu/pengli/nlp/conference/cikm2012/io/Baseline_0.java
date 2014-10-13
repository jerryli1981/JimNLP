package edu.pengli.nlp.conference.cikm2012.io;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.HashMap;

import edu.pengli.nlp.conference.cikm2012.generation.LexRanking;
import edu.pengli.nlp.conference.cikm2012.pipe.CharSequenceCleanNews;
import edu.pengli.nlp.conference.cikm2012.pipe.CharSequenceCleanTweets;
import edu.pengli.nlp.conference.cikm2012.pipe.iterator.TweetsUserIterator;
import edu.pengli.nlp.conference.cikm2012.types.GoogleNewsCorpus;
import edu.pengli.nlp.conference.cikm2012.types.TweetCorpus;
import edu.pengli.nlp.platform.pipe.CharSequence2TokenSequence;
import edu.pengli.nlp.platform.pipe.Input2CharSequence;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.CharSequenceCoreNLPAnnotation;
import edu.pengli.nlp.platform.pipe.TokenSequence2FeatureSequence;
import edu.pengli.nlp.platform.pipe.TokenSequenceLowercase;
import edu.pengli.nlp.platform.pipe.TokenSequenceRemoveStopwords;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerFileIterator;


public class Baseline_0 {

	/*
	 * we performed LexRank on the two corpus
	 * 
	 */

	public static void main(String[] args) throws InterruptedException {
		String[] topics = { "Marie_Colvin", "Poland_rail_crash",
				"Russian_presidential_election", "features_of_ipad3",
				"Syrian_uprising", "Dick_Clark", "Mexican_Drug_War",
				"Obama_same_sex_marriage_donation", "Russian_jet_crash",
				"Yulia_Tymoshenko_hunger_strike" };
//		int[] lengthLimit = { 404, 377, 365, 401, 357, 274, 282, 325, 337, 341 };
		int[] lengthLimit_n = { 198, 192, 151, 207, 161, 180, 178, 231, 201, 249 };
		int[] lengthLimit_t = { 206, 185, 214, 194, 196, 94, 104, 94, 136, 92 };

		double recall_T = 0.0;
		double recall_N = 0.0;
		int runTime = 10;
		
		for (int iter = 0; iter < runTime; iter++) {
			System.out.println(iter);
			for (int i = 0; i < topics.length; i++) {
				String topic = topics[i];
				int summaryLength_n = lengthLimit_n[i];
				int summaryLength_t = lengthLimit_t[i];
				// import Twitter and Google news collection

				String twiDir = "../data/CIKM2012/Topics/Twitter";
				TweetsUserIterator tUserIter = new TweetsUserIterator(twiDir,
						String.valueOf(topic));

				// one tweet as one sentence, so do not need sentence detection.
				PipeLine pipeLine = new PipeLine();
				pipeLine.addPipe(new CharSequenceCleanTweets());
				pipeLine.addPipe(new CharSequenceCoreNLPAnnotation());
				TweetCorpus tc = new TweetCorpus(tUserIter, pipeLine);

				pipeLine = new PipeLine();
				pipeLine.addPipe(new CharSequence2TokenSequence());
				pipeLine.addPipe(new TokenSequenceLowercase());
				pipeLine.addPipe(new TokenSequenceRemoveStopwords());
				pipeLine.addPipe(new TokenSequence2FeatureSequence());

				TweetCorpus ntc = new TweetCorpus(tc, pipeLine);
				
				LexRanking lr = new LexRanking(ntc);
				lr.run_tweets(summaryLength_t);
				lr.outputSummary(topic, iter+".T");
				
				

				String GoogleNewsDir = "../data/CIKM2012/Topics/Google";

				OneInstancePerFileIterator fIter = new OneInstancePerFileIterator(
						GoogleNewsDir + "/" + String.valueOf(topic));

				pipeLine = new PipeLine();
				pipeLine.addPipe(new Input2CharSequence("UTF-8"));
				pipeLine.addPipe(new CharSequenceCleanNews());
				pipeLine.addPipe(new CharSequenceCoreNLPAnnotation());
				GoogleNewsCorpus gc = new GoogleNewsCorpus(fIter, pipeLine);

				pipeLine = new PipeLine();
				pipeLine.addPipe(new CharSequence2TokenSequence());
				pipeLine.addPipe(new TokenSequenceLowercase());
				pipeLine.addPipe(new TokenSequenceRemoveStopwords());
				pipeLine.addPipe(new TokenSequence2FeatureSequence());

				GoogleNewsCorpus ngc = new GoogleNewsCorpus(gc, pipeLine);

				lr = new LexRanking(ngc);
				lr.run_news(summaryLength_n);
				lr.outputSummary(topic, iter+".N");
		
			}
/*			HashMap map_T = RougeEvaluationWrapper.run(iter, "T");
			System.out.println("Twitter Recall is "+(Double) map_T.get("R"));
			recall_T += (Double) map_T.get("R");
			
			HashMap map_N = RougeEvaluationWrapper.run(iter, "N");
			System.out.println("News Recall is "+(Double) map_N.get("R"));
			recall_N += (Double) map_N.get("R");*/

		}
		
		NumberFormat nf = new DecimalFormat();
		nf.setMaximumFractionDigits(5);
		nf.setMinimumFractionDigits(5);
		System.out.println("Average Twitter Recall is:" + nf.format(recall_T / runTime));	
		System.out.println("Average News Recall is:" + nf.format(recall_N / runTime));	
		
		System.out.println("Baseline_0 is done");
	}
}
