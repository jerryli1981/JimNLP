package edu.pengli.nlp.conference.cikm2012.io;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

import edu.pengli.nlp.conference.cikm2012.generation.BipartiteGraphRandomWalk;
import edu.pengli.nlp.conference.cikm2012.generation.simulateGoldStandard;
import edu.pengli.nlp.conference.cikm2012.pipe.CharSequenceCleanNews;
import edu.pengli.nlp.conference.cikm2012.pipe.CharSequenceCleanTweets;
import edu.pengli.nlp.conference.cikm2012.pipe.iterator.TweetsUserIterator;
import edu.pengli.nlp.conference.cikm2012.types.GoogleNewsCorpus;
import edu.pengli.nlp.conference.cikm2012.types.TweetCorpus;
import edu.pengli.nlp.platform.algorithms.lda.CCTAModel;
import edu.pengli.nlp.platform.pipe.CharSequence2TokenSequence;
import edu.pengli.nlp.platform.pipe.Input2CharSequence;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.CharSequenceTokenizationAndSentencesplit;
import edu.pengli.nlp.platform.pipe.TokenSequence2FeatureSequence;
import edu.pengli.nlp.platform.pipe.TokenSequenceLowercase;
import edu.pengli.nlp.platform.pipe.TokenSequenceRemoveStopwords;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerFileIterator;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.TimeWait;

public class Baseline_3_com {

	public static void main(String[] args) {
		String[] topics = { "Marie_Colvin", "Poland_rail_crash",
				"Russian_presidential_election", "features_of_ipad3", "Syrian_uprising",
				"Dick_Clark", "Mexican_Drug_War", "Obama_same_sex_marriage_donation",
				"Russian_jet_crash", "Yulia_Tymoshenko_hunger_strike"};
		int[] lengthLimit = { 404, 382, 365, 401, 357, 274, 282, 325, 347, 341};
		

		int runTime = 10;
		for (int iter = 0; iter < runTime; iter++) {
			System.out.println(iter);
			HashMap<String, HashMap<Instance, ArrayList<Instance>>> summaries = 
					new HashMap<String, HashMap<Instance, ArrayList<Instance>>>();
			for (int t = 0; t < topics.length; t++) {
				String topic = topics[t];
				int summaryLength = lengthLimit[t];

				// import Twitter and Google news collection
				ArrayList<InstanceList> colls = new ArrayList<InstanceList>();

				String twiDir = "../data/EMNLP2012/Topics/Twitter";
				TweetsUserIterator tUserIter = new TweetsUserIterator(twiDir,
						String.valueOf(topic));

				// one tweet as one sentence, so do not need sentence detection.
				PipeLine pipeLine = new PipeLine();
				pipeLine.addPipe(new CharSequenceCleanTweets());
				pipeLine.addPipe(new CharSequenceTokenizationAndSentencesplit());
				TweetCorpus tc = new TweetCorpus(tUserIter, pipeLine);

				pipeLine = new PipeLine();
				pipeLine.addPipe(new CharSequence2TokenSequence());
				pipeLine.addPipe(new TokenSequenceLowercase());
				pipeLine.addPipe(new TokenSequenceRemoveStopwords());
				pipeLine.addPipe(new TokenSequence2FeatureSequence());

				TweetCorpus ntc = new TweetCorpus(tc, pipeLine);
				colls.add(ntc);

				String GoogleNewsDir = "../data/EMNLP2012/Topics/Google";

				OneInstancePerFileIterator fIter = new OneInstancePerFileIterator(
						GoogleNewsDir + "/" + String.valueOf(topic));

				pipeLine = new PipeLine();
				pipeLine.addPipe(new Input2CharSequence("UTF-8"));
				pipeLine.addPipe(new CharSequenceCleanNews());
				pipeLine.addPipe(new CharSequenceTokenizationAndSentencesplit());
				GoogleNewsCorpus gc = new GoogleNewsCorpus(fIter, pipeLine);

				pipeLine = new PipeLine();
				pipeLine.addPipe(new CharSequence2TokenSequence());
				pipeLine.addPipe(new TokenSequenceLowercase());
				pipeLine.addPipe(new TokenSequenceRemoveStopwords());
				pipeLine.addPipe(new TokenSequence2FeatureSequence());

				GoogleNewsCorpus ngc = new GoogleNewsCorpus(gc, pipeLine);
				colls.add(ngc);

				double alpha = 10;
				double beta = 0.01;
				double delta = 10;
				double gammaX = 10;
				double gammaL = 10;
				int numTopics = 4;
				int numIters = 100;
				int numAspect = 2; 

				CCTAModel ccta = new CCTAModel(numTopics, numAspect, alpha,
						beta, gammaX, gammaL, delta, numIters);
				ccta.initEstimate(colls);
				ccta.estimate();

				simulateGoldStandard sgs = new simulateGoldStandard(ccta, ngc, ntc);
				summaries.put(topic,sgs.run_com(summaryLength));			
				sgs.outputSummary(topic, iter);
			}
		}
		
		System.out.println("Baseline 3 is done");

	}

}
