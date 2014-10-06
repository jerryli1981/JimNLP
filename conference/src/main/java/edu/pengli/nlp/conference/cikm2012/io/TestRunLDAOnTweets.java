package edu.pengli.nlp.conference.cikm2012.io;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;

import edu.pengli.nlp.conference.cikm2012.pipe.CharSequenceCleanTweets;
import edu.pengli.nlp.conference.cikm2012.pipe.iterator.TweetsUserIterator;
import edu.pengli.nlp.conference.cikm2012.types.TweetCorpus;
import edu.pengli.nlp.platform.algorithms.lda.TwitterLDAModel;
import edu.pengli.nlp.platform.pipe.CharSequence2TokenSequence;
import edu.pengli.nlp.platform.pipe.Noop;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.CharSequenceTokenizationAndSentencesplit;
import edu.pengli.nlp.platform.pipe.TokenSequence2FeatureSequence;
import edu.pengli.nlp.platform.pipe.TokenSequenceLowercase;
import edu.pengli.nlp.platform.pipe.TokenSequenceRemoveStopwords;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerLineIterator;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.FileOperation;

public class TestRunLDAOnTweets {

	public static void main(String[] args) {

		String topic  = "Marie_Colvin";
		

		String twiDir = "../data/EMNLP2012/Topics/Twitter";
		TweetsUserIterator tUserIter = new TweetsUserIterator(twiDir,
				String.valueOf(topic));
		
		//one tweet as one sentence, so do not need sentence detection.
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

		double alpha = 10; // 50/numTopics
		double beta = 0.01;
		double gamma = 20;
		int numTopics = 5;
		int numIters = 100;
		TwitterLDAModel model = new TwitterLDAModel(numTopics, alpha, beta,
				gamma, numIters);

		model.initEstimate(ntc);

		model.estimate();

		model.predict_labels(ntc);

		PrintWriter out = FileOperation.getPrintWriter(new File(
				"../data/EMNLP2012/Output"),
				topic);
		model.output_labels(out, ntc);
		
		model.output_model();
		
		String outputDir = "../data/EMNLP2012/";
		String outputName = "newsTopicsTwitter";
		model.writeModel(outputDir, outputName);

		System.out.println("done");

	}

}
