package edu.pengli.nlp.conference.cikm2012.evaluation;

import java.util.ArrayList;

import edu.pengli.nlp.conference.cikm2012.generation.BipartiteGraphRandomWalk;
import edu.pengli.nlp.conference.cikm2012.pipe.CharSequenceCleanNews;
import edu.pengli.nlp.conference.cikm2012.pipe.CharSequenceCleanTweets;
import edu.pengli.nlp.conference.cikm2012.pipe.CharSequenceExtractParagraph;
import edu.pengli.nlp.conference.cikm2012.pipe.iterator.TweetsUserIterator;
import edu.pengli.nlp.conference.cikm2012.types.GoogleNewsCorpus;
import edu.pengli.nlp.conference.cikm2012.types.TweetCorpus;
import edu.pengli.nlp.platform.algorithms.lda.CCTAModel;
import edu.pengli.nlp.platform.pipe.CharSequence2TokenSequence;
import edu.pengli.nlp.platform.pipe.Input2CharSequence;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.CharSequenceCoreNLPAnnotation;
import edu.pengli.nlp.platform.pipe.TokenSequence2FeatureSequence;
import edu.pengli.nlp.platform.pipe.TokenSequenceLowercase;
import edu.pengli.nlp.platform.pipe.TokenSequenceRemoveStopwords;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerFileIterator;
import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.FeatureSequence;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;

public class CorpusStatistics {

	public static void main(String[] args) {
		String[] topics = { "Marie_Colvin", "Poland_rail_crash",
				"Russian_presidential_election", "features_of_ipad3", "Syrian_uprising",
				"Dick_Clark", "Mexican_Drug_War", "Obama_same_sex_marriage_donation",
				"Russian_jet_crash", "Yulia_Tymoshenko_hunger_strike"};
		int[] lengthLimit = { 404, 382, 365, 401, 357, 274, 282, 325, 347, 341};
		for (int t = 0; t < topics.length; t++) {
			String topic = topics[t];
			int summaryLength = lengthLimit[t];

			// import Twitter and Google news collection

			String twiDir = "../data/EMNLP2012/Topics/Twitter";
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
			
			int ST = 0;
			for (int d = 0; d < ntc.size(); d++) {
				Instance doc = (Instance) ntc.get(d);
				InstanceList sents = (InstanceList) doc.getData();
				ST += sents.size();
			}
			

			String GoogleNewsDir = "/home/peng/Develop/Workspace/NLP/data/EMNLP2012/Topics/Google";

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
			
			int SN = 0;
			for (int d = 0; d < ngc.size(); d++) {
				Instance doc = (Instance) ngc.get(d);
				InstanceList sents = (InstanceList) doc.getData();
				SN += sents.size();
			}
			
			System.out.println(topic + " ST: " + ST + " SN: " + SN);

		}
	}
}
