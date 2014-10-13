package edu.pengli.nlp.conference.cikm2012.io;

import java.io.File;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

import edu.pengli.nlp.conference.cikm2012.generation.BipartiteGraphRandomWalk;
import edu.pengli.nlp.conference.cikm2012.pipe.CharSequenceCleanNews;
import edu.pengli.nlp.conference.cikm2012.pipe.CharSequenceCleanTweets;
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
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.FileOperation;

public class Baseline_2_LM {

	/**
	 * measure sentence pair complementary using language model
	 */
	
	public static void main(String[] args) {
		String[] topics = { "Marie_Colvin", "Poland_rail_crash",
				"Russian_presidential_election", "features_of_ipad3",
				"Syrian_uprising", "Dick_Clark", "Mexican_Drug_War",
				"Obama_same_sex_marriage_donation", "Russian_jet_crash",
				"Yulia_Tymoshenko_hunger_strike" };
		int[] lengthLimit = { 404, 377, 365, 401, 357, 274, 282, 325, 337, 341 };
		int[] lengthLimit_n = { 198, 192, 151, 207, 161, 180, 178, 231, 201, 249 };
		int[] lengthLimit_t = { 206, 185, 214, 194, 196, 94, 104, 94, 136, 92 };
		
		double recall = 0.0;
		double precision = 0.0;
		double fmeasure = 0.0;
		int runTime = 1;
		double maxR = 0.0;
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
				pipeLine.addPipe(new CharSequenceCoreNLPAnnotation());
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
				pipeLine.addPipe(new CharSequenceCoreNLPAnnotation());
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
				double gammaX = 20;
				double gammaL = 20;
				int numTopics = 4;
				int numIters = 100;
				int numAspect = 2; // always set 2 for complementary reason

				CCTAModel ccta = new CCTAModel(numTopics, numAspect, alpha,
						beta, gammaX, gammaL, delta, numIters);
				ccta.initEstimate(colls);
				
				BipartiteGraphRandomWalk bgrw = new BipartiteGraphRandomWalk(
						ccta);
				bgrw.contructTransitionMatrix();
				summaries.put(topic,bgrw.mutualReinforcement2(summaryLength));
				bgrw.outputSummary(topic, iter);
			}

/*			HashMap map = RougeEvaluationWrapper.run();
			recall += (Double) map.get("R");
			precision += (Double) map.get("P");
			fmeasure += (Double) map.get("F");
			String outputDir = "../data/EMNLP2012/Output/summary";
			if (maxR <= (Double) map.get("R")) {
				maxR = (Double) map.get("R");
				for (int t = 0; t < topics.length; t++) {
					String topic = topics[t];           	
					PrintWriter out = FileOperation.getPrintWriter(new File(
							outputDir), String.valueOf(topic+".MAN"));
					
					HashMap<Instance, ArrayList<Instance>> pairs = summaries.get(topic);
					Set<Instance> keys = pairs.keySet();
					Iterator i = keys.iterator();

					while(i.hasNext()){
						Instance ns = (Instance) i.next();
						ArrayList<Instance> al = pairs.get(ns);
						for(Instance inst : al){
							out.println("<SP"+ " "+ "score="+" "+" >");
							out.println("<NS>"+ns.getSource()+"</NS>");
							out.println("<TW>"+inst.getSource()+"</TW>");
							out.println("</SP>");
							out.println();
							out.println();
						}
					}
					out.close();
				}

			}*/
		}

/*		NumberFormat nf = new DecimalFormat();
		nf.setMaximumFractionDigits(5);
		nf.setMinimumFractionDigits(5);
		System.out.println("R:" + nf.format(recall / runTime) + " " + "P:"
				+ nf.format(precision / runTime) + " " + "F:"
				+ nf.format(fmeasure / runTime));*/
		System.out.println("Baseline 2 LM is done");

	}

}
