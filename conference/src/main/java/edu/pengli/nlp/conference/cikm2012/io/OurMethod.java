package edu.pengli.nlp.conference.cikm2012.io;

import java.io.File;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

import edu.pengli.nlp.conference.cikm2012.evaluation.RougeEvaluationWrapper;
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
import edu.pengli.nlp.platform.pipe.SentenceTokenization;
import edu.pengli.nlp.platform.pipe.TokenSequence2FeatureSequence;
import edu.pengli.nlp.platform.pipe.TokenSequenceLowercase;
import edu.pengli.nlp.platform.pipe.TokenSequenceRemoveStopwords;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerFileIterator;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.FileOperation;

public class OurMethod {

	public static void main(String[] args) {
		String[] topics = { "Marie_Colvin", "Poland_rail_crash",
				"Russian_presidential_election", "features_of_ipad3", "Syrian_uprising",
				"Dick_Clark", "Mexican_Drug_War", "Obama_same_sex_marriage_donation",
				"Russian_jet_crash", "Yulia_Tymoshenko_hunger_strike"};
	//	int[] lengthLimit = { 404, 382, 365, 401, 357, 274, 282, 325, 347, 341};
	//	int[] lengthLimit = { 450, 420, 400, 450, 400, 320, 320, 370, 390, 380};
		int[] lengthLimit = { 500, 500, 500, 500, 500, 500, 500, 500, 500, 500};
	//	int[] lengthLimit = { 490, 490, 490, 490, 490, 490, 490, 490, 490, 490};
		double recall = 0.0;
		int runTime = 10;
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

				String twiDir = "../data/CIKM2012/Topics/Twitter/cleaned";
				TweetsUserIterator tUserIter = new TweetsUserIterator(twiDir,
						String.valueOf(topic+".clean"));

				// one tweet as one sentence, so do not need sentence detection.
				PipeLine pipeLine = new PipeLine();
				pipeLine.addPipe(new CharSequenceCleanTweets());
				pipeLine.addPipe(new SentenceTokenization());
				TweetCorpus tc = new TweetCorpus(tUserIter, pipeLine);

				pipeLine = new PipeLine();
				pipeLine.addPipe(new CharSequence2TokenSequence());
				pipeLine.addPipe(new TokenSequenceLowercase());
				pipeLine.addPipe(new TokenSequenceRemoveStopwords());
				pipeLine.addPipe(new TokenSequence2FeatureSequence());

				TweetCorpus ntc = new TweetCorpus(tc, pipeLine);
				colls.add(ntc);

				String GoogleNewsDir = "../data/CIKM2012/Topics/Google";

				OneInstancePerFileIterator fIter = new OneInstancePerFileIterator(
						GoogleNewsDir + "/" + String.valueOf(topic));

				pipeLine = new PipeLine();
				pipeLine.addPipe(new Input2CharSequence("UTF-8"));
				pipeLine.addPipe(new CharSequenceCleanNews());
				pipeLine.addPipe(new SentenceTokenization());
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

				BipartiteGraphRandomWalk bgrw = new BipartiteGraphRandomWalk(
						ccta);
				bgrw.contructTransitionMatrix(ccta);
				summaries.put(topic,bgrw.mutualReinforcement2(summaryLength));
				bgrw.outputSummary(topic, iter);
			}

			HashMap map = RougeEvaluationWrapper.run(iter, "T");
			recall += (Double) map.get("R");
			String outputDir = "../data/CIKM2012/Output/summary";
			if (maxR <= (Double) map.get("R")) {
				maxR = (Double) map.get("R");
				for (int t = 0; t < topics.length; t++) {
					String topic = topics[t];           	
					PrintWriter out = FileOperation.getPrintWriter(new File(
							outputDir), String.valueOf(topic+".MAN."+iter));
					
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

			}
		}
        NumberFormat nf = new DecimalFormat();
		nf.setMaximumFractionDigits(5);
		nf.setMinimumFractionDigits(5);
		System.out.println("R:" + nf.format(recall / runTime));
		
		System.out.println("OurMethod is done");
		//R:0.06166 P:0.05874 F:0.06017
		//R:0.06256 P:0.07406 F:0.06779
		//R:0.06971 P:0.07182 F:0.07075
		//R:0.06434 P:0.06761 F:0.06593
		//R:0.06434 P:0.05594 F:0.05985
		//R:0.07104 P:0.06640 F:0.06864
		//R:0.07507 P:0.06829 F:0.07152
		//R:0.06756 P:0.07915 F:0.07288
		
		//Rouge-2:  0.35854
		//Rouge-2:  0.38994
		//Rouge-2:  0.43664
		
		//R:0.35127
	}

}
