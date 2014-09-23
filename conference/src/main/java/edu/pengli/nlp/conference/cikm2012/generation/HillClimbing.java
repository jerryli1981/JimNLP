package edu.pengli.nlp.conference.cikm2012.generation;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;

import edu.pengli.nlp.conference.cikm2012.types.Summary;
import edu.pengli.nlp.platform.algorithms.lda.CCTAModel;
import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.FileOperation;

public class HillClimbing {

	Summary newsSummary;
	Summary twitterSummary;
	
	InstanceList sentList_N;
	InstanceList sentList_T;
	
	CCTAModel ccta_T;
	CCTAModel ccta_N;
	
	int summaryLength_T;
	int summaryLength_N;

	public HillClimbing(CCTAModel ccta_T, CCTAModel ccta_N, int summaryLength_T, int summaryLength_N) {

		this.summaryLength_T = summaryLength_T;
		this.summaryLength_N = summaryLength_N;
		this.ccta_T = ccta_T;
		this.ccta_N = ccta_N;
		newsSummary = new Summary();
		twitterSummary = new Summary();
		InstanceList instances_T = ccta_T.getInstanceList();
		InstanceList instances_N = ccta_N.getInstanceList();
		sentList_N = new InstanceList(null);
		sentList_T = new InstanceList(null);
		for (Instance doc : instances_T) {
			InstanceList sents = (InstanceList) doc.getSource();
			for (Instance sent : sents) {
					sentList_T.add(sent);
			}
		}
		
		for (Instance doc : instances_N) {
			InstanceList sents = (InstanceList) doc.getSource();
			for (Instance sent : sents) {
					sentList_N.add(sent);
				
			}
		}
	}

	public void initializeSummary() {
		
		Random r_T = new Random();
		Random r_G = new Random();

		int totalWords_T = 0;
		int totalWords_N = 0;
		while (true) {
			int t = r_T.nextInt(sentList_T.size());
			Instance sent_T = sentList_T.get(t);
			totalWords_T += ((String) sent_T.getSource()).split(" ").length;
			if (totalWords_T >= summaryLength_T) {
				totalWords_T -= ((String) sent_T.getSource()).split(" ").length;
				break;
			}
			sentList_T.remove(t);
			twitterSummary.add(sent_T);

			int n = r_G.nextInt(sentList_N.size());
			Instance sent_N = sentList_N.get(n);
			totalWords_N += ((String) sent_N.getSource()).split(" ").length;
			if (totalWords_N >= summaryLength_N) {
				totalWords_N -= ((String) sent_N.getSource()).split(" ").length;
				break;
			}
			sentList_N.remove(n);
			newsSummary.add(sent_N);
		}
		
		//recovery sentList_N and sentList_T
		InstanceList instances_T = ccta_T.getInstanceList();
		InstanceList instances_N = ccta_N.getInstanceList();
		sentList_N = new InstanceList(null);
		sentList_T = new InstanceList(null);
		for (Instance doc : instances_T) {
			InstanceList sents = (InstanceList) doc.getSource();
			for (Instance sent : sents) {
					sentList_T.add(sent);
			}
		}
		
		for (Instance doc : instances_N) {
			InstanceList sents = (InstanceList) doc.getSource();
			for (Instance sent : sents) {
					sentList_N.add(sent);
				
			}
		}
		
	}

	private ArrayList<ArrayList<Summary>> getNeighbors(Summary currentSummary) {

		ArrayList<ArrayList<Summary>> ret = new ArrayList<ArrayList<Summary>>();

		InstanceList sents_T = new InstanceList(null);
		InstanceList sents_N = new InstanceList(null);
		for (Instance sent : currentSummary) {
			String name = (String) sent.getName();
			if (name.endsWith("_T")) {
				sents_T.add(sent);
			} else if (name.endsWith("_G")) {
				sents_N.add(sent);
			}
		}

		InstanceList remainSentences_T = new InstanceList(null);
		remainSentences_T.addAll(sents_T);

		InstanceList remainSentences_N = new InstanceList(null);
		remainSentences_N.addAll(sents_N);

		ArrayList<Summary> neighbors_T = new ArrayList<Summary>();
		for (int i = 0; i < sents_T.size(); i++) {
			Summary neighbor = new Summary();
			Instance s = sents_T.remove(i);
			neighbor.addAll(sents_T);
			for (int j = 0; j < remainSentences_T.size(); j++) {
				neighbor.add(remainSentences_T.get(j));
				neighbors_T.add(neighbor);
				neighbor.remove(neighbor.size() - 1);
			}
			sents_T.add(i, s);
		}

		ArrayList<Summary> neighbors_N = new ArrayList<Summary>();
		for (int i = 0; i < sents_N.size(); i++) {
			Summary neighbor = new Summary();
			Instance s = sents_N.remove(i);
			neighbor.addAll(sents_N);
			for (int j = 0; j < remainSentences_N.size(); j++) {
				neighbor.add(remainSentences_N.get(j));
				neighbors_N.add(neighbor);
				neighbor.remove(neighbor.size() - 1);
			}
			sents_N.add(i, s);
		}

		ret.add(neighbors_T);
		ret.add(neighbors_N);

		return ret;
	}
	
	private double KL_Divergence(InstanceList corpus, Summary summary){
		double score = 0;
		double alpha = 10;
		double beta = 0.01;
		double delta = 10;
		double gammaX = 20.0;
		double gammaL = 20.0;
		int numTopics = 6;
		int numIters = 100;
		int numAspect = 2; // always set 2 for complementary reason

		CCTAModel ccta_summary =new CCTAModel(numTopics, numAspect, alpha,
				beta, gammaX, gammaL, delta, numIters);

		ArrayList<InstanceList> colls = new ArrayList<InstanceList>();
		HashMap<String, InstanceList> map = new HashMap<String, InstanceList>();
		Alphabet dict_sum = null;
		for (Instance sent : summary) {
			String name = (String) sent.getName();
			String docName = name.split("_")[1];
			if (!map.containsKey(docName)) {
				InstanceList sents = new InstanceList(null);
				sents.add(sent);
				map.put(docName, sents);
			} else {
				InstanceList sents = map.get(docName);
				sents.add(sent);
				map.put(docName, sents);
			}
			dict_sum = sent.getDataAlphabet();
		}
		Set<String> keys = map.keySet();
		Iterator<String> iter = keys.iterator();
		InstanceList corp_sum = new InstanceList(null);
		while (iter.hasNext()) {
			String docName = iter.next();
			Instance doc = new Instance(map.get(docName), null, docName, null);
			corp_sum .add(doc);
		}

		corp_sum.setDataAlphabet(dict_sum);
		colls.add(corp_sum);
		ccta_summary.initEstimate(colls);
		ccta_summary.estimate();
		
		CCTAModel ccta_Corpus = null;
        if(corpus.equals(sentList_N)){
        	ccta_Corpus = ccta_N;
        }else if(corpus.equals(sentList_T)){
        	ccta_Corpus = ccta_T;
        }
		double[][] k_v_prob = ccta_Corpus.getGeneralTopicWordDistribution();
		Alphabet dict_Corpus = ccta_Corpus.getDictionary();

		double[][] k_v_prob_summary = ccta_summary
				.getGeneralTopicWordDistribution();
		Alphabet dict_summary = ccta_summary.getDictionary();

		for (int k = 0; k < k_v_prob.length; k++) {
			Object[] entries = dict_summary.toArray();
			double[] values_sum = new double[entries.length];
			for (int i = 0; i < entries.length; i++) {
				Object entry = entries[i];
				double pro_s = k_v_prob_summary[k][dict_summary
						.lookupIndex(entry)];
				int length = k_v_prob[k].length;
				if(length <= dict_Corpus.lookupIndex(entry)) continue;
				double pro = k_v_prob[k][dict_Corpus.lookupIndex(entry)];
				score += pro_s * Math.log((pro_s / pro)) / Math.log(2);
			}
		}

		return score;
	}

	private double L(Summary sum_T, Summary sum_N) {

		return KL_Divergence(sentList_N, sum_N) + KL_Divergence(sentList_N, sum_T) +
				KL_Divergence(sentList_T, sum_N) + KL_Divergence(sentList_T, sum_T);
	//	return KL_Divergence(sentList_N, sum_T) + KL_Divergence(sentList_T, sum_N);
		
	}

	public void run() {

		double maxVal = Double.NEGATIVE_INFINITY;

		Summary currentSummary = new Summary();
		currentSummary.addAll(newsSummary);
		currentSummary.addAll(twitterSummary);

		int iterTimes = 3; // 3 is the best
		for (int i = 0; i < iterTimes; i++) {

			ArrayList<ArrayList<Summary>> neighborsList = getNeighbors(currentSummary);
			ArrayList<Summary> neighbors_T = neighborsList.get(0);
			ArrayList<Summary> neighbors_N = neighborsList.get(1);
			double nextEval = Double.NEGATIVE_INFINITY;
			Summary nextSummary = new Summary();
			int count = 0;
			for (Summary sum_T : neighbors_T) {
				if(count++ > 2) break; 
				for (Summary sum_N : neighbors_N) {
					if( sum_T.length() <= summaryLength_T || sum_N.length() <=summaryLength_N){
//					if( (sum_T.length() + sum_N.length()) <= (summaryLength_N+summaryLength_T)){
						double tmpScore = L(sum_T, sum_N);
						if (tmpScore > nextEval) {
							nextSummary.addAll(sum_T);
							nextSummary.addAll(sum_N);
							nextEval = tmpScore;
						}
					}
				}
			}

			if (nextEval <= maxVal) {
				maxVal = nextEval;
				currentSummary.clear();
				for(Instance sent: nextSummary){
					currentSummary.add(sent);
				}

			}
		}
		newsSummary.clear();
		twitterSummary.clear();
		for(Instance sent : currentSummary){
			if(sent.getName().toString().endsWith("_T")){
				twitterSummary.add(sent);
			}else if(sent.getName().toString().endsWith("_G")){
				newsSummary.add(sent);
			}
		}
	}
	
	public void outputSummary(String topic, String iter) {

		String outputDir = "/home/peng/Develop/Workspace/NLP/data/CIKM2012/Output/summary";

		PrintWriter out_t = FileOperation.getPrintWriter(new File(outputDir),
				String.valueOf(topic+"."+iter+".T"));
		
		PrintWriter out_n = FileOperation.getPrintWriter(new File(outputDir),
				String.valueOf(topic+"."+iter+".N"));
		
		out_t.println("<html>");
		out_t.println("<body bgcolor=\"white\">");
		
		out_n.println("<html>");
		out_n.println("<body bgcolor=\"white\">");

		int i = 1;
		for (Instance sent : twitterSummary) {
			out_t.println("<a name=\"" + i + "\">[" + i + "]</a> "
					+ "<a href=\"#" + i + "\" " + "id=" + i + ">"
					+ sent.getSource() + "</a>");
			i++;
		}
		
		i = 1;
		for (Instance sent : newsSummary) {
			out_n.println("<a name=\"" + i + "\">[" + i + "]</a> "
					+ "<a href=\"#" + i + "\" " + "id=" + i + ">"
					+ sent.getSource() + "</a>");
			i++;
		}
		
		
		out_t.println("</body>");
		out_t.println("</html>");
		out_n.println("</body>");
		out_n.println("</html>");
		out_t.close();
		out_n.close();

	}

}
