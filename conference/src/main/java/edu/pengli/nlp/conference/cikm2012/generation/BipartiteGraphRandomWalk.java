package edu.pengli.nlp.conference.cikm2012.generation;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Set;

import edu.pengli.nlp.conference.cikm2012.types.Summary;
import edu.pengli.nlp.platform.algorithms.lda.CCTAModel;
import edu.pengli.nlp.platform.algorithms.ranking.LexRank;
import edu.pengli.nlp.platform.pipe.CharSequence2TokenSequence;
import edu.pengli.nlp.platform.pipe.FeatureIDFPipe;
import edu.pengli.nlp.platform.pipe.FeatureSequence2FeatureVector;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.TokenSequence2FeatureSequence;
import edu.pengli.nlp.platform.pipe.TokenSequenceLowercase;
import edu.pengli.nlp.platform.pipe.TokenSequenceRemoveStopwords;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.Maths;
import edu.pengli.nlp.platform.util.RankMap;
import edu.pengli.nlp.platform.util.matrix.Matrix;

public class BipartiteGraphRandomWalk {

	private Matrix matrix_UV;
	private Matrix matrix_VU;
	private Summary currentSummary;
	private CCTAModel ccta;
	private InstanceList sents_tc;
	private InstanceList sents_gc;
	private HashMap<Integer, Double> idxProb;

	public BipartiteGraphRandomWalk(CCTAModel ccta) {
		this.ccta = ccta;
		currentSummary = new Summary();
		sents_tc = new InstanceList(null);
		sents_gc = new InstanceList(null);

	}

	private void estimateLanguageModel() {
		PipeLine pipeLine = new PipeLine();
		pipeLine.addPipe(new CharSequence2TokenSequence());
		pipeLine.addPipe(new TokenSequenceLowercase());
		pipeLine.addPipe(new TokenSequenceRemoveStopwords());
		pipeLine.addPipe(new TokenSequence2FeatureSequence());
		pipeLine.addPipe(new FeatureSequence2FeatureVector());
		InstanceList tf_fvs = new InstanceList(pipeLine);

		InstanceList instances = ccta.getInstanceList();
		InstanceList sentenceList = new InstanceList(null);
		int docID = 0;
		for (Instance doc : instances) {
			int sentID = 0;
			InstanceList sentsList = (InstanceList) doc.getSource();
			for (Instance inst : sentsList) {
				String name = (String) inst.getName();
				inst.setName(docID + " " + (sentID++) + " " + name);
				Instance sent = new Instance(inst.getSource(), null,
						inst.getName(), inst.getSource());
				sentenceList.add(sent);
			}
			docID++;
		}

		tf_fvs.addThruPipe(sentenceList.iterator());
		HashMap<Integer, Double> idxCounter = new HashMap<Integer, Double>();
		double total = 0;
		for (Instance inst : tf_fvs) {
			FeatureVector fv = (FeatureVector) inst.getData();
			int[] idxs = fv.getIndices();
			double[] vals = fv.getValues();
			for (int i = 0; i < idxs.length; i++) {
				total += vals[i];
				int v = idxs[i];
				if (!idxCounter.containsKey(v)) {
					idxCounter.put(v, vals[i]);
				} else {
					double c = idxCounter.get(v);
					c += vals[i];
					idxCounter.put(v, c);
				}
			}
		}

		idxProb = new HashMap<Integer, Double>();
		for (Instance inst : tf_fvs) {
			FeatureVector fv = (FeatureVector) inst.getData();
			int[] idxs = fv.getIndices();
			for (int i = 0; i < idxs.length; i++) {
				int v = idxs[i];
				double prob = idxCounter.get(v) / total;
				idxProb.put(v, prob);
			}
		}
	}

	public void contructTransitionMatrix() {
		estimateLanguageModel();
		PipeLine pipeLine = new PipeLine();
		pipeLine.addPipe(new CharSequence2TokenSequence());
		pipeLine.addPipe(new TokenSequenceLowercase());
		pipeLine.addPipe(new TokenSequenceRemoveStopwords());
		pipeLine.addPipe(new TokenSequence2FeatureSequence());
		pipeLine.addPipe(new FeatureSequence2FeatureVector());
		InstanceList tf_fvs = new InstanceList(pipeLine);

		InstanceList instances = ccta.getInstanceList();
		InstanceList sentenceList = new InstanceList(null);
		int docID = 0;
		for (Instance doc : instances) {
			int sentID = 0;
			InstanceList sentsList = (InstanceList) doc.getSource();
			for (Instance inst : sentsList) {
				String name = (String) inst.getName();
				inst.setName(docID + " " + (sentID++) + " " + name);
				Instance sent = new Instance(inst.getSource(), null,
						inst.getName(), inst.getSource());
				sentenceList.add(sent);
			}
			docID++;
		}

		tf_fvs.addThruPipe(sentenceList.iterator());
		pipeLine = new PipeLine();
		pipeLine.addPipe(new FeatureIDFPipe(tf_fvs));
		InstanceList tf_idf_fvs = new InstanceList(pipeLine);
		tf_idf_fvs.addThruPipe(tf_fvs.iterator());

		for (Instance inst : tf_idf_fvs) {
			String name = (String) inst.getName();
			if (name.endsWith("_T")) {
				sents_tc.add(inst);
			} else if (name.endsWith("_G")) {
				sents_gc.add(inst);
			}
		}

		int m = sents_tc.size();
		int n = sents_gc.size();
		
		double[][] entriesMN = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				double sim = similarityLM(sents_tc.get(i), sents_gc.get(j));
				entriesMN[i][j] = sim;
			}
		}

		matrix_UV = new Matrix(entriesMN);

		double[][] entriesNM = new double[n][m];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				double sim = similarityLM(sents_gc.get(i), sents_tc.get(j));
				entriesNM[i][j] = sim;
			}
		}
		matrix_VU = new Matrix(entriesNM);
	}

	public void contructTransitionMatrix(CCTAModel ccta) {

		PipeLine pipeLine = new PipeLine();
		pipeLine.addPipe(new CharSequence2TokenSequence());
		pipeLine.addPipe(new TokenSequenceLowercase());
		pipeLine.addPipe(new TokenSequenceRemoveStopwords());
		pipeLine.addPipe(new TokenSequence2FeatureSequence());
		pipeLine.addPipe(new FeatureSequence2FeatureVector());
		InstanceList tf_fvs = new InstanceList(pipeLine);

		InstanceList instances = ccta.getInstanceList();
		InstanceList sentenceList = new InstanceList(null);
		int docID = 0;
		for (Instance doc : instances) {
			int sentID = 0;
			InstanceList sentsList = (InstanceList) doc.getSource();
			for (Instance inst : sentsList) {
				String name = (String) inst.getName();
				inst.setName(docID + " " + (sentID++) + " " + name);
				Instance sent = new Instance(inst.getSource(), null,
						inst.getName(), inst.getSource());
				sentenceList.add(sent);
			}
			docID++;
		}

		tf_fvs.addThruPipe(sentenceList.iterator());
		pipeLine = new PipeLine();
		pipeLine.addPipe(new FeatureIDFPipe(tf_fvs));
		InstanceList tf_idf_fvs = new InstanceList(pipeLine);
		tf_idf_fvs.addThruPipe(tf_fvs.iterator());

		for (Instance inst : tf_idf_fvs) {
			String name = (String) inst.getName();
			if (name.endsWith("_T")) {
				sents_tc.add(inst);
			} else if (name.endsWith("_G")) {
				sents_gc.add(inst);
			}
		}

		int m = sents_tc.size();
		int n = sents_gc.size();
		
		double maxIc = Double.MIN_VALUE, maxId = Double.MIN_VALUE;
		double minIc = Double.MAX_VALUE, minId = Double.MAX_VALUE;
		for (int i = 0; i < m; i++) {
			for(int j=0; j<n; j++){
				double simcom = commonMeasureCCTA(sents_tc.get(i), sents_gc.get(j));
				if(simcom > maxIc){
					maxIc = simcom;
				}
				if(simcom < minIc){
					minIc = simcom;
				}
				double simconstra = diffMeasureCCTAT2N(sents_tc.get(i), sents_gc.get(j));
				if(simconstra > maxId){
					maxId = simconstra;
				}
				if(simconstra < minId){
					minId= simconstra;
				}
			}
			
		}
					
		double[][] entries = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				double sim = similarityCCTAT2N(sents_tc.get(i), sents_gc.get(j), maxIc, minIc, maxId, minId);
				entries[i][j] = sim;
			}
		}
		matrix_UV = new Matrix(entries);

		entries = new double[n][m];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				double sim = similarityCCTAN2T(sents_gc.get(i), sents_tc.get(j), maxIc, minIc, maxId, minId);
				entries[i][j] = sim;
			}
		}

		matrix_VU = new Matrix(entries);

	}

	private double[] getInitialRankingVector(InstanceList sents) {
		InstanceList sentenceList = new InstanceList(null);
		for (Instance inst : sents) {
			Instance sent = new Instance(inst.getSource(), null,
					inst.getName(), inst.getSource());
			sentenceList.add(sent);
		}

		PipeLine pipeLine = new PipeLine();
		pipeLine.addPipe(new CharSequence2TokenSequence());
		pipeLine.addPipe(new TokenSequenceLowercase());
		pipeLine.addPipe(new TokenSequenceRemoveStopwords());
		pipeLine.addPipe(new TokenSequence2FeatureSequence());
		pipeLine.addPipe(new FeatureSequence2FeatureVector());
		InstanceList tf_fvs = new InstanceList(pipeLine);
		tf_fvs.addThruPipe(sentenceList.iterator());

		pipeLine = new PipeLine();
		pipeLine.addPipe(new FeatureIDFPipe(tf_fvs));
		InstanceList tf_idf_fvs = new InstanceList(pipeLine);
		tf_idf_fvs.addThruPipe(tf_fvs.iterator());

		double threshold = 0.2; 
		double damping = 0.1;
		double error = 0.1;
		LexRank lr = new LexRank(tf_idf_fvs, threshold, damping, error);
		lr.rank();
		double[] L = lr.getScore();
		return L;
	}

	private double[] smoothVector(double[] array) {
		double mean = 0.0;
		double sum = 0.0;
		for (double val : array) {
			mean += val;
			sum += val;
		}
		double max = Double.MIN_VALUE;
		for (double val : array) {
			if (max <= val) {
				max = val;
			}
		}
		mean /= array.length;
		double delta = 0.0;
		for (double val : array) {
			delta += Math.pow(Math.abs(val - mean), 2);
		}
		delta /= array.length;
		delta = Math.sqrt(delta);

		double[] sarray = new double[array.length];
		for (int i = 0; i < sarray.length; i++) {
			//sarray[i] = (array[i] - mean) * Math.sqrt(array.length) / delta; // 0.38 0.34
			//  sarray[i] = array[i]/ sum; // 0.38 0.33
			 //sarray[i] = 1 / (1 + Math.exp(-array[i])); //0.38 0.33
			sarray[i] = 1.0 / array.length; //0.35 0.28
		}
		return sarray;
	}

	private Matrix smoothMatrix(Matrix matrix) {
        int m = matrix.getRowDimension();
        int n = matrix.getColumnDimension();
		double[][] array = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				double val = matrix.get(i, j);
			//	double sum = matrix.sumRow(i); //R:0.04987 P:0.05655 F:0.05298
			//	double sum = matrix.sumRow(i) + matrix.sumCol(j); //R:0.05898 P:0.06791 F:0.06312
				double sum = matrix.sumCol(j); // R:0.04933 P:0.05873 F:0.05362
				if(sum == 0.0){
					sum = 1.0;
				}
				array[i][j] = val/sum;
			}
		}

		return new Matrix(array);
	}

	public void mutualReinforcement(int lengthLimit) {


		double[] init_score_t = getInitialRankingVector(sents_tc);
		double[] init_score_g = getInitialRankingVector(sents_gc);

		Matrix X0 = new Matrix(init_score_t, sents_tc.size());
		Matrix Y0 = new Matrix(init_score_g, sents_gc.size());

		// R:0.14577 P:0.16498 F:0.15477 0.9 0.1
		// R:0.14659 P:0.16604 F:0.15569 0.8 0.2
		// R:0.15504 P:0.17369 F:0.16382 0.7 0.3
		// R:0.14659 P:0.16721 F:0.15620 0.6 0.4
		// R:0.15013 P:0.16864 F:0.15883 0.5 0.5
		// R:0.14932 P:0.16793 F:0.15806 0.4 0.6
		// R:0.15204 P:0.17169 F:0.16125 0.3 0.7
		// R:0.15150 P:0.17014 F:0.16027 0.2 0.8
		// R:0.15095 P:0.17026 F:0.15999 0.1 0.9

		double alpha = 0.7;
		double beta = 0.3;
		Matrix Xt = X0;
		Matrix Yt = Y0;
		for (int t = 0; t < 10; t++) {
			Xt = X0.times(alpha).plus(
					matrix_VU.transpose().times(Yt).times(1 - alpha));
			Yt = Y0.times(beta).plus(
					matrix_UV.transpose().times(Xt).times(1 - beta));
		}

		double[] score_t = Xt.getCol(0);
		double[] score_g = Yt.getCol(0);

		HashMap<Integer, Double> locationScoreMap_g = new HashMap<Integer, Double>();
		for (int i = 0; i < score_g.length; i++) {
			locationScoreMap_g.put(i, score_g[i]);
		}

		HashMap<Integer, Double> locationScoreMap_t = new HashMap<Integer, Double>();
		for (int i = 0; i < score_t.length; i++) {
			locationScoreMap_t.put(i, score_t[i]);
		}

		LinkedHashMap sorted_g = RankMap.sortHashMapByValues(
				locationScoreMap_g, false);
		LinkedHashMap sorted_t = RankMap.sortHashMapByValues(
				locationScoreMap_t, false);

		Set<Integer> keys_g = sorted_g.keySet();
		Iterator<Integer> iter_g = keys_g.iterator();

		Set<Integer> keys_t = sorted_t.keySet();
		Iterator<Integer> iter_t = keys_t.iterator();

		while (iter_g.hasNext() && iter_t.hasNext()) {
			int location_g = iter_g.next();
			currentSummary.add(sents_gc.get(location_g));
			if (currentSummary.length() > lengthLimit) {
				currentSummary.remove(sents_gc.get(location_g));
				break;
			}

			int location_t = iter_t.next();
			currentSummary.add(sents_tc.get(location_t));
			if (currentSummary.length() > lengthLimit) {
				currentSummary.remove(sents_tc.get(location_t));
				break;
			}
		}
	}

	public HashMap<Instance, ArrayList<Instance>> mutualReinforcement2(
			int lengthLimit) {
		
		matrix_VU = smoothMatrix(matrix_VU);
		matrix_UV = smoothMatrix(matrix_UV);
			
		double[] init_score_t = getInitialRankingVector(sents_tc);
		double[] init_score_g = getInitialRankingVector(sents_gc);
		
		init_score_t = smoothVector(init_score_t);
		init_score_g = smoothVector(init_score_g);
				
		double[] score_t = new double[sents_tc.size()];
		for (int i = 0; i < score_t.length; i++) {
			score_t[i] = init_score_t[i];
		}

		double[] score_g = new double[sents_gc.size()];
		for (int j = 0; j < score_g.length; j++) {
			score_g[j] = init_score_g[j];
		}

		double alpha = 0.5;
		double beta = 0.5;

		for (int t = 0; t < 10; t++) {

			for (int i = 0; i < init_score_t.length; i++) {
				double sum1 = 0.0;
				for (int k = 0; k < init_score_g.length; k++) {
					sum1 += matrix_VU.get(k, i) * score_g[k];
				}
				double xi = alpha * init_score_t[i] + (1 - alpha) * sum1;
				score_t[i] = xi;
			}

			for (int k = 0; k < init_score_g.length; k++) {
				double sum2 = 0.0;
				for (int j = 0; j < init_score_t.length; j++) {
					sum2 += matrix_UV.get(j, k) * score_t[j];
				}
				double yk = beta * init_score_g[k] + (1 - beta) * sum2;
				score_g[k] = yk;
			}

		}

		HashMap<Integer, Double> locationScoreMap_g = new HashMap<Integer, Double>();
		for (int i = 0; i < score_g.length; i++) {
			locationScoreMap_g.put(i, score_g[i]);
		}

		LinkedHashMap sorted_g = RankMap.sortHashMapByValues(
				locationScoreMap_g, false);

		Set<Integer> keys_g = sorted_g.keySet();
		Iterator<Integer> iter_g = keys_g.iterator();

		HashMap<Instance, ArrayList<Instance>> pairs = new HashMap<Instance, ArrayList<Instance>>();
		int numN = 0;
		while (iter_g.hasNext()) {
			
			int location_g = iter_g.next();
	
			currentSummary.add(sents_gc.get(location_g));
			pairs.put(sents_gc.get(location_g), null);
			numN++;
			
			if (currentSummary.length() > lengthLimit) {
				currentSummary.remove(sents_gc.get(location_g));
				pairs.remove(sents_gc.get(location_g));
				numN--;
				break;
			}

			double[] adjs = matrix_UV.getCol(location_g);
			ArrayList<Instance> adjList = new ArrayList<Instance>();
			HashMap<Integer, Double> locationScoreMap_t = new HashMap<Integer, Double>();
			for (int i = 0; i < score_t.length; i++) {
				locationScoreMap_t.put(i, adjs[i]);
			}
			LinkedHashMap sorted_t = RankMap.sortHashMapByValues(
					locationScoreMap_t, false);
			Set<Integer> keys_t = sorted_t.keySet();
			Iterator<Integer> iter_t = keys_t.iterator();

			HashSet<String> strset = new HashSet<String>();
			int c = 0;
			while (iter_t.hasNext() && c < 2) {
				int location_t = iter_t.next();
				double val = (Double) sorted_t.get(location_t);
				if (val != 0.0) {
					String tmp = (String) sents_tc.get(location_t).getSource();
					if(!strset.contains(tmp)){
						currentSummary.add(sents_tc.get(location_t));
						adjList.add(sents_tc.get(location_t));
						strset.add(tmp);
						c++;
					}
					
				} else
					continue;

				if (currentSummary.length() > lengthLimit) {
					currentSummary.remove(sents_tc.get(location_t));
					adjList.remove(sents_tc.get(location_t));
					break;
				}
			}
			
			if(pairs.containsKey(sents_gc.get(location_g)))
			pairs.put(sents_gc.get(location_g), adjList);
			
		}

	//	System.out.println(numN);
	//	System.out.println(pairs.size());
		return pairs;
	}
	
	// separate generate 
	public HashMap<Instance, ArrayList<Instance>> mutualReinforcement3(
			int lengthLimit) {
		
		matrix_VU = smoothMatrix(matrix_VU);
		matrix_UV = smoothMatrix(matrix_UV);
			
		double[] init_score_t = getInitialRankingVector(sents_tc);
		double[] init_score_g = getInitialRankingVector(sents_gc);
		
		init_score_t = smoothVector(init_score_t);
		init_score_g = smoothVector(init_score_g);
				
		double[] score_t = new double[sents_tc.size()];
		for (int i = 0; i < score_t.length; i++) {
			score_t[i] = init_score_t[i];
		}

		double[] score_g = new double[sents_gc.size()];
		for (int j = 0; j < score_g.length; j++) {
			score_g[j] = init_score_g[j];
		}

		double alpha = 0.5;
		double beta = 0.5;

		for (int t = 0; t < 10; t++) {

			for (int i = 0; i < init_score_t.length; i++) {
				double sum1 = 0.0;
				for (int k = 0; k < init_score_g.length; k++) {
					sum1 += matrix_VU.get(k, i) * score_g[k];
				}
				double xi = alpha * init_score_t[i] + (1 - alpha) * sum1;
				score_t[i] = xi;
			}

			for (int k = 0; k < init_score_g.length; k++) {
				double sum2 = 0.0;
				for (int j = 0; j < init_score_t.length; j++) {
					sum2 += matrix_UV.get(j, k) * score_t[j];
				}
				double yk = beta * init_score_g[k] + (1 - beta) * sum2;
				score_g[k] = yk;
			}

		}

		HashMap<Integer, Double> locationScoreMap_g = new HashMap<Integer, Double>();
		for (int i = 0; i < score_g.length; i++) {
			locationScoreMap_g.put(i, score_g[i]);
		}

		LinkedHashMap sorted_g = RankMap.sortHashMapByValues(
				locationScoreMap_g, false);

		Set<Integer> keys_g = sorted_g.keySet();
		Iterator<Integer> iter_g = keys_g.iterator();

		HashMap<Instance, ArrayList<Instance>> pairs = new HashMap<Instance, ArrayList<Instance>>();
		int numN = 0;
		while (iter_g.hasNext()) {
			
			int location_g = iter_g.next();
	
			currentSummary.add(sents_gc.get(location_g));
			pairs.put(sents_gc.get(location_g), null);
			numN++;
			
			if (currentSummary.length() > lengthLimit) {
				currentSummary.remove(sents_gc.get(location_g));
				pairs.remove(sents_gc.get(location_g));
				numN--;
				break;
			}

			double[] adjs = matrix_UV.getCol(location_g);
			ArrayList<Instance> adjList = new ArrayList<Instance>();
			HashMap<Integer, Double> locationScoreMap_t = new HashMap<Integer, Double>();
			for (int i = 0; i < score_t.length; i++) {
				locationScoreMap_t.put(i, adjs[i]);
			}
			LinkedHashMap sorted_t = RankMap.sortHashMapByValues(
					locationScoreMap_t, false);
			Set<Integer> keys_t = sorted_t.keySet();
			Iterator<Integer> iter_t = keys_t.iterator();

			HashSet<String> strset = new HashSet<String>();
			int c = 0;
			while (iter_t.hasNext() && c < 2) {
				int location_t = iter_t.next();
				double val = (Double) sorted_t.get(location_t);
				if (val != 0.0) {
					String tmp = (String) sents_tc.get(location_t).getSource();
					if(!strset.contains(tmp)){
						currentSummary.add(sents_tc.get(location_t));
						adjList.add(sents_tc.get(location_t));
						strset.add(tmp);
						c++;
					}
					
				} else
					continue;

				if (currentSummary.length() > lengthLimit) {
					currentSummary.remove(sents_tc.get(location_t));
					adjList.remove(sents_tc.get(location_t));
					break;
				}
			}
			
			if(pairs.containsKey(sents_gc.get(location_g)))
			pairs.put(sents_gc.get(location_g), adjList);
			
		}

	//	System.out.println(numN);
	//	System.out.println(pairs.size());
		return pairs;
	}
	
	// separate generate 
	public void mutualReinforcement_AAAI_T(
			int lengthLimitT, int lengthLimitN) {
		
		matrix_VU = smoothMatrix(matrix_VU);
		matrix_UV = smoothMatrix(matrix_UV);
			
		double[] init_score_t = getInitialRankingVector(sents_tc);
		double[] init_score_g = getInitialRankingVector(sents_gc);
		
		init_score_t = smoothVector(init_score_t);
		init_score_g = smoothVector(init_score_g);
				
		double[] score_t = new double[sents_tc.size()];
		for (int i = 0; i < score_t.length; i++) {
			score_t[i] = init_score_t[i];
		}

		double[] score_g = new double[sents_gc.size()];
		for (int j = 0; j < score_g.length; j++) {
			score_g[j] = init_score_g[j];
		}

		double alpha = 0.5;
		double beta = 0.5;

		for (int t = 0; t < 10; t++) {

			for (int i = 0; i < init_score_t.length; i++) {
				double sum1 = 0.0;
				for (int k = 0; k < init_score_g.length; k++) {
					sum1 += matrix_VU.get(k, i) * score_g[k];
				}
				double xi = alpha * init_score_t[i] + (1 - alpha) * sum1;
				score_t[i] = xi;
			}

			for (int k = 0; k < init_score_g.length; k++) {
				double sum2 = 0.0;
				for (int j = 0; j < init_score_t.length; j++) {
					sum2 += matrix_UV.get(j, k) * score_t[j];
				}
				double yk = beta * init_score_g[k] + (1 - beta) * sum2;
				score_g[k] = yk;
			}

		}

		HashMap<Integer, Double> locationScoreMap_g = new HashMap<Integer, Double>();
		for (int i = 0; i < score_g.length; i++) {
			locationScoreMap_g.put(i, score_g[i]);
		}

		LinkedHashMap sorted_g = RankMap.sortHashMapByValues(
				locationScoreMap_g, false);

		Set<Integer> keys_g = sorted_g.keySet();
		Iterator<Integer> iter_g = keys_g.iterator();


		int numN = 0;
		
		Summary tSummary = new Summary();
		Summary nSummary = new Summary();
		while (iter_g.hasNext()) {
			
			int location_g = iter_g.next();
	
			nSummary.add(sents_gc.get(location_g));
			currentSummary.add(sents_gc.get(location_g));
			numN++;
			
			if (nSummary.length() > lengthLimitN) {
				nSummary.remove(sents_gc.get(location_g));
				currentSummary.remove(sents_gc.get(location_g));
				numN--;
				break;
			}

			double[] adjs = matrix_UV.getCol(location_g);
			ArrayList<Instance> adjList = new ArrayList<Instance>();
			HashMap<Integer, Double> locationScoreMap_t = new HashMap<Integer, Double>();
			for (int i = 0; i < score_t.length; i++) {
				locationScoreMap_t.put(i, adjs[i]);
			}
			LinkedHashMap sorted_t = RankMap.sortHashMapByValues(
					locationScoreMap_t, false);
			Set<Integer> keys_t = sorted_t.keySet();
			Iterator<Integer> iter_t = keys_t.iterator();

			HashSet<String> strset = new HashSet<String>();
			int c = 0;
			while (iter_t.hasNext() && c < 1) {
				int location_t = iter_t.next();
				double val = (Double) sorted_t.get(location_t);
				if (val != 0.0) {
					String tmp = (String) sents_tc.get(location_t).getSource();
					if(!strset.contains(tmp)){
						tSummary.add(sents_tc.get(location_t));
						currentSummary.add(sents_tc.get(location_t));
						adjList.add(sents_tc.get(location_t));
						strset.add(tmp);
						c++;
					}
					
				} else
					continue;

				if (tSummary.length() > lengthLimitT) {
					tSummary.remove(sents_tc.get(location_t));
					currentSummary.remove(sents_tc.get(location_t));
					adjList.remove(sents_tc.get(location_t));
					break;
				}
			}
						
		}

	}
	
	public void mutualReinforcement_AAAI_N(
			int lengthLimitT, int lengthLimitN) {
		
		matrix_VU = smoothMatrix(matrix_VU);
		matrix_UV = smoothMatrix(matrix_UV);
			
		double[] init_score_t = getInitialRankingVector(sents_tc);
		double[] init_score_g = getInitialRankingVector(sents_gc);
		
		init_score_t = smoothVector(init_score_t);
		init_score_g = smoothVector(init_score_g);
				
		double[] score_t = new double[sents_tc.size()];
		for (int i = 0; i < score_t.length; i++) {
			score_t[i] = init_score_t[i];
		}

		double[] score_g = new double[sents_gc.size()];
		for (int j = 0; j < score_g.length; j++) {
			score_g[j] = init_score_g[j];
		}

		double alpha = 0.5;
		double beta = 0.5;

		for (int t = 0; t < 10; t++) {

			for (int i = 0; i < init_score_t.length; i++) {
				double sum1 = 0.0;
				for (int k = 0; k < init_score_g.length; k++) {
					sum1 += matrix_VU.get(k, i) * score_g[k];
				}
				double xi = alpha * init_score_t[i] + (1 - alpha) * sum1;
				score_t[i] = xi;
			}

			for (int k = 0; k < init_score_g.length; k++) {
				double sum2 = 0.0;
				for (int j = 0; j < init_score_t.length; j++) {
					sum2 += matrix_UV.get(j, k) * score_t[j];
				}
				double yk = beta * init_score_g[k] + (1 - beta) * sum2;
				score_g[k] = yk;
			}

		}

		HashMap<Integer, Double> locationScoreMap_t = new HashMap<Integer, Double>();
		for (int i = 0; i < score_t.length; i++) {
			locationScoreMap_t.put(i, score_t[i]);
		}

		LinkedHashMap sorted_t = RankMap.sortHashMapByValues(
				locationScoreMap_t, false);

		Set<Integer> keys_t = sorted_t.keySet();
		Iterator<Integer> iter_t = keys_t.iterator();


		int numT = 0;
		
		Summary tSummary = new Summary();
		Summary nSummary = new Summary();
		while (iter_t.hasNext()) {
			
			int location_t = iter_t.next();
	
			tSummary.add(sents_tc.get(location_t));
			currentSummary.add(sents_tc.get(location_t));
			numT++;
			
			if (tSummary.length() > lengthLimitT) {
				tSummary.remove(sents_tc.get(location_t));
				currentSummary.remove(sents_tc.get(location_t));
				numT--;
				break;
			}

			double[] adjs = matrix_VU.getCol(location_t);
			ArrayList<Instance> adjList = new ArrayList<Instance>();
			HashMap<Integer, Double> locationScoreMap_g = new HashMap<Integer, Double>();
			for (int i = 0; i < score_g.length; i++) {
				locationScoreMap_g.put(i, adjs[i]);
			}
			LinkedHashMap sorted_g = RankMap.sortHashMapByValues(
					locationScoreMap_g, false);
			Set<Integer> keys_g = sorted_g.keySet();
			Iterator<Integer> iter_g = keys_g.iterator();

			HashSet<String> strset = new HashSet<String>();
			int c = 0;
			while (iter_g.hasNext() && c < 1) {
				int location_g= iter_g.next();
				double val = (Double) sorted_g.get(location_g);
				if (val != 0.0) {
					String tmp = (String) sents_gc.get(location_g).getSource();
					if(!strset.contains(tmp)){
						nSummary.add(sents_gc.get(location_g));
						currentSummary.add(sents_gc.get(location_g));
						adjList.add(sents_gc.get(location_g));
						strset.add(tmp);
						c++;
					}
					
				} else
					continue;

				if (nSummary.length() > lengthLimitN) {
					nSummary.remove(sents_gc.get(location_g));
					currentSummary.remove(sents_gc.get(location_g));
					adjList.remove(sents_gc.get(location_g));
					break;
				}
			}
						
		}

	}
	
	//fair compare
	public void mutualReinforcement_AAAI(
			int lengthLimitT, int lengthLimitN) {
		
		matrix_VU = smoothMatrix(matrix_VU);
		matrix_UV = smoothMatrix(matrix_UV);
			
		double[] init_score_t = getInitialRankingVector(sents_tc);
		double[] init_score_g = getInitialRankingVector(sents_gc);
		
		init_score_t = smoothVector(init_score_t);
		init_score_g = smoothVector(init_score_g);
				
		double[] score_t = new double[sents_tc.size()];
		for (int i = 0; i < score_t.length; i++) {
			score_t[i] = init_score_t[i];
		}

		double[] score_g = new double[sents_gc.size()];
		for (int j = 0; j < score_g.length; j++) {
			score_g[j] = init_score_g[j];
		}
       // 0.1 0.1 0.34 0.29  // 0 0 0.35 0.27 // 1 1 
		double alpha = 0.1;
		double beta = 0.1;

		for (int t = 0; t < 10; t++) {

			for (int i = 0; i < init_score_t.length; i++) {
				double sum1 = 0.0;
				for (int k = 0; k < init_score_g.length; k++) {
					sum1 += matrix_VU.get(k, i) * score_g[k];
				}
				double xi = alpha * init_score_t[i] + (1 - alpha) * sum1;
				score_t[i] = xi;
			}

			for (int k = 0; k < init_score_g.length; k++) {
				double sum2 = 0.0;
				for (int j = 0; j < init_score_t.length; j++) {
					sum2 += matrix_UV.get(j, k) * score_t[j];
				}
				double yk = beta * init_score_g[k] + (1 - beta) * sum2;
				score_g[k] = yk;
			}

		}

		HashMap<Integer, Double> locationScoreMap_g = new HashMap<Integer, Double>();
		for (int i = 0; i < score_g.length; i++) {
			locationScoreMap_g.put(i, score_g[i]);
		}

		LinkedHashMap sorted_g = RankMap.sortHashMapByValues(
				locationScoreMap_g, false);

		Set<Integer> keys_g = sorted_g.keySet();
		Iterator<Integer> iter_g = keys_g.iterator();


		Summary nSummary = new Summary();
		while (iter_g.hasNext()) {
			int location_g = iter_g.next();
	
			nSummary.add(sents_gc.get(location_g));
			currentSummary.add(sents_gc.get(location_g));
			
			if (nSummary.length() > lengthLimitN) {
				nSummary.remove(sents_gc.get(location_g));
				currentSummary.remove(sents_gc.get(location_g));
				break;
			}
		}
		
		HashMap<Integer, Double> locationScoreMap_t = new HashMap<Integer, Double>();
		for (int i = 0; i < score_t.length; i++) {
			locationScoreMap_t.put(i, score_t[i]);
		}

		LinkedHashMap sorted_t = RankMap.sortHashMapByValues(
				locationScoreMap_t, false);

		Set<Integer> keys_t = sorted_t.keySet();
		Iterator<Integer> iter_t = keys_t.iterator();


		Summary tSummary = new Summary();
		while (iter_t.hasNext()) {
			int location_t = iter_t.next();
	
			tSummary.add(sents_tc.get(location_t));
			currentSummary.add(sents_tc.get(location_t));
			
			if (tSummary.length() > lengthLimitT) {
				tSummary.remove(sents_tc.get(location_t));
				currentSummary.remove(sents_tc.get(location_t));
				break;
			}
		}

	}
	

	private InstanceList getNeibors_T(Instance i) {
		InstanceList list = new InstanceList(null);
		list.addAll(sents_tc);
		if (!list.remove(i)) {
			System.out.println("element doesn't exist");
			System.exit(0);
		}
		return list;
	}

	private InstanceList getNeibors_N(Instance i) {
		InstanceList list = new InstanceList(null);
		list.addAll(sents_gc);
		if (!list.remove(i)) {
			System.out.println("element doesn't exist");
			System.exit(0);
		}
		return list;
	}

	private double similarityCOS(Instance inst_i, Instance inst_j) {

		return Maths.idf_modified_cosine(inst_i, inst_j);
		
	}

	private double similarityLM(Instance inst_i, Instance inst_j) {

		HashMap<Integer, Double> thetaT = new HashMap<Integer, Double>();
		HashMap<Integer, Double> thetaN = new HashMap<Integer, Double>();
		FeatureVector fv_i = ((FeatureVector[]) inst_i.getData())[0];
		int[] idx_i = fv_i.getIndices();
		double[] val_i = fv_i.getValues();

		FeatureVector fv_j = ((FeatureVector[]) inst_j.getData())[0];
		int[] idx_j = fv_j.getIndices();
		double[] val_j = fv_j.getValues();

		Set<Integer> keys = idxProb.keySet();
		Iterator<Integer> iter = keys.iterator();
		while (iter.hasNext()) {
			int v = iter.next();
			double tf_i = 0.0;
			for (int i = 0; i < idx_i.length; i++) {
				if (v == idx_i[i])
					tf_i = val_i[i];
			}
			double prob_i = (tf_i + 2 * idxProb.get(v)) / (idx_i.length + 2);
			thetaT.put(v, prob_i);

			double tf_j = 0.0;
			for (int i = 0; i < idx_j.length; i++) {
				if (v == idx_j[i])
					tf_j = val_j[i];
			}

			double prob_j = (tf_j + 2 * idxProb.get(v)) / (idx_j.length + 2);
			thetaN.put(v, prob_j);
		}

		double score = 0.0;
		keys = idxProb.keySet();
		iter = keys.iterator();
		while (iter.hasNext()) {
			int v = iter.next();
			score += thetaT.get(v) * Math.log(thetaT.get(v)/thetaN.get(v))/Math.log(2);																									
		}

		return score;

	}
	
	private double similarityCCTAT2N(Instance inst_i, Instance inst_j, double maxIc, double minIc,
			double maxId, double minId) {

		double cos = Maths.idf_modified_cosine(inst_i, inst_j); 
		double Ic = commonMeasureCCTA(inst_i, inst_j);
		double Id = diffMeasureCCTAT2N(inst_i, inst_j);

		double Icnorm =(Ic-minIc)/(maxIc-minIc);
    	double Idnorm =(Id-minId)/(maxId-minId);

		
		double favor = 0.0;
		if(Icnorm >= Idnorm) {
		    favor = Idnorm / Icnorm; 
		}else if(Icnorm < Idnorm){
			favor = Icnorm / Idnorm;
		}
		
		//return cos * favor;
		return cos;
	}
	
	private double similarityCCTAN2T(Instance inst_i, Instance inst_j, double maxIc, double minIc,
			double maxId, double minId) {

		double cos = Maths.idf_modified_cosine(inst_i, inst_j); 
		double Ic = commonMeasureCCTA(inst_i, inst_j);
		double Id = diffMeasureCCTAN2T(inst_i, inst_j);

		double Icnorm =(Ic-minIc)/(maxIc-minIc);
    	double Idnorm =(Id-minId)/(maxId-minId);

		double favor = 0.0;
		if(Icnorm >= Idnorm) {
		   favor = Idnorm / Icnorm;
		}else if(Icnorm < Idnorm){
			favor = Icnorm / Idnorm;
		}
		
		//return cos * favor;
			return cos;
	}
	
	private double commonMeasureCCTA(Instance inst_i, Instance inst_j){
		double[][] k_v_prob = ccta.getGeneralTopicWordDistribution();
		double[][] a_v_prob = ccta.getGeneralAspectWordDistribution();
		double[][][] k_a_v_prob = ccta.getGeneralMixWordDistribution();

		FeatureVector vs_i = ((FeatureVector[]) inst_i.getData())[0];
		int[] idx_i = vs_i.getIndices();

		FeatureVector vs_j = ((FeatureVector[]) inst_j.getData())[0];
		int[] idx_j = vs_j.getIndices();

		int K = ccta.getNumberOfTopics();
		int A = ccta.getNumberOfAspects();
		
		double ret = 0.0;
		for (int k = 0; k < K; k++) {
	
			double u_i_z_prob_log = 0.0;
			int N=0;
			for (int i = 0; i < idx_i.length; i++) {
				int v = idx_i[i];
				u_i_z_prob_log += Math.log10(k_v_prob[k][v]);
				N++;

			}
			u_i_z_prob_log /= N;
			u_i_z_prob_log = Math.exp(u_i_z_prob_log);
			
			N=0;
			double v_j_z_prob_log = 0.0;
			for (int j = 0; j < idx_j.length; j++) {
				int v = idx_j[j];
				v_j_z_prob_log += Math.log10(k_v_prob[k][v]);
				N++;
			}
			v_j_z_prob_log /= N;
			v_j_z_prob_log = Math.exp(v_j_z_prob_log);

			//ret += (u_i_z_prob_log * v_j_z_prob_log);
			 ret += Math.log(u_i_z_prob_log) + Math.log(v_j_z_prob_log);
		}
		
		for (int a = 0; a < A; a++) {
            
			int N=0;
			double u_i_y_prob_log = 0.0;
			for (int i = 0; i < idx_i.length; i++) {
				int v = idx_i[i];
				u_i_y_prob_log += Math.log10(a_v_prob[a][v]);
				N++;
			}

			u_i_y_prob_log /= N;
			u_i_y_prob_log = Math.exp(u_i_y_prob_log);
			
			N=0;
			double v_j_y_prob_log = 0.0;
			for (int j = 0; j < idx_j.length; j++) {
				int v = idx_j[j];
				v_j_y_prob_log += Math.log10(a_v_prob[a][v]);
				N++;
			}
			v_j_y_prob_log /= N;
			v_j_y_prob_log = Math.exp(v_j_y_prob_log);
			
	         //ret += (u_i_y_prob_log * v_j_y_prob_log);
			 ret += Math.log(u_i_y_prob_log) + Math.log(v_j_y_prob_log);
		}
		
		for (int k = 0; k < K; k++) {
			for (int a = 0; a < A; a++) {
                int N=0;
				double u_i_zy_prob_log = 0.0;
				for (int i = 0; i < idx_i.length; i++) {
					int v = idx_i[i];
					u_i_zy_prob_log += Math.log10(k_a_v_prob[k][a][v]);
					N++;
				}
				u_i_zy_prob_log /= N;
				u_i_zy_prob_log = Math.exp(u_i_zy_prob_log);

				N=0;
				double v_j_zy_prob_log = 0.0;
				for (int j = 0; j < idx_j.length; j++) {
					int v = idx_j[j];
					v_j_zy_prob_log += Math.log10(k_a_v_prob[k][a][v]);
					N++;
				}
				v_j_zy_prob_log /= N;
				v_j_zy_prob_log = Math.exp(v_j_zy_prob_log);
				
				 //ret += (u_i_zy_prob_log * v_j_zy_prob_log);
				 ret += Math.log(u_i_zy_prob_log) + Math.log(v_j_zy_prob_log);

			}

		}

		return ret;
	}

	private double diffMeasureCCTAT2N(Instance inst_i, Instance inst_j) {

		double[][][] k_v_c_prob = ccta
				.getCollectionSpecificTopicWordDistribution();
		double[][][] a_v_c_prob = ccta
				.getCollectionSpecificAspectWordDistribution();
		double[][][][] k_a_v_c_prob = ccta
				.getCollectionSpecificMixWordDistribution();

		FeatureVector vs_i = ((FeatureVector[]) inst_i.getData())[0];
		int[] idx_i = vs_i.getIndices();

		FeatureVector vs_j = ((FeatureVector[]) inst_j.getData())[0];
		int[] idx_j = vs_j.getIndices();

		int K = ccta.getNumberOfTopics();
		int A = ccta.getNumberOfAspects();

		double f_z_E3 = 0.0;
		for (int k = 0; k < K; k++) {
			double t_j_z_t_prob_log = 0.0;
			int N = 0;
			for (int i = 0; i < idx_i.length; i++) {
				int v = idx_i[i];
				t_j_z_t_prob_log += Math.log10(k_v_c_prob[k][v][0]);
				N++;
			}
			t_j_z_t_prob_log /= N;
			t_j_z_t_prob_log = Math.exp(t_j_z_t_prob_log);

			double t_j_z_n_prob_log = 0.0;
			for (int i = 0; i < idx_i.length; i++) {
				int v = idx_i[i];
				t_j_z_n_prob_log += Math.log10(k_v_c_prob[k][v][1]);

			}
			t_j_z_n_prob_log /= N;
			t_j_z_n_prob_log = Math.exp(t_j_z_n_prob_log);
           
            N=0;
			double n_i_z_t_prob_log = 0.0;
			for (int j = 0; j < idx_j.length; j++) {
				int v = idx_j[j];
				n_i_z_t_prob_log += Math.log10(k_v_c_prob[k][v][0]);
				N++;
			}
			n_i_z_t_prob_log /= N;
			n_i_z_t_prob_log = Math.exp(n_i_z_t_prob_log);

			double n_i_z_n_prob_log = 0.0;
			for (int j = 0; j < idx_j.length; j++) {
				int v = idx_j[j];
				n_i_z_n_prob_log += Math.log10(k_v_c_prob[k][v][1]);

			}
			n_i_z_n_prob_log /= N;
			n_i_z_n_prob_log = Math.exp(n_i_z_n_prob_log);
				
			f_z_E3 += (n_i_z_n_prob_log * t_j_z_t_prob_log)/
					(n_i_z_t_prob_log * t_j_z_n_prob_log);


		}
		
		double f_y_E3 = 0.0;
		for (int a = 0; a < A; a++) {
			int N=0;
			double t_j_y_t_prob_log = 0.0;
			for (int i = 0; i < idx_i.length; i++) {
				int v = idx_i[i];
				t_j_y_t_prob_log += Math.log10(a_v_c_prob[a][v][0]);
				N++;
			}
			t_j_y_t_prob_log /= N;
			t_j_y_t_prob_log = Math.exp(t_j_y_t_prob_log);

			double t_j_y_n_prob_log = 0.0;
			for (int i = 0; i < idx_i.length; i++) {
				int v = idx_i[i];
				t_j_y_n_prob_log += Math.log10(a_v_c_prob[a][v][1]);

			}
			t_j_y_n_prob_log /= N;
			t_j_y_n_prob_log = Math.exp(t_j_y_n_prob_log);

			N=0;
			double n_i_y_t_prob_log = 0.0;
			for (int j = 0; j < idx_j.length; j++) {
				int v = idx_j[j];
				n_i_y_t_prob_log += Math.log10(a_v_c_prob[a][v][0]);
				N++;
			}
			n_i_y_t_prob_log /= N;
			n_i_y_t_prob_log = Math.exp(n_i_y_t_prob_log);
			
			double n_i_y_n_prob_log = 0.0;
			for (int j = 0; j < idx_j.length; j++) {
				int v = idx_j[j];
				n_i_y_n_prob_log += Math.log10(a_v_c_prob[a][v][1]);
			}
			n_i_y_n_prob_log /=N;
			n_i_y_n_prob_log = Math.exp(n_i_y_n_prob_log);

			f_y_E3 += (n_i_y_n_prob_log * t_j_y_t_prob_log)/
					(n_i_y_t_prob_log * t_j_y_n_prob_log);
						
		}

		double f_zy_E3 = 0.0;
		for (int k = 0; k < K; k++) {
			for (int a = 0; a < A; a++) {
				int N=0;
				double t_j_zy_t_prob_log = 0.0;
				for (int i = 0; i < idx_i.length; i++) {
					int v = idx_i[i];
					t_j_zy_t_prob_log += Math.log10(k_a_v_c_prob[k][a][v][0]); 
				N++;
				}
				t_j_zy_t_prob_log /= N;
				t_j_zy_t_prob_log = Math.exp(t_j_zy_t_prob_log);

				double t_j_zy_n_prob_log = 0.0;
				for (int i = 0; i < idx_i.length; i++) {
					int v = idx_i[i];
					t_j_zy_n_prob_log += Math.log10(k_a_v_c_prob[k][a][v][1]);

				}
				t_j_zy_n_prob_log /= N;
				t_j_zy_n_prob_log = Math.exp(t_j_zy_n_prob_log);

				N=0;
				double n_i_zy_t_prob_log = 0.0;
				for (int j = 0; j < idx_j.length; j++) {
					int v = idx_j[j];
					n_i_zy_t_prob_log += Math.log10(k_a_v_c_prob[k][a][v][0]);
                    N++;
				}
				n_i_zy_t_prob_log /= N;
				n_i_zy_t_prob_log = Math.exp(n_i_zy_t_prob_log);
				
				double n_i_zy_n_prob_log = 0.0;
				for (int j = 0; j < idx_j.length; j++) {
					int v = idx_j[j];
					n_i_zy_n_prob_log += Math.log10(k_a_v_c_prob[k][a][v][1]);
				}
				n_i_zy_n_prob_log /= N;
				n_i_zy_n_prob_log = Math.exp(n_i_zy_n_prob_log);

				f_zy_E3 += (n_i_zy_n_prob_log * t_j_zy_t_prob_log)/
						(n_i_zy_t_prob_log * t_j_zy_n_prob_log);
				
			}
		}
		
		return f_z_E3 + f_y_E3 + f_zy_E3; 
	}
	
	private double diffMeasureCCTAN2T(Instance inst_i, Instance inst_j) {

		double[][][] k_v_c_prob = ccta
				.getCollectionSpecificTopicWordDistribution();
		double[][][] a_v_c_prob = ccta
				.getCollectionSpecificAspectWordDistribution();
		double[][][][] k_a_v_c_prob = ccta
				.getCollectionSpecificMixWordDistribution();

		FeatureVector vs_i = ((FeatureVector[]) inst_i.getData())[0];
		int[] idx_i = vs_i.getIndices();

		FeatureVector vs_j = ((FeatureVector[]) inst_j.getData())[0];
		int[] idx_j = vs_j.getIndices();

		int K = ccta.getNumberOfTopics();
		int A = ccta.getNumberOfAspects();

		double f_z_E3 = 0.0;
		for (int k = 0; k < K; k++) {
			double n_i_z_n_prob_log = 0.0;
			int N = 0;
			for (int i = 0; i < idx_i.length; i++) {
				int v = idx_i[i];
				n_i_z_n_prob_log += Math.log10(k_v_c_prob[k][v][1]);
				N++;
			}
			n_i_z_n_prob_log /= N;
			n_i_z_n_prob_log = Math.exp(n_i_z_n_prob_log);

			double n_i_z_t_prob_log = 0.0;
			for (int i = 0; i < idx_i.length; i++) {
				int v = idx_i[i];
				n_i_z_t_prob_log += Math.log10(k_v_c_prob[k][v][0]);

			}
			n_i_z_t_prob_log /= N;
			n_i_z_t_prob_log = Math.exp(n_i_z_t_prob_log);
           
            N=0;
			double t_j_z_n_prob_log = 0.0;
			for (int j = 0; j < idx_j.length; j++) {
				int v = idx_j[j];
				t_j_z_n_prob_log += Math.log10(k_v_c_prob[k][v][1]);
				N++;
			}
			t_j_z_n_prob_log /= N;
			t_j_z_n_prob_log = Math.exp(t_j_z_n_prob_log);

			double t_j_z_t_prob_log = 0.0;
			for (int j = 0; j < idx_j.length; j++) {
				int v = idx_j[j];
				t_j_z_t_prob_log += Math.log10(k_v_c_prob[k][v][0]);

			}
			t_j_z_t_prob_log /= N;
			t_j_z_t_prob_log = Math.exp(t_j_z_t_prob_log);
			
			f_z_E3 += (n_i_z_n_prob_log * t_j_z_t_prob_log)/
					(n_i_z_t_prob_log * t_j_z_n_prob_log);

		}
		

		double f_y_E3 = 0.0;
		for (int a = 0; a < A; a++) {
			double n_i_y_n_prob_log = 0.0;
			int N = 0;
			for (int i = 0; i < idx_i.length; i++) {
				int v = idx_i[i];
				n_i_y_n_prob_log += Math.log10(a_v_c_prob[a][v][1]);
				N++;
			}
			n_i_y_n_prob_log /= N;
			n_i_y_n_prob_log = Math.exp(n_i_y_n_prob_log);

			double n_i_y_t_prob_log = 0.0;
			for (int i = 0; i < idx_i.length; i++) {
				int v = idx_i[i];
				n_i_y_t_prob_log += Math.log10(a_v_c_prob[a][v][0]);

			}
			n_i_y_t_prob_log /= N;
			n_i_y_t_prob_log = Math.exp(n_i_y_t_prob_log);
           
            N=0;
			double t_j_y_n_prob_log = 0.0;
			for (int j = 0; j < idx_j.length; j++) {
				int v = idx_j[j];
				t_j_y_n_prob_log += Math.log10(a_v_c_prob[a][v][1]);
				N++;
			}
			t_j_y_n_prob_log /= N;
			t_j_y_n_prob_log = Math.exp(t_j_y_n_prob_log);

			double t_j_y_t_prob_log = 0.0;
			for (int j = 0; j < idx_j.length; j++) {
				int v = idx_j[j];
				t_j_y_t_prob_log += Math.log10(a_v_c_prob[a][v][0]);

			}
			t_j_y_t_prob_log /= N;
			t_j_y_t_prob_log = Math.exp(t_j_y_t_prob_log);
			
			f_y_E3 += (n_i_y_n_prob_log * t_j_y_t_prob_log)/
					(n_i_y_t_prob_log * t_j_y_n_prob_log);

		}

		double f_zy_E3 = 0.0;
		for (int k = 0; k < K; k++) {
			for (int a = 0; a < A; a++) {
				int N=0;
				double n_i_zy_n_prob_log = 0.0;
				for (int i = 0; i < idx_i.length; i++) {
					int v = idx_i[i];
				n_i_zy_n_prob_log += Math.log10(k_a_v_c_prob[k][a][v][1]); 
				N++;
				}
				n_i_zy_n_prob_log /= N;
				n_i_zy_n_prob_log = Math.exp(n_i_zy_n_prob_log);

				double n_i_zy_t_prob_log = 0.0;
				for (int i = 0; i < idx_i.length; i++) {
					int v = idx_i[i];
					n_i_zy_t_prob_log += Math.log10(k_a_v_c_prob[k][a][v][0]);

				}
				n_i_zy_t_prob_log /= N;
				n_i_zy_t_prob_log = Math.exp(n_i_zy_t_prob_log);

				N=0;
				double t_j_zy_n_prob_log = 0.0;
				for (int j = 0; j < idx_j.length; j++) {
					int v = idx_j[j];
					t_j_zy_n_prob_log += Math.log10(k_a_v_c_prob[k][a][v][1]);
                    N++;
				}
				t_j_zy_n_prob_log /= N;
				t_j_zy_n_prob_log = Math.exp(t_j_zy_n_prob_log);
				
				double t_j_zy_t_prob_log = 0.0;
				for (int j = 0; j < idx_j.length; j++) {
					int v = idx_j[j];
					t_j_zy_t_prob_log += Math.log10(k_a_v_c_prob[k][a][v][0]);
				}
				t_j_zy_t_prob_log /= N;
				t_j_zy_t_prob_log = Math.exp(t_j_zy_t_prob_log);

				f_zy_E3 += (n_i_zy_n_prob_log * t_j_zy_t_prob_log)/
						(n_i_zy_t_prob_log * t_j_zy_n_prob_log);
				
			}
		}

		return f_z_E3 + f_y_E3 + f_zy_E3; 

	}



	public void outputSummary(String topic, int iterTime) {

		String outputDir = "../data/CIKM2012/Output/summary";

		PrintWriter out = FileOperation.getPrintWriter(new File(outputDir),
				String.valueOf(topic+"."+iterTime));
		
		PrintWriter out_t = FileOperation.getPrintWriter(new File(outputDir),
				String.valueOf(topic+"."+iterTime+".T"));
		
		PrintWriter out_n = FileOperation.getPrintWriter(new File(outputDir),
				String.valueOf(topic+"."+iterTime+".N"));

		//System.out.println(currentSummary.length());
		out.println("<html>");
		out.println("<body bgcolor=\"white\">");
		
		out_t.println("<html>");
		out_t.println("<body bgcolor=\"white\">");
		
		out_n.println("<html>");
		out_n.println("<body bgcolor=\"white\">");

		int i = 1;
		for (Instance sent : currentSummary) {
			// System.out.println(sent.getName() + ":" + sent.getSource());
			out.println("<a name=\"" + i + "\">[" + i + "]</a> "
					+ "<a href=\"#" + i + "\" " + "id=" + i + ">"
					+ sent.getSource() + "</a>");
			i++;
		}
		
		i = 1;
		for (Instance sent : currentSummary) {
			// System.out.println(sent.getName() + ":" + sent.getSource());
			if(sent.getName().toString().endsWith("_G")){
				out_n.println("<a name=\"" + i + "\">[" + i + "]</a> "
						+ "<a href=\"#" + i + "\" " + "id=" + i + ">"
						+ sent.getSource() + "</a>");
				i++;
			}
			
		}
		
		i = 1;
		for (Instance sent : currentSummary) {
			// System.out.println(sent.getName() + ":" + sent.getSource());
			if(sent.getName().toString().endsWith("_T")){
				out_t.println("<a name=\"" + i + "\">[" + i + "]</a> "
						+ "<a href=\"#" + i + "\" " + "id=" + i + ">"
						+ sent.getSource() + "</a>");
				i++;
			}
			
		}
		
		
		
		out.println("</body>");
		out.println("</html>");
		out.close();
		
		out_t.println("</body>");
		out_t.println("</html>");
		out_t.close();
		
		out_n.println("</body>");
		out_n.println("</html>");
		out_n.close();

	}

}
