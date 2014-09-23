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
import edu.pengli.nlp.platform.algorithms.classify.Clustering;
import edu.pengli.nlp.platform.algorithms.lda.CCTAModel;
import edu.pengli.nlp.platform.algorithms.ranking.LexRank;
import edu.pengli.nlp.platform.pipe.CharSequence2TokenSequence;
import edu.pengli.nlp.platform.pipe.FeatureIDFPipe;
import edu.pengli.nlp.platform.pipe.FeatureSequence2FeatureVector;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.TokenSequence2FeatureSequence;
import edu.pengli.nlp.platform.pipe.TokenSequenceLowercase;
import edu.pengli.nlp.platform.pipe.TokenSequenceRemoveStopwords;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.Maths;
import edu.pengli.nlp.platform.util.RankMap;
import lpsolve.LpSolve;
import lpsolve.LpSolveException;

public class IntegerLinearProgramming {

	CCTAModel model;
	private Summary currentSummary;
	LpSolve solver;
	InstanceList sentenceList;

	public IntegerLinearProgramming(CCTAModel model) throws LpSolveException {
		this.model = model;
		sentenceList = new InstanceList(new PipeLine());
		InstanceList docs = model.getInstanceList();
		for (Instance doc : docs) {
			InstanceList sents = (InstanceList) doc.getData();
			for (Instance sent : sents) {
				sentenceList.add(sent);
			}
		}
		

	}

	private void solve_N(Clustering clusters, int summaryLength_N)
			throws LpSolveException {
		
		currentSummary = new Summary();
		solver = LpSolve.makeLp(0, clusters.getInstances().size());
		solver.setOutputfile("");

		for (int i = 0; i < clusters.getInstances().size(); i++) {
			solver.setColName(i + 1, "s" + i);
			solver.setBinary(i + 1, true);
		}

		// set Objective Function
		StringBuffer sb_o = new StringBuffer();
		InstanceList[] cs = clusters.getClusters();
		for (int c = 0; c < cs.length; c++) {
			InstanceList cluster = cs[c];
			for (int l = 0; l < cluster.size(); l++) {
				int posi = l + 1;
				sb_o.append(posi + " ");
			}
		}

		solver.strSetObjFn(sb_o.toString().trim());
		solver.setMinim();

		// length constraints
		StringBuffer sb_L_N = new StringBuffer();
		for (int c = 0; c < cs.length; c++) {
			InstanceList cluster = cs[c];
			for (int l = 0; l < cluster.size(); l++) {
				Instance sent = cluster.get(l);
				String sentMention = (String) sent.getSource();
				sb_L_N.append(sentMention.split(" ").length + " ");

			}

		}

		solver.strAddConstraint(sb_L_N.toString().trim(), LpSolve.LE,
				summaryLength_N);

		// exclusivity constraints
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		int globalIdx = 0;
		for (int c = 0; c < cs.length; c++) {
			InstanceList cluster = cs[c];
			for (int l = 0; l < cluster.size(); l++) {
				map.put(c + "_" + l, globalIdx++);
			}
		}

		int start = 0;
		for (int g = 0; g < cs.length; g++) {
			StringBuffer sb_E_N = new StringBuffer();
			for (int c = 0; c < cs.length; c++) {
				InstanceList cluster = cs[c];
				for (int k = 0; k < cluster.size(); k++) {
					int idx = map.get(c + "_" + k);
					if (idx >= start && idx <= start + cluster.size() - 1) {
						sb_E_N.append(1 + " ");
					} else
						sb_E_N.append(0 + " ");

				}

			}
			start += cs[g].size();
			solver.strAddConstraint(sb_E_N.toString().trim(), LpSolve.EQ, 1);
		}

		// Redundancy Constraints
/*		for (int i = 0; i < cs.length; i++) {
			InstanceList clusteri = cs[i];
			for (int m = 0; m < clusteri.size(); m++) {

				for (int j = i + 1; j < cs.length; j++) {
					InstanceList clusterj = cs[j];
					for (int n = 0; n < clusterj.size(); n++) {

						solver.strAddConstraint(
								buildStrVector(m, n, i, j, clusters),
								LpSolve.LE, 0.05);

					}

				}

			}

		}*/

		solver.solve();

		solver.setVerbose(LpSolve.IMPORTANT);
		double[] var = solver.getPtrVariables();
		for (int i = 0; i < var.length; i++) {
			if (var[i] == 1.0) {
				Instance sent = clusters.getInstances().get(i);
				currentSummary.add(sent);

			}
		}

	}
	
	private void solve_T(Clustering clusters, int summaryLength_T)
			throws LpSolveException {
		
		currentSummary = new Summary();
		solver = LpSolve.makeLp(0, clusters.getInstances().size());
		solver.setOutputfile("");

		for (int i = 0; i < clusters.getInstances().size(); i++) {
			solver.setColName(i + 1, "s" + i);
			solver.setBinary(i + 1, true);
		}

		// set Objective Function
		StringBuffer sb_o = new StringBuffer();
		InstanceList[] cs = clusters.getClusters();
		for (int c = 0; c < cs.length; c++) {
			InstanceList cluster = cs[c];
			for (int l = 0; l < cluster.size(); l++) {
				int posi = l + 1;
				sb_o.append(posi + " ");
			}
		}

		solver.strSetObjFn(sb_o.toString().trim());
		solver.setMinim();

		// length constraints
		StringBuffer sb_L_T = new StringBuffer();
		for (int c = 0; c < cs.length; c++) {
			InstanceList cluster = cs[c];
			for (int l = 0; l < cluster.size(); l++) {
				Instance sent = cluster.get(l);
				String sentMention = (String) sent.getSource();
				sb_L_T.append(sentMention.split(" ").length + " ");

			}

		}

		solver.strAddConstraint(sb_L_T.toString().trim(), LpSolve.LE,
				summaryLength_T);

		// exclusivity constraints
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		int globalIdx = 0;
		for (int c = 0; c < cs.length; c++) {
			InstanceList cluster = cs[c];
			for (int l = 0; l < cluster.size(); l++) {
				map.put(c + "_" + l, globalIdx++);
			}
		}

		int start = 0;
		for (int g = 0; g < cs.length; g++) {
			StringBuffer sb_E_T = new StringBuffer();
			for (int c = 0; c < cs.length; c++) {
				InstanceList cluster = cs[c];
				for (int k = 0; k < cluster.size(); k++) {
					int idx = map.get(c + "_" + k);
					if (idx >= start && idx <= start + cluster.size() - 1) {
						sb_E_T.append(1 + " ");
					} else
						sb_E_T.append(0 + " ");

				}

			}
			start += cs[g].size();
			solver.strAddConstraint(sb_E_T.toString().trim(), LpSolve.EQ, 1);
		}

		solver.solve();

		solver.setVerbose(LpSolve.IMPORTANT);
		double[] var = solver.getPtrVariables();
		for (int i = 0; i < var.length; i++) {
			if (var[i] == 1.0) {
				Instance sent = clusters.getInstances().get(i);
				currentSummary.add(sent);

			}
		}

	}

	private String buildStrVector(int m, int n, int i, int j,
			Clustering clusters) {

		InstanceList[] cs = clusters.getClusters();

		InstanceList clusteri = cs[i];
		Instance instm = clusteri.get(m);

		InstanceList clusterj = cs[j];
		Instance instn = clusterj.get(n);

		double sim = Maths.idf_modified_cosine(instm, instn);

		int mdirect = 0;

		for (int k = 0; k < clusters.getNumClusters(); k++) {
			if (k == i) {
				mdirect += m;
				break;
			} else
				mdirect += cs[k].size();
		}

		int ndirect = 0;
		for (int k = 0; k < clusters.getNumClusters(); k++) {
			if (k == j) {
				ndirect += n;
				break;
			} else
				ndirect += cs[k].size();
		}

		StringBuffer sb = new StringBuffer();
		for (int k = 0; k < sentenceList.size(); k++) {
			if (k == mdirect || k == ndirect) {
				sb.append(sim + " ");
			} else
				sb.append(0 + " ");
		}

		return sb.toString().trim();
	}

	public void outputSummary_N(String topic, int iterTime) {
		String outputDir = "/home/peng/Develop/Workspace/NLP/data/CIKM2012/Output/summary";

		PrintWriter out_n = FileOperation.getPrintWriter(new File(outputDir),
				String.valueOf(topic + "." + iterTime + ".N"));

		// System.out.println(currentSummary.length());

		out_n.println("<html>");
		out_n.println("<body bgcolor=\"white\">");

		int i = 1;
		for (Instance sent : currentSummary) {

			out_n.println("<a name=\"" + i + "\">[" + i + "]</a> "
					+ "<a href=\"#" + i + "\" " + "id=" + i + ">"
					+ sent.getSource() + "</a>");
			i++;

		}

		out_n.println("</body>");
		out_n.println("</html>");
		out_n.close();
	}
	
	public void outputSummary_T(String topic, int iterTime) {
		String outputDir = "/home/peng/Develop/Workspace/NLP/data/CIKM2012/Output/summary";

		PrintWriter out_t = FileOperation.getPrintWriter(new File(outputDir),
				String.valueOf(topic + "." + iterTime + ".T"));

		// System.out.println(currentSummary.length());

		out_t.println("<html>");
		out_t.println("<body bgcolor=\"white\">");

		int i = 1;
		for (Instance sent : currentSummary) {

			out_t.println("<a name=\"" + i + "\">[" + i + "]</a> "
					+ "<a href=\"#" + i + "\" " + "id=" + i + ">"
					+ sent.getSource() + "</a>");
			i++;

		}

		out_t.println("</body>");
		out_t.println("</html>");
		out_t.close();
	}

	public void outputSummary(String topic, int iterTime)
			throws LpSolveException {

		solver.setVerbose(LpSolve.IMPORTANT);
		double[] var = solver.getPtrVariables();
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < var.length; i++) {
			if (var[i] == 1.0) {
				Instance sent = sentenceList.get(i);
				currentSummary.add(sent);

			}
		}

		String outputDir = "/home/peng/Develop/Workspace/NLP/data/CIKM2012/Output/summary";

		PrintWriter out = FileOperation.getPrintWriter(new File(outputDir),
				String.valueOf(topic + "." + iterTime));

		PrintWriter out_t = FileOperation.getPrintWriter(new File(outputDir),
				String.valueOf(topic + "." + iterTime + ".T"));

		PrintWriter out_n = FileOperation.getPrintWriter(new File(outputDir),
				String.valueOf(topic + "." + iterTime + ".N"));

		// System.out.println(currentSummary.length());
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
			if (sent.getName().toString().endsWith("_G")) {
				out_n.println("<a name=\"" + i + "\">[" + i + "]</a> "
						+ "<a href=\"#" + i + "\" " + "id=" + i + ">"
						+ sent.getSource() + "</a>");
				i++;
			}

		}

		i = 1;
		for (Instance sent : currentSummary) {
			// System.out.println(sent.getName() + ":" + sent.getSource());
			if (sent.getName().toString().endsWith("_T")) {
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
	
	public void runTwitters(int summaryLength_T, int averageNumCluster)
			throws LpSolveException {

		InstanceList localList = new InstanceList(null);

		HashMap<String, Integer> labelMap = new HashMap<String, Integer>();
		int index = 0;
		for (Instance sent : sentenceList) {

			Instance inst = new Instance(sent.getSource(), sent.getTarget(),
					sent.getName(), sent.getSource());
			if (sent.getName().toString().endsWith("_T")) {
				localList.add(inst);
				String label = (String) sent.getTarget();
				if (!labelMap.containsKey(label)) {
					labelMap.put(label, index++);
				}
			}

		}

		HashMap<Instance, Integer> sentIdxMap = new HashMap<Instance, Integer>();
		int l = 0;
		int[] labels = new int[localList.size()];
		for (Instance sent : localList) {
			sentIdxMap.put(sent, l);
			labels[l++] = labelMap.get(sent.getTarget());

		}
		int numLabels = labelMap.keySet().size();
		Clustering clusters = null;
		if (numLabels != averageNumCluster) {

			Clustering tmp = new Clustering(localList, numLabels, labels);
			int maxClusterid = -1;
			int MaxSize = -1;
			for (int i = 0; i < numLabels; i++) {
				if (tmp.getCluster(i).size() >= MaxSize) {
					MaxSize = tmp.getCluster(i).size();
					maxClusterid = i;
				}
			}
			int newChangedNumLabels = averageNumCluster - numLabels + 1;
			int startid = numLabels + 1;
			int[] indexs = tmp.getIndicesWithLabel(maxClusterid);
			for (int j = 0, freq = 0; j < indexs.length; j++, freq++) {
				int idx = indexs[j];
				int newLabel = startid + (freq % newChangedNumLabels);
				tmp.setLabel(idx, newLabel);
			}

			int[] tmplabels = tmp.getLabels();
			HashMap<Integer, Integer> tmpMap = new HashMap<Integer, Integer>();
			int val = 0;
			for (int i = 0; i < tmplabels.length; i++) {
				if (!tmpMap.containsKey(tmplabels[i]))
					tmpMap.put(tmplabels[i], val++);
			}

			// wrong mapping
			int[] newLabels = new int[labels.length];
			for (int i = 0; i < labels.length; i++) {
				newLabels[i] = tmpMap.get(tmplabels[i]);
			}

			clusters = new Clustering(localList, averageNumCluster, newLabels);

			numLabels = averageNumCluster;
		} else {
			clusters = new Clustering(localList, averageNumCluster, labels);
		}

		// begin ranking in each cluster

		labels = new int[clusters.getNumInstances()];
		InstanceList rankedSentenceList = new InstanceList(new PipeLine());
		int j = 0;
		for (int c = 0; c < clusters.numLabels; c++) {

			PipeLine pipeLine = new PipeLine();
			pipeLine.addPipe(new CharSequence2TokenSequence());
			pipeLine.addPipe(new TokenSequenceLowercase());
			pipeLine.addPipe(new TokenSequenceRemoveStopwords());
			pipeLine.addPipe(new TokenSequence2FeatureSequence());
			pipeLine.addPipe(new FeatureSequence2FeatureVector());

			InstanceList tf_fvs = new InstanceList(pipeLine);

			InstanceList sentList = clusters.getCluster(c);
			tf_fvs.addThruPipe(sentList.iterator());
			pipeLine = new PipeLine();
			pipeLine.addPipe(new FeatureIDFPipe(tf_fvs));
			InstanceList tf_idf_fvs = new InstanceList(pipeLine);
			tf_idf_fvs.addThruPipe(tf_fvs.iterator());

			for (int g = 0; g < sentList.size(); g++) {
				Instance s = sentList.get(g);
				s.setData(tf_idf_fvs.get(g).getData());
			}

			double threshold = 0.2;
			double damping = 0.1;
			double error = 0.1;

			LexRank lr = new LexRank(tf_idf_fvs, threshold, damping, error);
			lr.rank();
			double[] L = lr.getScore();

			ArrayList<Double> score = new ArrayList<Double>();
			for (double d : L) {
				score.add(d);
			}

			int[] indices = clusters.getIndicesWithLabel(c);
			HashMap<Integer, Double> map = new HashMap<Integer, Double>();
			for (int i = 0; i < L.length; i++) {
				map.put(indices[i], score.get(i));
			}
			LinkedHashMap ranked = RankMap.sortHashMapByValues(map, false);
			Iterator iter = ranked.keySet().iterator();
			while (iter.hasNext()) {
				int idx = (Integer) iter.next();
				Instance s = clusters.getInstances().get(idx);
				rankedSentenceList.add(s);
				labels[j++] = clusters.getLabel(idx);
			}
		}

		Clustering rankedClusters = new Clustering(rankedSentenceList,
				numLabels, labels);

		solve_T(rankedClusters, summaryLength_T);

	}

	// smooth clustering
	public void runNews(int summaryLength_N, int averageNumCluster)
			throws LpSolveException {

		InstanceList localList = new InstanceList(null);

		HashMap<String, Integer> labelMap = new HashMap<String, Integer>();
		int index = 0;
		for (Instance sent : sentenceList) {

			Instance inst = new Instance(sent.getSource(), sent.getTarget(),
					sent.getName(), sent.getSource());
			if (sent.getName().toString().endsWith("_G")) {
				localList.add(inst);
				String label = (String) sent.getTarget();
				if (!labelMap.containsKey(label)) {
					labelMap.put(label, index++);
				}
			}

		}

		HashMap<Instance, Integer> sentIdxMap = new HashMap<Instance, Integer>();
		int l = 0;
		int[] labels = new int[localList.size()];
		for (Instance sent : localList) {
			sentIdxMap.put(sent, l);
			labels[l++] = labelMap.get(sent.getTarget());

		}
		int numLabels = labelMap.keySet().size();
		Clustering clusters = null;
		if (numLabels != averageNumCluster) {

			Clustering tmp = new Clustering(localList, numLabels, labels);
			int maxClusterid = -1;
			int MaxSize = -1;
			for (int i = 0; i < numLabels; i++) {
				if (tmp.getCluster(i).size() >= MaxSize) {
					MaxSize = tmp.getCluster(i).size();
					maxClusterid = i;
				}
			}
			int newChangedNumLabels = averageNumCluster - numLabels + 1;
			int startid = numLabels + 1;
			int[] indexs = tmp.getIndicesWithLabel(maxClusterid);
			for (int j = 0, freq = 0; j < indexs.length; j++, freq++) {
				int idx = indexs[j];
				int newLabel = startid + (freq % newChangedNumLabels);
				tmp.setLabel(idx, newLabel);
			}

			int[] tmplabels = tmp.getLabels();
			HashMap<Integer, Integer> tmpMap = new HashMap<Integer, Integer>();
			int val = 0;
			for (int i = 0; i < tmplabels.length; i++) {
				if (!tmpMap.containsKey(tmplabels[i]))
					tmpMap.put(tmplabels[i], val++);
			}

			// wrong mapping
			int[] newLabels = new int[labels.length];
			for (int i = 0; i < labels.length; i++) {
				newLabels[i] = tmpMap.get(tmplabels[i]);
			}

			clusters = new Clustering(localList, averageNumCluster, newLabels);

			numLabels = averageNumCluster;
		} else {
			clusters = new Clustering(localList, averageNumCluster, labels);
		}

		// begin ranking in each cluster

		labels = new int[clusters.getNumInstances()];
		InstanceList rankedSentenceList = new InstanceList(new PipeLine());
		int j = 0;
		for (int c = 0; c < clusters.numLabels; c++) {

			PipeLine pipeLine = new PipeLine();
			pipeLine.addPipe(new CharSequence2TokenSequence());
			pipeLine.addPipe(new TokenSequenceLowercase());
			pipeLine.addPipe(new TokenSequenceRemoveStopwords());
			pipeLine.addPipe(new TokenSequence2FeatureSequence());
			pipeLine.addPipe(new FeatureSequence2FeatureVector());

			InstanceList tf_fvs = new InstanceList(pipeLine);

			InstanceList sentList = clusters.getCluster(c);
			tf_fvs.addThruPipe(sentList.iterator());
			pipeLine = new PipeLine();
			pipeLine.addPipe(new FeatureIDFPipe(tf_fvs));
			InstanceList tf_idf_fvs = new InstanceList(pipeLine);
			tf_idf_fvs.addThruPipe(tf_fvs.iterator());

			for (int g = 0; g < sentList.size(); g++) {
				Instance s = sentList.get(g);
				s.setData(tf_idf_fvs.get(g).getData());
			}

			double threshold = 0.1;
			double damping = 0.1;
			double error = 0.1;

			LexRank lr = new LexRank(tf_idf_fvs, threshold, damping, error);
			lr.rank();
			double[] L = lr.getScore();

			ArrayList<Double> score = new ArrayList<Double>();
			for (double d : L) {
				score.add(d);
			}

			int[] indices = clusters.getIndicesWithLabel(c);
			HashMap<Integer, Double> map = new HashMap<Integer, Double>();
			for (int i = 0; i < L.length; i++) {
				map.put(indices[i], score.get(i));
			}
			LinkedHashMap ranked = RankMap.sortHashMapByValues(map, false);
			Iterator iter = ranked.keySet().iterator();
			while (iter.hasNext()) {
				int idx = (Integer) iter.next();
				Instance s = clusters.getInstances().get(idx);
				rankedSentenceList.add(s);
				labels[j++] = clusters.getLabel(idx);
			}
		}

		Clustering rankedClusters = new Clustering(rankedSentenceList,
				numLabels, labels);

		solve_N(rankedClusters, summaryLength_N);

	}

	public void run(int summaryLength_T, int summaryLength_N)
			throws LpSolveException {

		InstanceList localList = new InstanceList(null);

		HashMap<String, Integer> labelMap = new HashMap<String, Integer>();
		int index = 0;
		for (Instance sent : sentenceList) {

			Instance inst = new Instance(sent.getSource(), sent.getTarget(),
					sent.getName(), sent.getSource());
			localList.add(inst);

			String label = (String) sent.getTarget();
			if (!labelMap.containsKey(label)) {
				labelMap.put(label, index++);
			}

		}

		HashMap<Instance, Integer> sentIdxMap = new HashMap<Instance, Integer>();
		int l = 0;
		int[] labels = new int[sentenceList.size()];
		for (Instance sent : sentenceList) {
			sentIdxMap.put(sent, l);
			labels[l++] = labelMap.get(sent.getTarget());

		}
		int numLabels = labelMap.keySet().size();
		Clustering clusters = new Clustering(localList, numLabels, labels);

		// begin ranking in each cluster

		labels = new int[localList.size()];
		InstanceList rankedSentenceList = new InstanceList(new PipeLine());
		int j = 0;
		for (InstanceList sentList : clusters.getClusters()) {

			PipeLine pipeLine = new PipeLine();
			pipeLine.addPipe(new CharSequence2TokenSequence());
			pipeLine.addPipe(new TokenSequenceLowercase());
			pipeLine.addPipe(new TokenSequenceRemoveStopwords());
			pipeLine.addPipe(new TokenSequence2FeatureSequence());
			pipeLine.addPipe(new FeatureSequence2FeatureVector());

			InstanceList tf_fvs = new InstanceList(pipeLine);
			tf_fvs.addThruPipe(sentList.iterator());
			pipeLine = new PipeLine();
			pipeLine.addPipe(new FeatureIDFPipe(tf_fvs));
			InstanceList tf_idf_fvs = new InstanceList(pipeLine);
			tf_idf_fvs.addThruPipe(tf_fvs.iterator());

			for (int g = 0; g < sentList.size(); g++) {
				Instance s = sentList.get(g);
				s.setData(tf_idf_fvs.get(g).getData());
			}

			double threshold = 0.15;
			double damping = 0.15;
			double error = 0.1;

			LexRank lr = new LexRank(tf_idf_fvs, threshold, damping, error);
			lr.rank();
			double[] L = lr.getScore();

			ArrayList<Double> score = new ArrayList<Double>();
			for (double d : L) {
				score.add(d);
			}

			HashMap<Instance, Double> map = new HashMap<Instance, Double>();
			for (int i = 0; i < sentList.size(); i++) {
				map.put(sentList.get(i), score.get(i));
			}
			LinkedHashMap ranked = RankMap.sortHashMapByValues(map, false);
			Iterator iter = ranked.keySet().iterator();
			while (iter.hasNext()) {
				Instance s = (Instance) iter.next();
				labels[j++] = labelMap.get(s.getTarget());
				rankedSentenceList.add(s);
			}
		}

		Clustering rankedClusters = new Clustering(rankedSentenceList,
				numLabels, labels);

		InstanceList[] cs = rankedClusters.getClusters();

		InstanceList cluster_t = new InstanceList(null);
		InstanceList cluster_n = new InstanceList(null);
		Set<String> numlabels_t = new HashSet<String>();
		Set<String> numlabels_n = new HashSet<String>();
		for (int c = 0; c < cs.length; c++) {
			InstanceList cluster = cs[c];
			for (int t = 0; t < cluster.size(); t++) {
				Instance sent = cluster.get(t);
				if (sent.getName().toString().endsWith("_T")) {
					cluster_t.add(sent);
					numlabels_t.add((String) sent.getTarget());
				} else if (sent.getName().toString().endsWith("_G")) {
					cluster_n.add(sent);
					numlabels_n.add((String) sent.getTarget());
				}
			}

		}

		int[] labels_t = new int[cluster_t.size()];
		int[] labels_n = new int[cluster_n.size()];
		int a = 0;
		int b = 0;
		for (int c = 0; c < cs.length; c++) {
			InstanceList cluster = cs[c];
			for (int t = 0; t < cluster.size(); t++) {
				Instance sent = cluster.get(t);
				if (sent.getName().toString().endsWith("_T")) {
					labels_t[a++] = labels[sentIdxMap.get(sent)];
				} else if (sent.getName().toString().endsWith("_G")) {
					labels_n[b++] = labels[sentIdxMap.get(sent)];
				}
			}
		}

		if ((cluster_t.size() + cluster_n.size()) != labels.length) {
			System.out.println();
		}

		Clustering clusters_n = new Clustering(cluster_n, numlabels_n.size(),
				labels_n);
		solve_N(clusters_n, summaryLength_N);

		Clustering clusters_t = new Clustering(cluster_t, numlabels_t.size(),
				labels_t);
		// solve_T(clusters_t, summaryLength_t);

	}
}
