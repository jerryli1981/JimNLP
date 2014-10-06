package edu.pengli.nlp.conference.cikm2012.generation;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Set;

import edu.pengli.nlp.conference.cikm2012.types.GoogleNewsCorpus;
import edu.pengli.nlp.conference.cikm2012.types.TweetCorpus;
import edu.pengli.nlp.platform.algorithms.lda.CCTAModel;
import edu.pengli.nlp.platform.algorithms.ranking.LexRank;
import edu.pengli.nlp.platform.pipe.CharSequence2TokenSequence;
import edu.pengli.nlp.platform.pipe.FeatureSequence2FeatureVector;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.TokenSequence2FeatureSequence;
import edu.pengli.nlp.platform.pipe.TokenSequenceLowercase;
import edu.pengli.nlp.platform.pipe.TokenSequenceRemoveStopwords;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Summary;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.Maths;
import edu.pengli.nlp.platform.util.RankMap;

public class simulateGoldStandard {

	Summary currentSummary;
	CCTAModel ccta;
	GoogleNewsCorpus gc;
	TweetCorpus tc;

	double maxIc = Double.MIN_VALUE, maxId = Double.MIN_VALUE;
	double minIc = Double.MAX_VALUE, minId = Double.MAX_VALUE;

	public simulateGoldStandard(GoogleNewsCorpus gc, TweetCorpus tc) {
		currentSummary = new Summary();
		this.gc = gc;
		this.tc = tc;

	}

	public simulateGoldStandard(CCTAModel ccta, GoogleNewsCorpus gc,
			TweetCorpus tc) {
		this.ccta = ccta;
		currentSummary = new Summary();
		this.gc = gc;
		this.tc = tc;
		setmaxminIcId();
	}

	private Summary generateNewsSummary() {
		InstanceList sentenceList = new InstanceList(null);
		for (Instance doc : gc) {
			InstanceList sentsList = (InstanceList) doc.getSource();
			for (Instance inst : sentsList) {
				Instance sent = new Instance(inst.getSource(), null,
						inst.getName(), inst.getSource());
				sentenceList.add(sent);
			}

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
//		pipeLine.addPipe(new FeatureVectorTFIDFWeight(tf_fvs));
		InstanceList tf_idf_fvs = new InstanceList(pipeLine);
		tf_idf_fvs.addThruPipe(tf_fvs.iterator());

		double threshold = 0.15;
		double damping = 0.15;
		double error = 0.1;
		LexRank lr = new LexRank(tf_idf_fvs, threshold, damping, error);
		lr.rank();
		double[] L = lr.getScore();

		assert (sentenceList.size() == L.length);

		HashMap<Integer, Double> rankmap = new HashMap<Integer, Double>();
		for (int i = 0; i < L.length; i++) {
			rankmap.put(i, L[i]);
		}
		LinkedHashMap map = RankMap.sortHashMapByValues(rankmap, false);
		Set<Integer> keys = map.keySet();
		Iterator<Integer> iter = keys.iterator();
		int count = 0;
		Summary sum = new Summary();
		while (iter.hasNext() && count < 10) {
			int key = iter.next();
			sum.add(tf_idf_fvs.get(key));
			count++;
		}
		return sum;
	}

	private void setmaxminIcId() {

		InstanceList sents_tc = new InstanceList(null);
		InstanceList sents_gc = new InstanceList(null);
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
//		pipeLine.addPipe(new FeatureVectorTFIDFWeight(tf_fvs));
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

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				double simcom = commonMeasureCCTA(sents_tc.get(i),
						sents_gc.get(j));
				if (simcom > maxIc) {
					maxIc = simcom;
				}
				if (simcom < minIc) {
					minIc = simcom;
				}
				double simconstra = diffMeasureCCTAT2N(sents_tc.get(i),
						sents_gc.get(j));
				if (simconstra > maxId) {
					maxId = simconstra;
				}
				if (simconstra < minId) {
					minId = simconstra;
				}
			}

		}
	}

	private ArrayList<Instance> findMostComplementaryTweets(
			InstanceList tf_idf_fvs, Instance inst) {
		ArrayList<Instance> ret = new ArrayList<Instance>();
		double[] L = new double[tf_idf_fvs.size()];
		for (int i = 0; i < tf_idf_fvs.size(); i++) {
			// L[i] = Maths.idf_modified_cosine(tf_idf_fvs.get(i), inst);
			L[i] = similarityCCTAT2N(tf_idf_fvs.get(i), inst, maxIc, minIc,
					maxId, minId);
		}
		HashMap<Integer, Double> rankmap = new HashMap<Integer, Double>();
		for (int i = 0; i < L.length; i++) {
			rankmap.put(i, L[i]);
		}
		LinkedHashMap map = RankMap.sortHashMapByValues(rankmap, false);
		Set<Integer> keys = map.keySet();
		Iterator<Integer> iter = keys.iterator();
		int count = 0;
		while (iter.hasNext() && count < 2) {
			int key = iter.next();
			ret.add(tf_idf_fvs.get(key));
			count++;
		}
		return ret;
	}

	private double similarityCCTAT2N(Instance inst_i, Instance inst_j,
			double maxIc, double minIc, double maxId, double minId) {

		double cos = Maths.idf_modified_cosine(inst_i, inst_j);
		double Ic = commonMeasureCCTA(inst_i, inst_j);
		double Id = diffMeasureCCTAT2N(inst_i, inst_j);

		double Icnorm = (Ic - minIc) / (maxIc - minIc);
		double Idnorm = (Id - minId) / (maxId - minId);

		double favor = 0.0;
		if (Icnorm >= Idnorm) {
			favor = Idnorm / Icnorm;
		} else if (Icnorm < Idnorm) {
			favor = Icnorm / Idnorm;
		}

		return cos * favor;
	}

	private double commonMeasureCCTA(Instance inst_i, Instance inst_j) {
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
			int N = 0;
			for (int i = 0; i < idx_i.length; i++) {
				int v = idx_i[i];
				u_i_z_prob_log += Math.log10(k_v_prob[k][v]);
				N++;

			}
			u_i_z_prob_log /= N;
			u_i_z_prob_log = Math.exp(u_i_z_prob_log);

			N = 0;
			double v_j_z_prob_log = 0.0;
			for (int j = 0; j < idx_j.length; j++) {
				int v = idx_j[j];
				v_j_z_prob_log += Math.log10(k_v_prob[k][v]);
				N++;
			}
			v_j_z_prob_log /= N;
			v_j_z_prob_log = Math.exp(v_j_z_prob_log);

			ret += (u_i_z_prob_log * v_j_z_prob_log);
			// ret += Math.log(u_i_z_prob_log) + Math.log(v_j_z_prob_log);
		}

		for (int a = 0; a < A; a++) {

			int N = 0;
			double u_i_y_prob_log = 0.0;
			for (int i = 0; i < idx_i.length; i++) {
				int v = idx_i[i];
				u_i_y_prob_log += Math.log10(a_v_prob[a][v]);
				N++;
			}

			u_i_y_prob_log /= N;
			u_i_y_prob_log = Math.exp(u_i_y_prob_log);

			N = 0;
			double v_j_y_prob_log = 0.0;
			for (int j = 0; j < idx_j.length; j++) {
				int v = idx_j[j];
				v_j_y_prob_log += Math.log10(a_v_prob[a][v]);
				N++;
			}
			v_j_y_prob_log /= N;
			v_j_y_prob_log = Math.exp(v_j_y_prob_log);

			ret += (u_i_y_prob_log * v_j_y_prob_log);
			// ret += Math.log(u_i_y_prob_log) + Math.log(v_j_y_prob_log);
		}

		for (int k = 0; k < K; k++) {
			for (int a = 0; a < A; a++) {
				int N = 0;
				double u_i_zy_prob_log = 0.0;
				for (int i = 0; i < idx_i.length; i++) {
					int v = idx_i[i];
					u_i_zy_prob_log += Math.log10(k_a_v_prob[k][a][v]);
					N++;
				}
				u_i_zy_prob_log /= N;
				u_i_zy_prob_log = Math.exp(u_i_zy_prob_log);

				N = 0;
				double v_j_zy_prob_log = 0.0;
				for (int j = 0; j < idx_j.length; j++) {
					int v = idx_j[j];
					v_j_zy_prob_log += Math.log10(k_a_v_prob[k][a][v]);
					N++;
				}
				v_j_zy_prob_log /= N;
				v_j_zy_prob_log = Math.exp(v_j_zy_prob_log);

				ret += (u_i_zy_prob_log * v_j_zy_prob_log);
				// ret += Math.log(u_i_zy_prob_log) + Math.log(v_j_zy_prob_log);

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

			N = 0;
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

			f_z_E3 += (n_i_z_n_prob_log * t_j_z_t_prob_log)
					/ (n_i_z_t_prob_log * t_j_z_n_prob_log);

		}

		double f_y_E3 = 0.0;
		for (int a = 0; a < A; a++) {
			int N = 0;
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

			N = 0;
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
			n_i_y_n_prob_log /= N;
			n_i_y_n_prob_log = Math.exp(n_i_y_n_prob_log);

			f_y_E3 += (n_i_y_n_prob_log * t_j_y_t_prob_log)
					/ (n_i_y_t_prob_log * t_j_y_n_prob_log);

		}

		double f_zy_E3 = 0.0;
		for (int k = 0; k < K; k++) {
			for (int a = 0; a < A; a++) {
				int N = 0;
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

				N = 0;
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

				f_zy_E3 += (n_i_zy_n_prob_log * t_j_zy_t_prob_log)
						/ (n_i_zy_t_prob_log * t_j_zy_n_prob_log);

			}
		}

		return f_z_E3 + f_y_E3 + f_zy_E3;
	}

	public HashMap<Instance, ArrayList<Instance>> run_com(int summaryLength) {
		Summary newSummary = generateNewsSummary();
		InstanceList sentenceList = new InstanceList(null);
		for (Instance doc : tc) {
			InstanceList sentsList = (InstanceList) doc.getSource();
			for (Instance inst : sentsList) {
				Instance sent = new Instance(inst.getSource(), null,
						inst.getName(), inst.getSource());
				sentenceList.add(sent);
			}
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
//		pipeLine.addPipe(new FeatureVectorTFIDFWeight(tf_fvs));
		InstanceList tf_idf_fvs = new InstanceList(pipeLine);
		tf_idf_fvs.addThruPipe(tf_fvs.iterator());

		HashMap<Instance, ArrayList<Instance>> pairs = new HashMap<Instance, ArrayList<Instance>>();

		for (Instance inst : newSummary) {
			currentSummary.add(inst);
			if (currentSummary.length() > summaryLength)
				break;
			else {
				ArrayList<Instance> tweets = findMostComplementaryTweets(
						tf_idf_fvs, inst);
				currentSummary.addAll(tweets);
				pairs.put(inst, tweets);
			}
		}

		return pairs;
	}

	public void outputSummary(String topic, int iterTime) {

		String outputDir = "../data/EMNLP2012/Output/summary";

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

}
