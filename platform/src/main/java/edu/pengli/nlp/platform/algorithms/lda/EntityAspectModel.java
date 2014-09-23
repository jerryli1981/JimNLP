package edu.pengli.nlp.platform.algorithms.lda;

import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Set;

import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.FeatureSequence;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.RankMap;

public class EntityAspectModel {

	private InstanceList instances; // here means collection of documents

	private int A; // number of aspects
	private int D; // number of documents
	private int V; // vocabulary size
	private double alpha;
	private double beta;
	private double gamma;
	private int numIters; // number of Gibbs sampling iteration

	private int[][] z; // aspect assignment for sentence in document
	private int[][][] y; // indicator assignment for word of sentence in
							// document. here y = 1,2,3

	private int[] a_cnt; // the number of sentences assigned to aspect a
	private int a_sum; // the total number of sentences
	private int[] t_cnt; // the number of words assigned to t
	private int t_sum; // the total number of words
	private int[] B_v_cnt; // the number of times word v has been assigned to
							// background word
	private int B_v_sum; // the total number of words assigned to Background
							// word
	private int[][] d_v_cnt; // the number of times word v has been assigned to
								// document word
	private int[] d_v_sum; // the total number of words assigned to document
							// word
	private int[][] a_v_cnt; // the number of times word v has been assigned to
								// aspect a
	private int[] a_v_sum; // the total number of words assigned to aspect a

	private double[] a_avg_cnt;
	private double a_avg_sum;
	private double[] t_avg_cnt;
	private double t_avg_sum;
	private double[] B_v_avg_cnt;
	private double B_v_avg_sum;
	private double[][] d_v_avg_cnt;
	private double[] d_v_avg_sum;
	private double[][] a_v_avg_cnt;
	private double[] a_v_avg_sum;


	private double[] a_prob; //theta
	private double[] t_prob;
	private double[] B_v_prob;
	private double[][] d_v_prob;
	private double[][] a_v_prob;

	double beta_sum;
	double gamma_sum;
	double alpha_sum;
	
	private double[] p;
	private double[] log_v_p;

	public EntityAspectModel(int numAspects, double alpha, double beta,
			double gamma, int numIters) {
		A = numAspects;
		this.numIters = numIters;
		this.alpha = alpha;
		this.beta = beta;
		this.gamma = gamma;
	}

	public void initEstimate(InstanceList instances) {

		int d, s, n;
		int a, t, v;

		Alphabet dict = instances.getDataAlphabet();
		V = dict.size();
		this.instances = instances;
		D = instances.size();
		
		beta_sum = V * beta;
		gamma_sum = 3 * gamma;
		alpha_sum = A * alpha;

		p = new double[A];
		log_v_p = new double[A];

		a_cnt = new int[A];
		for (a = 0; a < A; a++) {
			a_cnt[a] = 0;
		}
		a_sum = 0;

		t_cnt = new int[3];
		for (t = 0; t < 3; t++) {
			t_cnt[t] = 0;
		}
		t_sum = 0;

		B_v_cnt = new int[V];
		for (v = 0; v < V; v++) {
			B_v_cnt[v] = 0;
		}
		B_v_sum = 0;

		d_v_cnt = new int[D][V];
		d_v_sum = new int[D];
		for (d = 0; d < D; d++) {
			for (v = 0; v < V; v++) {
				d_v_cnt[d][v] = 0;
			}
			d_v_sum[d] = 0;
		}

		a_v_cnt = new int[A][V];
		a_v_sum = new int[A];
		for (a = 0; a < A; a++) {
			for (v = 0; v < V; v++) {
				a_v_cnt[a][v] = 0;
			}
			a_v_sum[a] = 0;
		}

		a_avg_cnt = new double[A];

		t_avg_cnt = new double[3];

		B_v_avg_cnt = new double[V];

		d_v_avg_cnt = new double[D][];
		d_v_avg_sum = new double[D];
		for (d = 0; d < D; d++) {
			d_v_avg_cnt[d] = new double[V];
		}

		a_v_avg_cnt = new double[A][];
		a_v_avg_sum = new double[A];
		for (a = 0; a < A; a++) {
			a_v_avg_cnt[a] = new double[V];
		}
		
		a_prob = new double[A];

		t_prob = new double[3];

		B_v_prob = new double[V];

		d_v_prob = new double[D][];
		for (d = 0; d < D; d++) {
			d_v_prob[d] = new double[V];
		}

		a_v_prob = new double[A][];
		for (a = 0; a < A; a++) {
			a_v_prob[a] = new double[V];
		}

		z = new int[D][];
		y = new int[D][][];
		for (d = 0; d < D; d++) {
			Instance doc = (Instance) instances.get(d);
			InstanceList sents = (InstanceList) doc.getData();
			int S = sents.size();
			z[d] = new int[S];
			y[d] = new int[S][];

			for (s = 0; s < S; s++) {
				FeatureSequence sent = (FeatureSequence) sents.get(s).getData();
				int N = sent.size();
				y[d][s] = new int[N];
			}
		}

		System.out.println("randomly initializing z and y...");

		for (d = 0; d < D; d++) {

			Instance doc = (Instance) instances.get(d);
			InstanceList sents = (InstanceList) doc.getData();
			int S = sents.size();
			for (s = 0; s < S; s++) {

				a = (int) Math.floor(Math.random() * A);
				z[d][s] = a;

				a_cnt[a]++;
				a_sum++; //total number of sentence

				FeatureSequence sent = (FeatureSequence) sents.get(s).getData();
				int N = sent.size();

				for (n = 0; n < N; n++) {

					t = (int) Math.floor(Math.random() * 3);
					y[d][s][n] = t;

					t_cnt[t]++;
					t_sum++; //total number of words

					v = sent.getIndexAtPosition(n); // v is word index in
													// dictionary

					if (t == 0) {
						B_v_cnt[v]++;
						B_v_sum++;
					} else if (t == 1) {
						d_v_cnt[d][v]++;
						d_v_sum[d]++;
					} else {
						a_v_cnt[a][v]++;
						a_v_sum[a]++;
					}
				}
			}
		}

	}

	public void estimate() {

		for (int i = 0; i < numIters; i++) {
			System.out.println("Iteration " + (i + 1) + " ...");
			sweep(instances);
		}// end iterations

		System.out.println("Taking samples...");
		for (int i = 0; i < 100; i++) {
			sweep(instances);
			if (i % 10 == 0) {
				updateAvgCnt();
			}
		}

		 averageCnt();
		 
		 updateProbabilities();

	}

	private void sweep(InstanceList instances) {
		for (int d = 0; d < D; d++) {
			Instance doc = (Instance) instances.get(d);
			InstanceList sents = (InstanceList) doc.getData();
			for (int s = 0; s < sents.size(); s++) {
				FeatureSequence sent = (FeatureSequence) sents.get(s).getData();
				int N = sent.size();
				int a = sampling_z(d, s, sent);
				z[d][s] = a;
				for (int n = 0; n < N; n++) {
					y[d][s][n] = sampling_y(d, s, n, sent);
				}

			}

		}

	}

	private int sampling_z(int d, int s, FeatureSequence sent) {

		int N = sent.size();
		int n;

		int a, t, v;

		// *********************************************
		// decrease the counts

		a = z[d][s];

		a_cnt[a]--;
		a_sum--;

		for (n = 0; n < N; n++) {

			t = y[d][s][n];

			t_cnt[t]--;
			t_sum--;

			v = sent.getIndexAtPosition(n);

			if (t == 0) {
				B_v_cnt[v]--;
				B_v_sum--;
			} else if (t == 1) {
				d_v_cnt[d][v]--;
				d_v_sum[d]--;
			} else {
				a_v_cnt[a][v]--;
				a_v_sum[a]--;
			}

		}

		// *********************************************
		// draw a new topic based on the assignments of all other hidden
		// variables

		a = draw_z(d, s, sent);

		// *********************************************
		// Increase the counts

		z[d][s] = a;

		a_cnt[a]++;
		a_sum++;

		for (n = 0; n < N; n++) {

			// *********************************

			t = y[d][s][n];

			t_cnt[t]++;
			t_sum++;

			v = sent.getIndexAtPosition(n);

			if (t == 0) {
				B_v_cnt[v]++;
				B_v_sum++;
			} else if (t == 1) {
				d_v_cnt[d][v]++;
				d_v_sum[d]++;
			} else {
				a_v_cnt[a][v]++;
				a_v_sum[a]++;
			}

			// *********************************
		}

		return a;
	}

	private int draw_z(int d, int s, FeatureSequence sent) {
		int N = sent.size();
		int n;
		int a, t, v;
		double a_p, v_p;
		double max_log_v_p;

		// markov chain
		for (a = 0; a < A; a++) {
			a_p = (a_cnt[a] + alpha) / (a_sum + alpha_sum);
			int my_a_v_sum = 0;
			HashMap<Integer, Integer> my_a_v_cnt = new HashMap<Integer, Integer>();
			for (n = 0; n < N; n++) {
				t = y[d][s][n];
				if (t == 2) {
					v = sent.getIndexAtPosition(n);
					if (!my_a_v_cnt.containsKey(v)) {
						my_a_v_cnt.put(v, 1);
					} else {
						int cnt = my_a_v_cnt.get(v);
						my_a_v_cnt.put(v, (++cnt));
					}
					my_a_v_sum++;
				}

			}

			// gibbs sampling
			log_v_p[a] = 0.0;
			Set<Integer> keys = my_a_v_cnt.keySet();
			Iterator iter = keys.iterator();
			while (iter.hasNext()) {
				v = (Integer) iter.next();
				int cnt = my_a_v_cnt.get(v);
				for (int i = 0; i < cnt; i++) {
					log_v_p[a] += Math.log(a_v_cnt[a][v] + beta + i);
				}
			}

			for (int i = 0; i < my_a_v_sum; i++) {
				log_v_p[a] -= Math.log(a_v_sum[a] + beta_sum + i);
			}

			p[a] = a_p;

		}

		max_log_v_p = log_v_p[0];
		for (a = 1; a < A; a++) {
			if (log_v_p[a] > max_log_v_p) {
				max_log_v_p = log_v_p[a];
			}
		}
		// smoothing
		for (a = 0; a < A; a++) {
			log_v_p[a] -= max_log_v_p;
			v_p = Math.exp(log_v_p[a]);
			p[a] *= v_p;
		}

		for (a = 1; a < A; a++) {
			p[a] += p[a - 1];
		}

		if (p[A - 1] == 0) {
			System.out.println("Error: p[A - 1] is zero!");
			System.exit(0);
		}

		// scaled sample because of unnormalized p[]
//		double u = (new Random()).nextInt(Integer.MAX_VALUE)/Integer.MAX_VALUE * p[A - 1];
		// //doesn't work
    	double u = Math.random() * p[A - 1];

		for (a = 0; a < A; a++) {
			if (p[a] > u) // sample topic w.r.t distribution p
				break;
		}
		if (a == 10) {
			System.out.println("debug");
		}
		return a;
	}

	private int sampling_y(int d, int s, int n, FeatureSequence sent) {

		int N = sent.size();
		int a = z[d][s];
		int v = sent.getIndexAtPosition(n);

		int t;

		// *******************************************************
		// Decrease the counts

		t = y[d][s][n];

		t_cnt[t]--;
		t_sum--;

		if (t == 0) {
			B_v_cnt[v]--;
			B_v_sum--;
		} else if (t == 1) {
			d_v_cnt[d][v]--;
			d_v_sum[d]--;
		} else {
			a_v_cnt[a][v]--;
			a_v_sum[a]--;
		}

		// *******************************************************

		t = draw_y(d, a, v);

		// *******************************************************
		// Increase the counts

		y[d][s][n] = t;

		t_cnt[t]++;
		t_sum++;

		if (t == 0) {
			B_v_cnt[v]++;
			B_v_sum++;
		} else if (t == 1) {
			d_v_cnt[d][v]++;
			d_v_sum[d]++;
		} else {
			a_v_cnt[a][v]++;
			a_v_sum[a]++;
		}
		return t;
	}

	private int draw_y(int d, int a, int v) {

		int t;
		double t_p, v_p;
		double[] q = new double[3];

		for (t = 0; t < 3; t++) {

			t_p = (t_cnt[t] + gamma) / (t_sum + gamma_sum);

			if (t == 0) {
				v_p = (B_v_cnt[v] + beta) / (B_v_sum + beta_sum);
			} else if (t == 1) {
				v_p = (d_v_cnt[d][v] + beta) / (d_v_sum[d] + beta_sum);
			} else {
				v_p = (a_v_cnt[a][v] + beta) / (a_v_sum[a] + beta_sum);
			}

			q[t] = t_p * v_p;
		}

		for (t = 1; t < 3; t++) {
			q[t] += q[t - 1];
		}

		if (q[2] == 0) {
			System.out.println("Error: q[2] is zero!");
			System.exit(0);
		}

	//	 double u = new Random().nextInt(Integer.MAX_VALUE)/Integer.MAX_VALUE* q[2];
		 double u = Math.random() * q[2];
		for (t = 0; t < 3; t++) {
			if (q[t] >= u) {
				break;
			}
		}

		return t;
	}

	private void updateAvgCnt() {
		int d, a, t, v;

		for (a = 0; a < A; a++) {
			a_avg_cnt[a] += a_cnt[a];
		}
		a_avg_sum += a_sum;

		for (t = 0; t < 3; t++) {
			t_avg_cnt[t] += t_cnt[t];
		}
		t_avg_sum += t_sum;

		for (v = 0; v < V; v++) {
			B_v_avg_cnt[v] += B_v_cnt[v];
		}
		B_v_avg_sum += B_v_sum;

		for (d = 0; d < D; d++) {
			for (v = 0; v < V; v++) {
				d_v_avg_cnt[d][v] += d_v_cnt[d][v];
			}
			d_v_avg_sum[d] += d_v_sum[d];
		}

		for (a = 0; a < A; a++) {
			for (v = 0; v < V; v++) {
				a_v_avg_cnt[a][v] += a_v_cnt[a][v];
			}
			a_v_avg_sum[a] += a_v_sum[a];
		}

	}

	private void averageCnt() {
		int a, t, v, d;
		for (a = 0; a < A; a++) {
			a_avg_cnt[a] /= 10.0;
		}
		a_avg_sum /= 10.0;

		for (t = 0; t < 3; t++) {
			t_avg_cnt[t] /= 10.0;
		}
		t_avg_sum /= 10.0;

		for (v = 0; v < V; v++) {
			B_v_avg_cnt[v] /= 10.0;
		}
		B_v_avg_sum /= 10.0;

		for (d = 0; d < D; d++) {
			for (v = 0; v < V; v++) {
				d_v_avg_cnt[d][v] /= 10.0;
			}
			d_v_avg_sum[d] /= 10.0;
		}

		for (a = 0; a < A; a++) {
			for (v = 0; v < V; v++) {
				a_v_avg_cnt[a][v] /= 10.0;
			}
			a_v_avg_sum[a] /= 10.0;
		}
	}
	
	private void updateProbabilities(){
		int d, a, t, v;

		for (a = 0; a < A; a++) { // theta a
			// a_prob[a] = (a_cnt[a] + alpha) / (a_sum + alpha_sum);
			a_prob[a] = (a_avg_cnt[a] + alpha) / (a_avg_sum + alpha_sum);
		}

		for (t = 0; t < 3; t++) { //pi t
			 //t_prob[t] = (t_cnt[t] + gamma) / (t_sum + gamma_sum);
			t_prob[t] = (t_avg_cnt[t] + gamma) / (t_avg_sum + gamma_sum);
		}

		for (v = 0; v < V; v++) { //phi B
			// B_v_prob[v] = (B_v_cnt[v] + beta) / (B_v_sum + beta_sum);
			B_v_prob[v] = (B_v_avg_cnt[v] + beta) / (B_v_avg_sum + beta_sum);
		}

		for (d = 0; d < D; d++) {
			for (v = 0; v < V; v++) { //phi d
				//d_v_prob[d][v] = (d_v_cnt[d][v] + beta) / (d_v_sum[d] + beta_sum);
				d_v_prob[d][v] = (d_v_avg_cnt[d][v] + beta)/ (d_v_avg_sum[d] + beta_sum);
			}
		}

		for (a = 0; a < A; a++) {
			for (v = 0; v < V; v++) { //phi a
				//a_v_prob[a][v] = (a_v_cnt[a][v] + beta) / (a_v_sum[a] + beta_sum);
				a_v_prob[a][v] = (a_v_avg_cnt[a][v] + beta)/ (a_v_avg_sum[a] + beta_sum);
			}
		}
	}

	public void predict_labels(InstanceList instances){
		
		int D, S;
		int d, s;

		D = instances.size();

		for (d = 0; d < D; d++) {
			Instance doc = (Instance) instances.get(d);
			InstanceList sents = (InstanceList) doc.getData();
			S = sents.size();
			for (s = 0; s < S; s++) {
				FeatureSequence sent = (FeatureSequence) sents.get(s).getData();
				assign_y_z(d, s, sent);
			}
		}
	}
	
	private void assign_y_z(int d, int s, FeatureSequence sent){
		int N = sent.size();

		int n, a, t, v;

		double max_prob, prob;
		int best_y;
		double max_log_sent_prob = 0.0, log_sent_prob;
		int best_z = 0;

		int[][] tmp_y = new int[A][];
		for (a = 0; a < A; a++) {
			tmp_y[a] = new int[N];
		}

		for (a = 0; a < A; a++) {

			log_sent_prob = 0.0;

			for (n = 0; n < N; n++) {
				v = sent.getIndexAtPosition(n);
				max_prob = 0.0;
				best_y = -1;

				for (t = 0; t < 3; t++) {

					if (t == 0) {
						prob = t_prob[t] * B_v_prob[v];
					} else if (t == 1) {
						prob = t_prob[t] * d_v_prob[d][v];
					} else {
						prob = t_prob[t] * a_v_prob[a][v];
					}

					if (prob > max_prob) {
						max_prob = prob;
						best_y = t;
					}
				}

				// this is caused by you counting dict begin from 0 or 1,jingjing is from 1, I am from 0
				if (best_y == -1) {
					System.out.println("Error: best_y is -1");
					System.exit(0);
				}

				log_sent_prob += Math.log(max_prob);
				tmp_y[a][n] = best_y;
			} // for(n = 0; n < N; n++)

			if (a > 0) {
				if (log_sent_prob > max_log_sent_prob) {
					max_log_sent_prob = log_sent_prob;
					best_z = a;
				}
			} else {
				max_log_sent_prob = log_sent_prob;
				best_z = a;
			}

		} // for(a = 0; a < A; a++)

		z[d][s] = best_z;

		for (n = 0; n < N; n++) {
			y[d][s][n] = tmp_y[best_z][n];
		}

	}
	
	public InstanceList getInstanceList() {
		return instances;
	}
	public int[][] getTopicAssignment() {
		return z;
	}
	
	public void output_labels(PrintWriter out, InstanceList instances){
		int D, S, N;
		int d, s, n, a, t;

		D = instances.size();
		Alphabet dict = instances.getDataAlphabet();

		for (d = 0; d < D; d++) {
			Instance doc = (Instance) instances.get(d);
			InstanceList sents = (InstanceList) doc.getData();
			S = sents.size();
            
		//	out.println(doc.getName()+" "+S);
			
			for (s = 0; s < S; s++) {
				FeatureSequence sent = (FeatureSequence) sents.get(s).getData();
				a = z[d][s] + 1;

				out.print(a +": ");
				N = sent.size();
				for (n = 0; n < N; n++) {
					int v = sent.getIndexAtPosition(n);
                    out.print(dict.lookupObject(v)+"/");

					t = y[d][s][n];
					if (t == 0) {
						out.print("B/0");
					} else if (t == 1) {
						out.print("D/0");
					} else {
						out.print("A/");
						double prob = a_v_prob[a - 1][v];
						if (prob >= 0.001) {
							out.print(prob);
						} else {
							out.print(0);
						}
					}
					out.print(" ");

				}
				out.print("\n");
			}
		}

		out.close();
	}
	
	public void output_model(){
		
		int topK=10;
    	Alphabet dict = instances.getDataAlphabet();
 		if (topK > V) {
 			topK = V;
 		}
 		for (int k = 0; k < A; k++) {
 			System.out.println("Topic " + k + "th:");
 			HashMap<Integer, Double> map = new HashMap<Integer, Double>();
 			for (int v = 0; v < V; v++) {
 				double pro = a_v_prob[k][v];
 				map.put(v, pro); // phi[k][w] compute by nw[w][k], w is the idx
 									// of words in dictionary
 			}
 			LinkedHashMap rankedMap = RankMap.sortHashMapByValues(map, false);
 			Set<Integer> keys = rankedMap.keySet();
 			Iterator iter = keys.iterator();
 			int i = 0;
 			while (iter.hasNext() && i < topK) {
 				int idx = (Integer) iter.next();
 				Object obj = dict.lookupObject(idx);
 				double pro = map.get(idx);
 				System.out.println("\t" + obj.toString() + " " + pro);
 				i++;
 			}

 		}
		
	}

}
