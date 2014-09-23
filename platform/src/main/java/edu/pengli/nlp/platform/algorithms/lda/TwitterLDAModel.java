package edu.pengli.nlp.platform.algorithms.lda;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Set;

import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.FeatureSequence;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.RankMap;

public class TwitterLDAModel implements Serializable{

	private InstanceList instances; // here means collection of tweets

	private int K; // number of topics
	private int U; // number of users
	private int V; // vocabulary size
	private double alpha;
	private double beta;
	private double gamma;
	private int numIters; // number of Gibbs sampling iteration

	private int[][] z; // topic assignment for tweets in each user z_{u,s}
	private int[][][] y; // indicator assignment for word of tweets in here y = 1,2 y_{u,s,n}

	//below variables for sample z_{u,s}
	private int[] k_cnt; // the number of tweets assigned to topic k : C_{k}^{K}
	private int k_sum; // the total number of tweets
	private int[][] k_v_cnt; // the number of times word v has been assigned to topic k : C_{v}^{k}
	private int[] k_v_sum; // the total number of words assigned to topic k : C_{.}^{k}

	//below variables for sample y_{u,s,n}
	private int[] B_v_cnt; // the number of times word v has been assigned to background word : C_{w_{u,s,n}}^B
	private int B_v_sum; // the total number of words assigned to Background word : C_{.}^B	
	private int[] t_cnt; // the number of words assigned to t 
	private int t_sum; // the total number of words
	
  
	private double[] k_avg_cnt;
	private double k_avg_sum;
	private double[] t_avg_cnt;
	private double t_avg_sum;
	private double[] B_v_avg_cnt;
	private double B_v_avg_sum;
	private double[][] k_v_avg_cnt;
	private double[] k_v_avg_sum;

	private double[] t_prob; // pi_{t}
	private double[] B_v_prob; // phi_{v}^{B}
	private double[][] k_v_prob; // phi_{v}^{K}
	
	private double[] p;
	private double[] log_v_p;

	private double beta_sum;
	private double gamma_sum;
	private double alpha_sum;
	
	

	public TwitterLDAModel(int numTopics, double alpha, double beta,
			double gamma, int numIters) {
		K = numTopics;
		this.numIters = numIters;
		this.alpha = alpha;
		this.beta = beta;
		this.gamma = gamma;
	}
	
	public InstanceList getInstanceList(){
		return instances;
	}
	public Alphabet getAlphabet(){
		return instances.getDataAlphabet();
	}
	public double[][] getTopicWordDistribution(){
		return k_v_prob;
	}

	public void initEstimate(InstanceList instances) {
		int u, s, n;
		int k, t, v;

		Alphabet dict = instances.getDataAlphabet();
		V = dict.size();
		this.instances = instances;
		U = instances.size();
		
		beta_sum = V * beta;
		gamma_sum = 2 * gamma;
		alpha_sum = K * alpha;
		
		z = new int[U][];
		y = new int[U][][];
		for (u = 0; u < U; u++) {
			Instance user = (Instance) instances.get(u);
			InstanceList sents = (InstanceList) user.getData();
			int S = sents.size();
			z[u] = new int[S];
			y[u] = new int[S][];

			for (s = 0; s < S; s++) {
				FeatureSequence sent = (FeatureSequence) sents.get(s).getData();
				int N = sent.size();
				y[u][s] = new int[N];
			}
		}

		k_cnt = new int[K];
		for (k = 0; k < K; k++) {
			k_cnt[k] = 0;
		}
		k_sum = 0;
		
		k_v_cnt = new int[K][V];
		k_v_sum = new int[K];
		for (k = 0; k < K; k++) {
			for (v = 0; v < V; v++) {
				k_v_cnt[k][v] = 0;
			}
			k_v_sum[k] = 0;
		}
		
		B_v_cnt = new int[V];
		for (v = 0; v < V; v++) {
			B_v_cnt[v] = 0;
		}
		B_v_sum = 0;

		t_cnt = new int[3];
		for (t = 0; t < 3; t++) {
			t_cnt[t] = 0;
		}
		t_sum = 0;
		
		k_avg_cnt = new double[K];

		t_avg_cnt = new double[3];

		B_v_avg_cnt = new double[V];

		k_v_avg_cnt = new double[K][];
		k_v_avg_sum = new double[K];
		for (k = 0; k < K; k++) {
			k_v_avg_cnt[k] = new double[V];
		}
		
		t_prob = new double[3];

		B_v_prob = new double[V];

		k_v_prob = new double[K][];
		for (k = 0; k < K; k++) {
			k_v_prob[k] = new double[V];
		}
	
		p = new double[K];
		log_v_p = new double[K];


		System.out.println("randomly initializing z and y...");

		for (u = 0; u < U; u++) {

			
			Instance user = (Instance) instances.get(u);
			InstanceList sents = (InstanceList) user.getData();
			int S = sents.size();
			for (s = 0; s < S; s++) {

				k = (int) Math.floor(Math.random() * K);
				z[u][s] = k;
				
			
				k_cnt[k]++;
				k_sum++;

				FeatureSequence sent = (FeatureSequence) sents.get(s).getData();
				int N = sent.size();

				for (n = 0; n < N; n++) {

					t = (int) Math.floor(Math.random() * 2);
					y[u][s][n] = t;

					t_cnt[t]++;
					t_sum++;

					v = sent.getIndexAtPosition(n); // v is word index in dictionary
					
					if (t == 0) {
						B_v_cnt[v]++;
						B_v_sum++;
					} else if (t == 1) {
						k_v_cnt[k][v]++;
						k_v_sum[k]++;
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
		for (int u = 0; u < U; u++) {
			Instance user = (Instance) instances.get(u);
			InstanceList sents = (InstanceList) user.getData();
			for (int s = 0; s < sents.size(); s++) {
				FeatureSequence sent = (FeatureSequence) sents.get(s).getData();
				int N = sent.size();
				int k = sampling_z(u, s, sent);
				z[u][s] = k;
				for (int n = 0; n < N; n++) {
					y[u][s][n] = sampling_y(u, s, n, sent);
				}

			}

		}

	}

	private int sampling_z(int u, int s, FeatureSequence sent) {
		
		int N = sent.size();
		int n;

		int k, t, v;

		// *********************************************
		// decrease the counts

		k = z[u][s];

		k_cnt[k]--;
		k_sum--;
		
		for (n = 0; n < N; n++) {

			t = y[u][s][n];

			t_cnt[t]--;
			t_sum--;

			v = sent.getIndexAtPosition(n);

			if (t == 0) {
				B_v_cnt[v]--;
				B_v_sum--;
			} else if (t == 1) {
				k_v_cnt[k][v]--;
				k_v_sum[k]--;
			} 

		}

		// *********************************************
		// draw a new topic based on the assignments of all other hidden
		// variables

		k = draw_z(u, s, sent);

		// *********************************************
		// Increase the counts

		z[u][s] = k;

		k_cnt[k]++;
		k_sum++;
		

		for (n = 0; n < N; n++) {

			// *********************************

			t = y[u][s][n];

			t_cnt[t]++;
			t_sum++;

			v = sent.getIndexAtPosition(n);

			if (t == 0) {
				B_v_cnt[v]++;
				B_v_sum++;
			} else if (t == 1) {
				k_v_cnt[k][v]++;
				k_v_sum[k]++;
			} 

			// *********************************
		}

		return k;

	}

	private int draw_z(int u, int s, FeatureSequence sent) {
		int N = sent.size();
		int n;
		int k, t, v;
		double k_p, v_p;
		double max_log_v_p;

		// markov chain
		for (k = 0; k < K; k++) {
			k_p = (k_cnt[k] + alpha) / (k_sum + alpha_sum);
			int my_k_v_sum = 0;
			HashMap<Integer, Integer> my_k_v_cnt = new HashMap<Integer, Integer>();
			for (n = 0; n < N; n++) {
				t = y[u][s][n];
				if (t == 1) {
					v = sent.getIndexAtPosition(n);
					if (!my_k_v_cnt.containsKey(v)) {
						my_k_v_cnt.put(v, 1);
					} else {
						int cnt = my_k_v_cnt.get(v);
						my_k_v_cnt.put(v, (++cnt));
					}
					my_k_v_sum++;
				}

			}

			// gibbs sampling
			log_v_p[k] = 0.0;
			Set<Integer> keys = my_k_v_cnt.keySet();
			Iterator iter = keys.iterator();
			while (iter.hasNext()) {
				v = (Integer) iter.next();
				int cnt = my_k_v_cnt.get(v);
				for (int i = 0; i < cnt; i++) {
					log_v_p[k] += Math.log(k_v_cnt[k][v] + beta + i);
				}
			}

			for (int i = 0; i < my_k_v_sum; i++) {
				log_v_p[k] -= Math.log(k_v_sum[k] + beta_sum + i);
			}

			p[k] = k_p;

		}

		max_log_v_p = log_v_p[0];
		for (k = 1; k < K; k++) {
			if (log_v_p[k] > max_log_v_p) {
				max_log_v_p = log_v_p[k];
			}
		}
		// smoothing
		for (k = 0; k < K; k++) {
			log_v_p[k] -= max_log_v_p;
			v_p = Math.exp(log_v_p[k]);
			p[k] *= v_p;
		}

		for (k = 1; k < K; k++) {
			p[k] += p[k - 1];
		}

		if (p[K - 1] == 0) {
			System.out.println("Error: p[A - 1] is zero!");
			System.exit(0);
		}

		// scaled sample because of unnormalized p[]
	//	double u = (new Random()).nextInt(Integer.MAX_VALUE)/Integer.MAX_VALUE * p[A - 1];
		// //doesn't work
	double X = Math.random() * p[K - 1];

		for (k = 0; k < K; k++) {
			if (p[k] > X) // sample topic w.r.t distribution p
				break;
		}

		return k;
	}
	


	private int sampling_y(int u, int s, int n, FeatureSequence sent) {
		
		int N = sent.size();
		int k = z[u][s];
		int v = sent.getIndexAtPosition(n);

		int t;

		// *******************************************************
		// Decrease the counts

		t = y[u][s][n];

		t_cnt[t]--;
		t_sum--;

		if (t == 0) {
			B_v_cnt[v]--;
			B_v_sum--;
		} else if (t == 1) {
			k_v_cnt[k][v]--;
			k_v_sum[k]--;
		} 

		// *******************************************************

		t = draw_y(u, k, v);

		// *******************************************************
		// Increase the counts

		y[u][s][n] = t;

		t_cnt[t]++;
		t_sum++;

		if (t == 0) {
			B_v_cnt[v]++;
			B_v_sum++;
		} else if (t == 1) {
			k_v_cnt[k][v]++;
			k_v_sum[k]++;
		} 
		return t;
	}

	private int draw_y(int u, int k, int v) {

		int t;
		double t_p, v_p = 0;
		double[] q = new double[2];

		for (t = 0; t < 2; t++) {

			t_p = (t_cnt[t] + gamma) / (t_sum + gamma_sum);

			if (t == 0) {
				v_p = (B_v_cnt[v] + beta) / (B_v_sum + beta_sum);
			} else if (t == 1) {
				v_p = (k_v_cnt[k][v] + beta) / (k_v_sum[k] + beta_sum);
			}
			q[t] = t_p * v_p;
		}

		for (t = 1; t < 2; t++) {
			q[t] += q[t - 1];
		}

		if (q[1] == 0) {
			System.out.println("Error: q[1] is zero!");
			System.exit(0);
		}

		// double u = new Random().nextInt(Integer.MAX_VALUE)/Integer.MAX_VALUE* q[2];
		 double X = Math.random() * q[1];
		for (t = 0; t < 2; t++) {
			if (q[t] >= X) {
				break;
			}
		}

		return t;
	}

	private void updateAvgCnt() {
		int u, k, t, v;

		for (k = 0; k < K; k++) {
			k_avg_cnt[k] += k_cnt[k];
		}
		k_avg_sum += k_sum;

		for (t = 0; t < 2; t++) {
			t_avg_cnt[t] += t_cnt[t];
		}
		t_avg_sum += t_sum;

		for (v = 0; v < V; v++) {
			B_v_avg_cnt[v] += B_v_cnt[v];
		}
		B_v_avg_sum += B_v_sum;


		for (k = 0; k < K; k++) {
			for (v = 0; v < V; v++) {
				k_v_avg_cnt[k][v] += k_v_cnt[k][v];
			}
			k_v_avg_sum[k] += k_v_sum[k];
		}
				
	}

	private void averageCnt() {
		int k, t, v, u;
		for (k = 0; k < K; k++) {
			k_avg_cnt[k] /= 10.0;
		}
		k_avg_sum /= 10.0;

		for (t = 0; t < 2; t++) {
			t_avg_cnt[t] /= 10.0;
		}
		t_avg_sum /= 10.0;

		for (v = 0; v < V; v++) {
			B_v_avg_cnt[v] /= 10.0;
		}
		B_v_avg_sum /= 10.0;

		for (k = 0; k < K; k++) {
			for (v = 0; v < V; v++) {
				k_v_avg_cnt[k][v] /= 10.0;
			}
			k_v_avg_sum[k] /= 10.0;
		}
				
	}

	private void updateProbabilities() {
		
		for (int t = 0; t < 3; t++) { // pi_{t}
			 //t_prob[t] = (t_cnt[t] + gamma) / (t_sum + gamma_sum);
			t_prob[t] = (t_avg_cnt[t] + gamma) / (t_avg_sum + gamma_sum);
		}

		for (int v = 0; v < V; v++) { // phi_{v}^{B}
			// B_v_prob[v] = (B_v_cnt[v] + beta) / (B_v_sum + beta_sum);
			B_v_prob[v] = (B_v_avg_cnt[v] + beta) / (B_v_avg_sum + beta_sum);
		}


		for (int k = 0; k < K; k++) { //phi_{v}^{K}
			for (int v = 0; v < V; v++) { //phi a
				//a_v_prob[a][v] = (a_v_cnt[a][v] + beta) / (a_v_sum[a] + beta_sum);
				k_v_prob[k][v] = (k_v_avg_cnt[k][v] + beta)/ (k_v_avg_sum[k] + beta_sum);
			}
		}
		
	}

	public void predict_labels(InstanceList instances) {
		int U, S;
		int u, s;

		U = instances.size();

		for (u = 0; u < U; u++) {
			Instance user = (Instance) instances.get(u);
			InstanceList sents = (InstanceList) user.getData();
			S = sents.size();
			for (s = 0; s < S; s++) {
				FeatureSequence sent = (FeatureSequence) sents.get(s).getData();
				assign_y_z(u, s, sent);
			}
		}
		

	}

	private void assign_y_z(int u, int s, FeatureSequence sent) {
		int N = sent.size();

		int n, k, t, v;

		double max_prob, prob = 0.0;
		int best_y;
		double max_log_sent_prob = 0.0, log_sent_prob;
		int best_z = 0;

		int[][] tmp_y = new int[K][];
		for (k = 0; k < K; k++) {
			tmp_y[k] = new int[N];
		}

		for (k = 0; k < K; k++) {

			log_sent_prob = 0.0;

			for (n = 0; n < N; n++) {
				v = sent.getIndexAtPosition(n);
				max_prob = 0.0;
				best_y = -1;

				for (t = 0; t < 2; t++) {

					if (t == 0) {
						prob = t_prob[t] * B_v_prob[v];
					} else if (t == 1) {
						prob = t_prob[t] * k_v_prob[k][v];
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
				tmp_y[k][n] = best_y;
			} // for(n = 0; n < N; n++)

			if (k > 0) {
				if (log_sent_prob > max_log_sent_prob) {
					max_log_sent_prob = log_sent_prob;
					best_z = k;
				}
			} else {
				max_log_sent_prob = log_sent_prob;
				best_z = k;
			}

		} // for(a = 0; a < A; a++)

		z[u][s] = best_z;

		for (n = 0; n < N; n++) {
			y[u][s][n] = tmp_y[best_z][n];
		}

	}

	public void output_labels(PrintWriter out, InstanceList instances) {

		int U, S, N;
		int u, s, n, k, t;

		U = instances.size();
		Alphabet dict = instances.getDataAlphabet();

		for (u = 0; u < U; u++) {
			Instance user = (Instance) instances.get(u);
			InstanceList sents = (InstanceList) user.getData();
			S = sents.size();
            		
			for (s = 0; s < S; s++) {
				FeatureSequence sent = (FeatureSequence) sents.get(s).getData();
				k = z[u][s] + 1;

				out.print(k +": ");
				N = sent.size();
				for (n = 0; n < N; n++) {
					int v = sent.getIndexAtPosition(n);
                    out.print(dict.lookupObject(v)+"/");

					t = y[u][s][n];
					if (t == 0) {
						out.print("B/0");
					} else if (t == 1) {
						out.print("T/");
						double prob = k_v_prob[k - 1][v];
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
 		for (int k = 0; k < K; k++) {
 			System.out.println("Topic " + k + "th:");
 			HashMap<Integer, Double> map = new HashMap<Integer, Double>();
 			for (int v = 0; v < V; v++) {
 				double pro = k_v_prob[k][v];
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
     
 	public void writeModel(String outputDir, String fileName){
		File dir = new File(outputDir);
		FileOutputStream fos;
		try {
			fos = new FileOutputStream(new File(dir, fileName + ".ser"));
			ObjectOutputStream out = new ObjectOutputStream(fos);
			out.writeObject(instances);
			out.writeObject(k_v_prob);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
	public void readModel(String outputDir, String fileName){
		File dir = new File(outputDir);
		FileInputStream fis;
		ObjectInputStream in = null;
		try {
			fis = new FileInputStream(new File(dir, fileName + ".ser"));
			in = new ObjectInputStream(fis);
			instances = (InstanceList) in.readObject();
			k_v_prob = (double[][]) in.readObject();

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
