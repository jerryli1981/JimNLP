package edu.pengli.nlp.platform.algorithms.lda;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Random;
import java.util.Set;

import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.FeatureSequence;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.RankMap;

public class CCLDAModel {

	private InstanceList instances;
	private Alphabet totalDict;

	private int K; // number of topics
	private int V; // vocabulary size
	private int D; // number of documents
	private int C; // number of collections
	private double alpha;
	private double beta;
	private double gamma0;
	private double gamma1;
	private int numIters; // number of Gibbs sampling iteration

	private int[][] z; // topic assignment for word of document z_{d,n}
	private int[][] l; // indicator assignment for word of document l_{d,n}
	
	private int[][] totalWordIndexMap;

	private int[] doc2Collection;


	// & l=0
	private int[][] v_k_cnt; // the number of times word v has been assigned to
								// topic k : n_{wi}^{zk}
	private int[] k_sum; // the total number of words assigned to topic k
							// :n_{.}^{zk}

	// & l=1
	private int[][][] v_c_k_cnt; // the number of times word v from collection c
									// has been assigned to topic k :
									// n_{wi}^{zk, c}
	private int[][] c_k_sum; // the total number of words from collection c
								// assigned to topic k :n_{.}^{zk, c}
	
	//////////////////////////////////////////////////////////////////
	
	// below variables for sample z_{d,n}
	private int[][] d_k_cnt; // the number of words from document i assigned to
								// topic k : n_{zk}^{di}
	private int[] d_sum; // the total number of words from document i :
							// n_{.}^{di}

	// below variables for sample l_{c,d,n}
	private int[][][] l_c_k_cnt; // In level l, the number of words from collection c
									// assigned to topic k
									// : n_{l}^{zk, c}
	
	private int[][] d_k_avg_cnt;
	private int[] d_avg_sum;
	private int[][] v_k_avg_cnt;
	private int[] k_avg_sum;
	private int[][][] v_c_k_avg_cnt;
	private int[][] c_k_avg_sum;
	private int[][][] l_c_k_avg_cnt;
	
	private double[][] k_v_prob; // phi_{v}^{K}
	private double[][][] k_v_c_prob; // phi_{v}^{K}
	private double[][] d_k_prob; // phi_{k}^{D}
	private double[][][] t_c_k_prob; // pi_{t}
	

	private double beta_sum;
	private double alpha_sum;

	public CCLDAModel(int numTopics, double alpha, double beta, double gamma0, double gamma1,
			int numIters) {
		K = numTopics;
		this.numIters = numIters;
		this.alpha = alpha;
		this.beta = beta;
		this.gamma0 = gamma0;
		this.gamma1 = gamma1;
	}

	public void initEstimate(ArrayList<InstanceList> collections) {
		C = collections.size();
		totalDict = new Alphabet();
		D = 0;
		instances = new InstanceList(null);

		HashMap<Instance, Integer> docCollectionMap = new HashMap<Instance, Integer>();
		int cIDx = 0;

		for (InstanceList collection : collections) {

			instances.addAll(collection);
			for (Instance doc : collection) {
				docCollectionMap.put(doc, cIDx);
			}
			cIDx++;
			Alphabet dict = collection.getDataAlphabet();
			D += collection.size();
			Object[] entries = dict.toArray();
			for (int i = 0; i < entries.length; i++) {
				Object entry = entries[i];
				totalDict.lookupIndex(entry);
			}
		}
		doc2Collection = new int[D];
		int dIDx = 0;
		cIDx = 0;
		for (InstanceList collection : collections) {
			for (Instance doc : collection) {
				doc2Collection[(dIDx++)] = cIDx;
			}
			cIDx++;
		}

		V = totalDict.size();
		beta_sum = V * beta;
		alpha_sum = K * alpha;
		
		

		d_k_cnt = new int[D][K];
		d_k_avg_cnt = new int[D][K];
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < K; j++) {
				d_k_cnt[i][j] = 0;
				d_k_avg_cnt[i][j] = 0; 
			}
		}

		d_sum = new int[D];
		d_avg_sum = new int[D];
		for (int i = 0; i < D; i++) {
			d_sum[i] = 0;
			d_avg_sum[i] = 0;
		}

		v_k_cnt = new int[V][K];
		v_k_avg_cnt = new int[V][K];
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < K; j++) {
				v_k_cnt[i][j] = 0;
				v_k_avg_cnt[i][j] = 0;
			}
		}

		k_sum = new int[K];
		k_avg_sum = new int[K];
		for (int i = 0; i < K; i++) {
			k_sum[i] = 0;
			k_avg_sum[i] = 0;
		}

		v_c_k_cnt = new int[V][C][K];
		v_c_k_avg_cnt = new int[V][C][K];
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < C; j++) {
				for (int k = 0; k < K; k++) {
					v_c_k_cnt[i][j][k] = 0;
					v_c_k_avg_cnt[i][j][k] = 0;
				}
			}
		}

		c_k_sum = new int[C][K];
		c_k_avg_sum = new int[C][K];
		for (int i = 0; i < C; i++) {
			for (int k = 0; k < K; k++) {
				c_k_sum[i][k] = 0;
				c_k_avg_sum[i][k] = 0;
			}
		}

		l_c_k_cnt = new int[2][C][K];
		l_c_k_avg_cnt = new int[2][C][K];
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < C; j++) {
				for (int k = 0; k < K; k++) {
					l_c_k_cnt[i][j][k] = 0;
					l_c_k_avg_cnt[i][j][k] = 0;
				}
			}
		}

			
		k_v_prob = new double[K][V];
		for(int i=0; i<K; i++){
			for(int j=0; j<V; j++){
				k_v_prob[i][j] = 0;
			}
		}
		
		k_v_c_prob = new double[K][V][C];
		for(int i=0; i<K; i++){
			for(int j=0; j<V; j++){
				for(int m=0; m<C; m++){
					k_v_c_prob[i][j][m] = 0;
				}
			}
		}
		
		d_k_prob = new double[D][K];
		for(int i=0; i<D; i++){
			for(int j=0; j<K; j++){
				d_k_prob[i][j] = 0;
			}
		}
		
		t_c_k_prob = new double[2][C][K];
		for(int i=0; i<2; i++){
			for(int j=0; j<C; j++){
				for(int k=0; k<K; k++){
					t_c_k_prob[i][j][k] = 0;
				}
				
			}
			
		}
		

		System.out.println("randomly initializing z and l...");
		Random r = new Random();

		
		z = new int[D][];
		l = new int[D][];
		totalWordIndexMap = new int[D][];
		for (int d = 0; d < D; d++) {
			Instance doc = (Instance) instances.get(d);
			FeatureSequence fs = (FeatureSequence) doc.getData();
			z[d] = new int[fs.size()];
			l[d] = new int[fs.size()];
			totalWordIndexMap[d] = new int[fs.size()];
			for (int n = 0; n < fs.size(); n++) {
				Object entry = fs.get(n);
				int v = totalDict.lookupIndex(entry);
				totalWordIndexMap[d][n] = v;
				
			}
		}

		for (int d = 0; d < D; d++) {
			Instance doc = (Instance) instances.get(d);
			int cIdx = docCollectionMap.get(doc);
			FeatureSequence fs = (FeatureSequence) doc.getData();
			for (int n = 0; n < fs.size(); n++) {
		//  	int k = (int) Math.floor(Math.random() * K);
		    	int k = r.nextInt(K);	
				z[d][n] = k;
			//	int x = (int) Math.floor(Math.random() * 2);
				int x = r.nextInt(2);	
				l[d][n] = x;

				d_k_cnt[d][k]++;
				d_sum[d]++;

				l_c_k_cnt[x][cIdx][k]++;
				
                int v = totalWordIndexMap[d][n]; // here is not 
//				int v = fs.getIndexAtPosition(n); 
				if (x == 0) {
					v_k_cnt[v][k]++;
					k_sum[k]++;

				} else if (x == 1) {
					v_c_k_cnt[v][cIdx][k]++;
					c_k_sum[cIdx][k]++;
				}
			}
		}
	}

	public void estimate() {

		for (int i = 0; i < numIters; i++) {
			System.out.println("Iteration " + (i + 1) + " ...");
			sweep(instances);
		}

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
			FeatureSequence fs = (FeatureSequence) doc.getData();
			for (int n = 0; n < fs.size(); n++) {
				sampling_z_l(d, n, fs);
			}

		}
	}

	private void sampling_z_l(int d, int n, FeatureSequence fs) {

		int topic = z[d][n];
		int route = l[d][n];
		int v = totalWordIndexMap[d][n];
//		int v = fs.getIndexAtPosition(n); 
		int c = doc2Collection[d];

		// decrease the count
		d_k_cnt[d][topic]--;
		d_sum[d]--;

		l_c_k_cnt[route][c][topic]--;

		if (route == 0) {
			v_k_cnt[v][topic]--;
			k_sum[topic]--;

		} else if (route == 1) {
			v_c_k_cnt[v][c][topic]--;
			c_k_sum[c][topic]--;
		}

		// sample new value for route
		double t_p, v_p = 0;
		double[] q = new double[2];
		for (int t = 0; t < 2; t++) {
			t_p = (l_c_k_cnt[route][c][topic] + gamma0)
					/ (c_k_sum[c][topic] + gamma0+gamma1);

			if (t == 0) {
				v_p = (v_k_cnt[v][topic] + beta) / (k_sum[topic] + beta_sum);

			} else if (t == 1) {
				v_p = (v_c_k_cnt[v][c][topic] + beta)
						/ (c_k_sum[c][topic] + beta_sum);
			}
			q[t] = t_p * v_p;

		}

		for (int t = 1; t < 2; t++) {
			q[t] += q[t - 1];
		}

		double X = Math.random() * q[1];
		for (int t = 0; t < 2; t++) {
			if (q[t] >= X) {
				route = t;
				break;
			}
		}

		// sample new value for topic
		double k_p;
		v_p = 0;
		double[] p = new double[K];

		if (route == 0) {
			for (int k = 0; k < K; k++) {
				k_p = (d_k_cnt[d][k] + alpha) / (d_sum[d] + alpha_sum);
				v_p = (v_k_cnt[v][k] + beta) / (k_sum[k] + beta_sum);
				p[k] = k_p * v_p;
			}

		} else if (route == 1) {
			for (int k = 0; k < K; k++) {
				k_p = (d_k_cnt[d][k] + alpha) / (d_sum[d] + alpha_sum);
				v_p = (v_c_k_cnt[v][c][k] + beta) / (c_k_sum[c][k] + beta_sum);
				p[k] = k_p * v_p;
			}
		}

		for (int k = 1; k < K; k++) {
			p[k] += p[K - 1];
		}

		X = Math.random() * p[K - 1];

		for (int k = 0; k < K; k++) {
			if (p[k] > X) {
				topic = k;
				break;
			}
		}

		// increment counts

		d_k_cnt[d][topic]++;
		d_sum[d]++;

		l_c_k_cnt[route][c][topic]++;


		if (route == 0) {
			v_k_cnt[v][topic]++;
			k_sum[topic]++;

		} else if (route == 1) {
			v_c_k_cnt[v][c][topic]++;
			c_k_sum[c][topic]++;
		}

		// set new assignments

		z[d][n] = topic;
		l[d][n] = route;

	}

	private void updateAvgCnt() {
		
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < K; j++) {
				d_k_avg_cnt[i][j] += d_k_cnt[i][j]; 
			}
			d_avg_sum[i] += d_sum[i];
		}
		
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < K; j++) {
				v_k_avg_cnt[i][j] += v_k_cnt[i][j];
			}
		}
		
		for (int i = 0; i < K; i++) {
			k_avg_sum[i] += k_sum[i] ;
		}
		
		
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < C; j++) {
				for (int k = 0; k < K; k++) {
					v_c_k_avg_cnt[i][j][k] += v_c_k_cnt[i][j][k] ;
				}
			}
		}

		for (int i = 0; i < C; i++) {
			for (int k = 0; k < K; k++) {
				c_k_avg_sum[i][k] += c_k_sum[i][k];
			}
		}
		
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < C; j++) {
				for (int k = 0; k < K; k++) {
					l_c_k_avg_cnt[i][j][k] += l_c_k_cnt[i][j][k];
				}
			}
		}			
	}

	private void averageCnt() {
		
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < K; j++) {
				d_k_avg_cnt[i][j] /= 10.0; 
			}
			d_avg_sum[i] /= 10.0; 
		}
		
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < K; j++) {
				v_k_avg_cnt[i][j] /= 10.0; 
			}
		}
		
		for (int i = 0; i < K; i++) {
			k_avg_sum[i] /= 10.0; 
		}
		
		
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < C; j++) {
				for (int k = 0; k < K; k++) {
					v_c_k_avg_cnt[i][j][k] /= 10.0; 
				}
			}
		}

		for (int i = 0; i < C; i++) {
			for (int k = 0; k < K; k++) {
				c_k_avg_sum[i][k] /= 10.0; 
			}
		}
		
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < C; j++) {
				for (int k = 0; k < K; k++) {
					l_c_k_avg_cnt[i][j][k] /= 10.0; 
				}
			}
		}
	
	}

	private void updateProbabilities() {
		
		for (int k = 0; k < K; k++) { //phi_{v}^{K}
			for (int v = 0; v < V; v++) { 
				k_v_prob[k][v] = (v_k_avg_cnt[v][k] + beta)/ (k_avg_sum[k] + beta_sum);
			}
		}
		
		for (int k = 0; k < K; k++) { //phi_{v}^{K}
			for (int v = 0; v < V; v++) { 
				for(int c=0; c < C; c++){
					k_v_c_prob[k][v][c] = (v_c_k_avg_cnt[v][c][k] + beta)/ (k_avg_sum[k] + beta_sum);
				}
			}
		}
		
		for(int d=0; d<D; d++){ // phi_{U}^{k}
			for(int k=0; k<K; k++){
				d_k_prob[d][k] = (d_k_avg_cnt[d][k] + alpha) / (d_avg_sum[d] + alpha_sum);
			}
		}
		
		for (int t = 0; t < 2; t++) { // pi_{t}
			for(int c=0; c<C; c++){
				for(int k=0; k<K; k++){
					t_c_k_prob[t][c][k] = (l_c_k_avg_cnt[t][c][k] + gamma0) / (c_k_avg_sum[t][c] + gamma0+gamma1);
				}

			}
		}	
	}

	public void output_model() {
    	int topK=10;

 		if (topK > V) {
 			topK = V;
 		}
 		
 		for (int k = 0; k < K; k++) {
 			System.out.println("Common Topic " + k + "th:");
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
 				Object obj = totalDict.lookupObject(idx);
 				double pro = map.get(idx);
 				System.out.println("\t" + obj.toString() + " " + pro);
 				i++;
 			}
 		}
 		
 		for(int c=0; c < C; c++){
 	 		for (int k = 0; k < K; k++) {
 	 			System.out.println("Collection " +c+" "+"Specific Topic " + k + "th:");
 	 			HashMap<Integer, Double> map = new HashMap<Integer, Double>();
 	 			for (int v = 0; v < V; v++) {
 	 				double pro = k_v_c_prob[k][v][c];
	 				map.put(v, pro); 
 	 			}
 	 			LinkedHashMap rankedMap = RankMap.sortHashMapByValues(map, false);
 	 			Set<Integer> keys = rankedMap.keySet();
 	 			Iterator iter = keys.iterator();
 	 			int i = 0;
 	 			while (iter.hasNext() && i < topK) {
 	 				int idx = (Integer) iter.next();
 	 				Object obj = totalDict.lookupObject(idx);
 	 				double pro = map.get(idx);
 	 				System.out.println("\t" + obj.toString() + " " + pro);
 	 				i++;
 	 			}
 	 		}
 		}
	}

	public void writeModel(String outputDir, String fileName) {

	}

	public void readModel(String outputDir, String fileName) {

	}
}
