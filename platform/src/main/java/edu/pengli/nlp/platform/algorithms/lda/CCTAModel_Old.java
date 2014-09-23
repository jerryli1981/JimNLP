package edu.pengli.nlp.platform.algorithms.lda;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;

import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.FeatureSequence;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;

public class CCTAModel_Old {

	private InstanceList instances;
	private Alphabet totalDict;

	private int numIters; // number of Gibbs sampling iteration

	private int K; // number of topics
	private int V; // vocabulary size
	private int D; // number of documents
	private int A; // number of aspects
	private int C; // number of collections

	private double alpha;
	private double alpha_sum;

	private double beta;
	private double beta_sum;

	private double gammaX;
	private double gammaX_sum;
	private double gammaL;
	private double gammaL_sum;
	
	private double delta;
	private double delta_sum;

	private int[][] z; // topic assignment for sentence of document z_{d,s}
	private int[][] y; // aspect assignment for sentence of document y_{d,s}
	private int[][][] l; // level assignment for word of sentence in document
							// l_{d,s,n}
	private int[][][] x; // route assignment for word of sentence in document
							// x_{d,s,n}

	private double[][][][] d_s_k_c_prob;

	private double[][] k_v_prob;
	private double[][][] k_v_c_prob;
	private double[][] a_v_prob;
	private double[][][] a_v_c_prob;

	private double[][][] k_a_v_prob;
	private double[][][][] k_a_v_c_prob;

	private double[] B_v_prob;
	private double[][] L_X_prob;

	private int[] doc2Collection;
	private int[][][] totalWordIndexMap;

	// //////////////////////////////////////////////////////////////////////////////////
	// below variables for compute v_p
	// l=0 & x=0: the word come from the background model
	private int[] v_B_cnt; // the number of times word v has been assigned to
							// Background topic
	private int B_sum; // the total number of words assigned to Background topic
	private double[] v_B_avg_cnt;
	private double B_avg_sum;

	// l=1 & x=1 : the sentence come from the general topic model
	private int[][] v_k_cnt; // the number of times word v has been assigned to
								// topic k
	private double[][] v_k_avg_cnt;

	private int[] v_k_sum; // the total number of words assigned to topic k
	private double[] v_k_avg_sum;

	private int[] k_cnt; // the number of sentences assigned to topic k
	private double[] k_avg_cnt;

	private int k_sum; // the total number of sentences
	private int k_avg_sum;

	// l=1 & x=2 : the word come from the general aspect model
	private int[][] v_a_cnt; // the number of times word v has been
								// assigned to aspect a
	private double[][] v_a_avg_cnt;

	private int[] v_a_sum; // the total number of words assigned to aspect a
	private double[] v_a_avg_sum;

	private int[] a_cnt; // the number of sentences assigned to aspect a
	private double[] a_avg_cnt;

	private int a_sum; // the total number of sentences
	private int a_avg_sum;

	// l=1 & x=3 : the word come from the general [topic&aspect] model
	private int[][][] v_k_a_cnt; // the number of times word v
									// belongs to aspect
	private double[][][] v_k_a_avg_cnt;
	// a has been assigned to topic k

	private int[][] v_k_a_sum; // the total number of words belongs
								// to aspect a
	private double[][] v_k_a_avg_sum;
	// assigned to topic k

	// l=2 & x=1 : the word come from the specific topic model
	private int[][][] v_c_k_cnt;
	private double[][][] v_c_k_avg_cnt;
	private int[][] v_c_k_sum;
	private double[][] v_c_k_avg_sum;

	// l=2 & x=2 : the word come from the specific aspect model
	private int[][][] v_c_a_cnt;
	private double[][][] v_c_a_avg_cnt;
	private int[][] v_c_a_sum;
	private double[][] v_c_a_avg_sum;

	// l=2 & x=3 : the word come from the specific [topic&aspect] model
	private int[][][][] v_c_k_a_cnt;
	private double[][][][] v_c_k_a_avg_cnt;
	private int[][][] v_c_k_a_sum;
	private double[][][] v_c_k_a_avg_sum;

	// /////////////////////////////////////////////////////////////////////////////////////////////////////////
	// below variables for compute k_p
	private int[][] d_k_cnt; // the number of words from document i assigned to
								// general topic k
	private double[][] d_k_avg_cnt;
	private int[][][] d_c_k_cnt; // the number of words from document i assigned
									// to specifc topic k
	private double[][][] d_c_k_avg_cnt;
	private int[] d_sum; // the total number of words from document i
	private double[] d_avg_sum;
	// below variables for compute a_p
	private int[][] d_a_cnt; // the number of words from document i has been
								// assigned to general aspect a
	private double[][] d_a_avg_cnt;
	private int[][][] d_c_a_cnt; // the number of words from document i has been
									// assigned to specific aspect a
	private double[][][] d_c_a_avg_cnt;

	// below variables for compute l_p;
	private int[][] d_l_cnt; // the number of words from document i has been
								// assigned to level l
	private double[][] d_l_avg_cnt;

	private int[][][] d_c_l_cnt;
	private double[][][] d_c_l_avg_cnt;

	// below variables for compute x_p
	private int[][][] l_x_a_cnt;
	private double[][][] l_x_a_avg_cnt;
	private int[][] l_a_sum;
	private double[][] l_a_avg_sum;
	private int[][][][] l_x_c_a_cnt;
	private double[][][][] l_x_c_a_avg_cnt;
	private int[][][] l_c_a_sum;
	private double[][][] l_c_a_avg_sum;

	private int[][][] l_x_k_cnt;
	private double[][][] l_x_k_avg_cnt;
	private int[][] l_k_sum;
	private double[][] l_k_avg_sum;
	private int[][][][] l_x_c_k_cnt;
	private double[][][][] l_x_c_k_avg_cnt;
	private int[][][] l_c_k_sum;
	private double[][][] l_c_k_avg_sum;

	private int[][] l_x_B_cnt;
	private double[][] l_x_B_avg_cnt;
	private int[] l_B_sum;
	private double[] l_B_avg_sum;

	public CCTAModel_Old(int numTopics, int numAspects, double alpha, double beta,
			double gammaX, double gammaL, double delta, int numIters) {
		K = numTopics;
		A = numAspects;
		this.numIters = numIters;
		this.alpha = alpha;
		this.beta = beta;
		this.delta = delta;
		this.gammaX = gammaX;
		this.gammaL = gammaL;
	}

	public CCTAModel copyObject() {
		return new CCTAModel(K, A, alpha, beta, gammaX, gammaL, delta, numIters);
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
			// System.out.println(dict.size());
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
		delta_sum = A * delta;
		gammaX_sum = 4 * gammaX; // 0,1,2,3
		gammaL_sum = 3 * gammaL; // 0,1,2

		// ///////////////////////////
		v_B_cnt = new int[V];
		v_B_avg_cnt = new double[V];
		B_sum = 0;
		for (int i = 0; i < V; i++) {
			v_B_cnt[i] = 0;
		}

		v_k_cnt = new int[V][K];
		v_k_avg_cnt = new double[V][K];
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < K; j++) {
				v_k_cnt[i][j] = 0;
				v_k_avg_cnt[i][j] = 0;
			}
		}
		v_k_sum = new int[K];
		v_k_avg_sum = new double[K];
		k_cnt = new int[K];
		k_avg_cnt = new double[K];
		for (int k = 0; k < K; k++) {
			v_k_sum[k] = 0;
			v_k_avg_sum[k] = 0;
			k_cnt[k] = 0;
			k_avg_cnt[k] = 0;
		}
		k_sum = 0;

		v_a_cnt = new int[V][A];
		v_a_avg_cnt = new double[V][A];
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < A; j++) {
				v_a_cnt[i][j] = 0;
				v_a_avg_cnt[i][j] = 0;
			}
		}

		v_a_sum = new int[A];
		v_a_avg_sum = new double[A];
		a_cnt = new int[A];
		a_avg_cnt = new double[A];
		for (int a = 0; a < A; a++) {
			v_a_sum[a] = 0;
			v_a_avg_sum[a] = 0;
			a_cnt[a] = 0;
			a_avg_cnt[a] = 0;
		}
		a_sum = 0;

		v_k_a_cnt = new int[V][K][A];
		v_k_a_avg_cnt = new double[V][K][A];
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < K; j++) {
				for (int m = 0; m < A; m++) {
					v_k_a_cnt[i][j][m] = 0;
					v_k_a_avg_cnt[i][j][m] = 0;
				}
			}
		}

		v_k_a_sum = new int[K][A];
		v_k_a_avg_sum = new double[K][A];
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < A; j++) {
				v_k_a_sum[i][j] = 0;
				v_k_a_avg_sum[i][j] = 0;
			}
		}

		v_c_k_cnt = new int[V][C][K];
		v_c_k_avg_cnt = new double[V][C][K];
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < K; m++) {
					v_c_k_cnt[i][j][m] = 0;
					v_c_k_avg_cnt[i][j][m] = 0;
				}
			}
		}

		v_c_k_sum = new int[C][K];
		v_c_k_avg_sum = new double[C][K];
		for (int j = 0; j < C; j++) {
			for (int m = 0; m < K; m++) {
				v_c_k_sum[j][m] = 0;
				v_c_k_avg_sum[j][m] = 0;
			}
		}

		v_c_a_cnt = new int[V][C][A];
		v_c_a_avg_cnt = new double[V][C][A];
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < A; m++) {
					v_c_a_cnt[i][j][m] = 0;
					v_c_a_avg_cnt[i][j][m] = 0;
				}
			}
		}

		v_c_a_sum = new int[C][A];
		v_c_a_avg_sum = new double[C][A];
		for (int j = 0; j < C; j++) {
			for (int m = 0; m < A; m++) {
				v_c_a_sum[j][m] = 0;
				v_c_a_avg_sum[j][m] = 0;
			}
		}

		v_c_k_a_cnt = new int[V][C][K][A];
		v_c_k_a_avg_cnt = new double[V][C][K][A];
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < K; m++) {
					for (int n = 0; n < A; n++) {
						v_c_k_a_cnt[i][j][m][n] = 0;
						v_c_k_a_avg_cnt[i][j][m][n] = 0;
					}
				}
			}
		}

		v_c_k_a_sum = new int[C][K][A];
		v_c_k_a_avg_sum = new double[C][K][A];
		for (int j = 0; j < C; j++) {
			for (int m = 0; m < K; m++) {
				for (int n = 0; n < A; n++) {
					v_c_k_a_sum[j][m][n] = 0;
					v_c_k_a_avg_sum[j][m][n] = 0;
				}
			}
		}

		d_k_cnt = new int[D][K];
		d_k_avg_cnt = new double[D][K];
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < K; j++) {
				d_k_cnt[i][j] = 0;
				d_k_avg_cnt[i][j] = 0;
			}
		}

		d_c_k_cnt = new int[D][C][K];
		d_c_k_avg_cnt = new double[D][C][K];
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < K; m++) {
					d_c_k_cnt[i][j][m] = 0;
					d_c_k_avg_cnt[i][j][m] = 0;
				}
			}
		}

		d_sum = new int[D];
		d_avg_sum = new double[D];
		for (int i = 0; i < D; i++) {
			d_sum[i] = 0;
			d_avg_sum[i] = 0;
		}

		d_a_cnt = new int[D][A];
		d_a_avg_cnt = new double[D][A];
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < A; j++) {
				d_a_cnt[i][j] = 0;
				d_a_avg_cnt[i][j] = 0;
			}
		}

		d_c_a_cnt = new int[D][C][A];
		d_c_a_avg_cnt = new double[D][C][A];
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < A; m++) {
					d_c_a_cnt[i][j][m] = 0;
					d_c_a_avg_cnt[i][j][m] = 0;
				}
			}
		}

		d_l_cnt = new int[D][3];
		d_l_avg_cnt = new double[D][3];
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < 3; j++) {
				d_l_cnt[i][j] = 0;
				d_l_avg_cnt[i][j] = 0;
			}
		}

		d_c_l_cnt = new int[D][C][3];
		d_c_l_avg_cnt = new double[D][C][3];
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < 3; m++) {
					d_c_l_cnt[i][j][m] = 0;
					d_c_l_avg_cnt[i][j][m] = 0;
				}
			}
		}

		l_x_a_cnt = new int[3][4][A];
		l_x_a_avg_cnt = new double[3][4][A];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				for (int m = 0; m < A; m++) {
					l_x_a_cnt[i][j][m] = 0;
					l_x_a_avg_cnt[i][j][m] = 0;
				}
			}
		}

		l_a_sum = new int[3][A];
		l_a_avg_sum = new double[3][A];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < A; j++) {
				l_a_sum[i][j] = 0;
				l_a_avg_sum[i][j] = 0;
			}
		}

		l_x_c_a_cnt = new int[3][4][C][A];
		l_x_c_a_avg_cnt = new double[3][4][C][A];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				for (int m = 0; m < C; m++) {
					for (int n = 0; n < A; n++) {
						l_x_c_a_cnt[i][j][m][n] = 0;
						l_x_c_a_avg_cnt[i][j][m][n] = 0;
					}
				}
			}
		}

		l_c_a_sum = new int[3][C][A];
		l_c_a_avg_sum = new double[3][C][A];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < A; m++) {
					l_c_a_sum[i][j][m] = 0;
					l_c_a_avg_sum[i][j][m] = 0;
				}
			}
		}

		l_x_k_cnt = new int[3][4][K];
		l_x_k_avg_cnt = new double[3][4][K];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				for (int m = 0; m < K; m++) {
					l_x_k_cnt[i][j][m] = 0;
					l_x_k_avg_cnt[i][j][m] = 0;
				}
			}
		}

		l_k_sum = new int[3][K];
		l_k_avg_sum = new double[3][K];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < K; j++) {
				l_k_sum[i][j] = 0;
				l_k_avg_sum[i][j] = 0;
			}
		}

		l_x_c_k_cnt = new int[3][4][C][K];
		l_x_c_k_avg_cnt = new double[3][4][C][K];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				for (int m = 0; m < C; m++) {
					for (int n = 0; n < K; n++) {
						l_x_c_k_cnt[i][j][m][n] = 0;
						l_x_c_k_avg_cnt[i][j][m][n] = 0;
					}
				}
			}
		}

		l_c_k_sum = new int[3][C][K];
		l_c_k_avg_sum = new double[3][C][K];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m > K; m++) {
					l_c_k_sum[i][j][m] = 0;
					l_c_k_avg_sum[i][j][m] = 0;
				}
			}
		}

		l_x_B_cnt = new int[3][4];
		l_x_B_avg_cnt = new double[3][4];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				l_x_B_cnt[i][j] = 0;
				l_x_B_avg_cnt[i][j] = 0;
			}
		}

		l_B_sum = new int[3];
		l_B_avg_sum = new double[3];
		for (int i = 0; i < 3; i++) {
			l_B_sum[i] = 0;
			l_B_avg_sum[i] = 0;
		}

		// initial probability

		B_v_prob = new double[V];
		L_X_prob = new double[3][4];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				L_X_prob[i][j] = 0.0;
			}
		}

		k_v_prob = new double[K][V];
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < V; j++) {
				k_v_prob[i][j] = 0;
			}
		}

		k_v_c_prob = new double[K][V][C];
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < V; j++) {
				for (int m = 0; m < C; m++) {
					k_v_c_prob[i][j][m] = 0;
				}
			}
		}

		a_v_prob = new double[A][V];
		for (int i = 0; i < A; i++) {
			for (int j = 0; j < V; j++) {
				a_v_prob[i][j] = 0;
			}
		}

		a_v_c_prob = new double[A][V][C];
		for (int i = 0; i < A; i++) {
			for (int j = 0; j < V; j++) {
				for (int m = 0; m < C; m++) {
					a_v_c_prob[i][j][m] = 0;
				}
			}
		}

		k_a_v_prob = new double[K][A][V];
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < A; j++) {
				for (int m = 0; m < V; m++) {
					k_a_v_prob[i][j][m] = 0;
				}
			}
		}

		k_a_v_c_prob = new double[K][A][V][C];
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < A; j++) {
				for (int m = 0; m < V; m++) {
					for (int n = 0; n < C; n++) {
						k_a_v_c_prob[i][j][m][n] = 0;
					}
				}
			}
		}

		// System.out.println("randomly initializing z y, l and x...");

		Random r = new Random();

		d_s_k_c_prob = new double[D][][][];
		for (int d = 0; d < D; d++) {
			Instance doc = (Instance) instances.get(d);
			InstanceList sents = (InstanceList) doc.getData();
			d_s_k_c_prob[d] = new double[sents.size()][][];
			for(int s=0; s<sents.size(); s++){
				d_s_k_c_prob[d][s] = new double[K][];
				for(int k=0; k<K; k++){
					d_s_k_c_prob[d][s][k] = new double[C];
				}
				
			}
		}

		z = new int[D][];
		l = new int[D][][];
		y = new int[D][];
		x = new int[D][][];
		totalWordIndexMap = new int[D][][];
		for (int d = 0; d < D; d++) {
			Instance doc = (Instance) instances.get(d);
			InstanceList sents = (InstanceList) doc.getData();
			z[d] = new int[sents.size()];
			y[d] = new int[sents.size()];
			l[d] = new int[sents.size()][];
			x[d] = new int[sents.size()][];

			totalWordIndexMap[d] = new int[sents.size()][];
			for (int s = 0; s < sents.size(); s++) {
				Instance sent = sents.get(s);
				FeatureSequence fs = (FeatureSequence) sent.getData();
				l[d][s] = new int[fs.size()];
				x[d][s] = new int[fs.size()];
				totalWordIndexMap[d][s] = new int[fs.size()];
				for (int n = 0; n < fs.size(); n++) {
					Object entry = fs.get(n);
					int v = totalDict.lookupIndex(entry);
					totalWordIndexMap[d][s][n] = v;
				}

			}
		}

		for (int d = 0; d < D; d++) {
			Instance doc = (Instance) instances.get(d);
			int c = docCollectionMap.get(doc);
			InstanceList sents = (InstanceList) doc.getData();
			for (int s = 0; s < sents.size(); s++) {
				int topic = (int) Math.floor(Math.random() * K);
				int aspect = (int) Math.floor(Math.random() * A);
				z[d][s] = topic;
				y[d][s] = aspect;

				k_sum++;
				a_sum++;

				k_cnt[topic]++;
				a_cnt[aspect]++;
				Instance sent = sents.get(s);
				FeatureSequence fs = (FeatureSequence) sent.getData();
				for (int n = 0; n < fs.size(); n++) {
					int level = r.nextInt(3);
					l[d][s][n] = level;
					int route = r.nextInt(4);
					x[d][s][n] = route;

					int v = totalWordIndexMap[d][s][n];
					if (level == 0 && route == 0) {
						v_B_cnt[v]++;
						B_sum++;
					} else if (level == 1 && route == 1) {
						v_k_cnt[v][topic]++;
						v_k_sum[topic]++;
					} else if (level == 1 && route == 2) {
						v_a_cnt[v][aspect]++;
						v_a_sum[aspect]++;

					} else if (level == 1 && route == 3) {
						v_k_a_cnt[v][topic][aspect]++;
						v_k_a_sum[topic][aspect]++;
					} else if (level == 2 && route == 1) {
						v_c_k_cnt[v][c][topic]++;
						v_c_k_sum[c][topic]++;
					} else if (level == 2 && route == 2) {
						v_c_a_cnt[v][c][aspect]++;
						v_c_a_sum[c][aspect]++;
					} else if (level == 2 && route == 3) {
						v_c_k_a_cnt[v][c][topic][aspect]++;
						v_c_k_a_sum[c][topic][aspect]++;
					}

					d_k_cnt[d][topic]++;
					d_c_k_cnt[d][c][topic]++;
					d_sum[d]++;
					d_a_cnt[d][aspect]++;
					d_c_a_cnt[d][c][aspect]++;
					d_l_cnt[d][level]++;
					d_c_l_cnt[d][c][level]++;

					l_x_a_cnt[level][route][aspect]++;
					l_a_sum[level][aspect]++;
					l_x_c_a_cnt[level][route][c][aspect]++;
					l_c_a_sum[level][c][aspect]++;

					l_x_k_cnt[level][route][topic]++;
					l_k_sum[level][topic]++;
					l_x_c_k_cnt[level][route][c][topic]++;
					l_c_k_sum[level][c][topic]++;

					l_x_B_cnt[level][route]++;
					l_B_sum[level]++;

				}

			}
		}

	}

	public void estimate() {
		for (int i = 0; i < numIters; i++) {
			// System.out.println("Iteration " + (i + 1) + " ...");
			sweep(instances);
		}

		// System.out.println("Taking samples...");
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
				sampling_z_y(d, s, sent);
				int N = sent.size();
				for (int n = 0; n < N; n++) {
					sampling_l_x(d, s, n);
				}

			}

		}
	}

	private void sampling_z_y(int d, int s, FeatureSequence sent) {
		int c = doc2Collection[d];
		int N = sent.size();

		int topic = z[d][s];
		int aspect = y[d][s];

		// Decrease the counts

		k_cnt[topic]--;
		k_sum--;
		a_cnt[aspect]--;
		a_sum--;

		for (int n = 0; n < N; n++) {

			int level = l[d][s][n];
			int route = x[d][s][n];

			int v = totalWordIndexMap[d][s][n];
			if (level == 0 && route == 0) {
				v_B_cnt[v]--;
				B_sum--;
			} else if (level == 1 && route == 1) {
				v_k_cnt[v][topic]--;
				v_k_sum[topic]--;
			} else if (level == 1 && route == 2) {
				v_a_cnt[v][aspect]--;
				v_a_sum[aspect]--;

			} else if (level == 1 && route == 3) {
				v_k_a_cnt[v][topic][aspect]--;
				v_k_a_sum[topic][aspect]--;
			} else if (level == 2 && route == 1) {
				v_c_k_cnt[v][c][topic]--;
				v_c_k_sum[c][topic]--;
			} else if (level == 2 && route == 2) {
				v_c_a_cnt[v][c][aspect]--;
				v_c_a_sum[c][aspect]--;
			} else if (level == 2 && route == 3) {
				v_c_k_a_cnt[v][c][topic][aspect]--;
				v_c_k_a_sum[c][topic][aspect]--;
			}

		}

		// *********************************************
		// draw a new topic and aspect based on the assignments of all other
		// hidden
		// variables

		topic = draw_z(d, s, sent);
		aspect = draw_y(d, s, sent);

		// *********************************************
		// Increase the counts

		k_cnt[topic]++;
		k_sum++;
		a_cnt[aspect]++;
		a_sum++;

		for (int n = 0; n < N; n++) {

			int level = l[d][s][n];
			int route = x[d][s][n];

			int v = totalWordIndexMap[d][s][n];

			if (level == 0 && route == 0) {
				v_B_cnt[v]++;
				B_sum++;
			} else if (level == 1 && route == 1) {
				v_k_cnt[v][topic]++;
				v_k_sum[topic]++;
			} else if (level == 1 && route == 2) {
				v_a_cnt[v][aspect]++;
				v_a_sum[aspect]++;
			} else if (level == 1 && route == 3) {
				v_k_a_cnt[v][topic][aspect]++;
				v_k_a_sum[topic][aspect]++;
			} else if (level == 2 && route == 1) {
				v_c_k_cnt[v][c][topic]++;
				v_c_k_sum[c][topic]++;
			} else if (level == 2 && route == 2) {
				v_c_a_cnt[v][c][aspect]++;
				v_c_a_sum[c][aspect]++;
			} else if (level == 2 && route == 3) {
				v_c_k_a_cnt[v][c][topic][aspect]++;
				v_c_k_a_sum[c][topic][aspect]++;
			}
		}

		z[d][s] = topic;
		y[d][s] = aspect;
	}

	private int draw_z(int d, int s, FeatureSequence sent) {
		int a = y[d][s];
		double[] p = new double[K];
		double[] log_v_p = new double[K];
        int c = doc2Collection[d];
		for (int k = 0; k < K; k++) {
			double k_p = (k_cnt[k] + alpha) / (k_sum + alpha_sum);
			HashMap<Integer, Integer> my_k_v_cnt = new HashMap<Integer, Integer>();
			HashMap<Integer, String> v_l_r_map = new HashMap<Integer, String>();
			for (int n = 0; n < sent.size(); n++) {
				int level = l[d][s][n];
				int route = x[d][s][n];
				if ((level == 1 && route == 1) || (level == 1 && route == 3) 
						|| (level == 2 && route == 1) || (level == 2 && route == 3)) {
					int v = totalWordIndexMap[d][s][n];
					v_l_r_map.put(v, level+" "+route);
					if (!my_k_v_cnt.containsKey(v)) {
						my_k_v_cnt.put(v, 1);
					} else {
						int cnt = my_k_v_cnt.get(v);
						my_k_v_cnt.put(v, (++cnt));
					}
				}
			}

			log_v_p[k] = 0.0;
			Set<Integer> keys = my_k_v_cnt.keySet();
			Iterator iter = keys.iterator();
			while (iter.hasNext()) {
				int v = (Integer) iter.next();
				String l_r = v_l_r_map.get(v);
				int l = Integer.parseInt(l_r.split(" ")[0]);
				int r = Integer.parseInt(l_r.split(" ")[1]);
				int cnt = my_k_v_cnt.get(v);
				if( l ==1 && r == 1){
					for (int i = 0; i < cnt; i++) {
						log_v_p[k] += Math.log(v_k_cnt[v][k] + beta + i);
					}
				}else if(l ==1 && r == 3){
					for (int i = 0; i < cnt; i++) {
						log_v_p[k] += Math.log(v_k_a_cnt[v][k][a] + beta + i);
					}
					
				}else if(l ==2 && r == 1){
					for (int i = 0; i < cnt; i++) {
						log_v_p[k] += Math.log(v_c_k_cnt[v][c][k] + beta + i);
					}
				}else if(l ==2 && r == 3){
					for (int i = 0; i < cnt; i++) {
						log_v_p[k] += Math.log(v_c_k_a_cnt[v][c][k][a] + beta + i);
					}
				}	
			}
			
			keys = my_k_v_cnt.keySet();
			iter = keys.iterator();
			while (iter.hasNext()) {
				int v = (Integer) iter.next();
				String l_r = v_l_r_map.get(v);
				int l = Integer.parseInt(l_r.split(" ")[0]);
				int r = Integer.parseInt(l_r.split(" ")[1]);
				int cnt = my_k_v_cnt.get(v);
				if( l ==1 && r == 1){
					for (int i = 0; i < cnt; i++) {
						log_v_p[k] -= Math.log(v_k_sum[k] + beta_sum + i);
					}
				}else if(l ==1 && r == 3){
					for (int i = 0; i < cnt; i++) {
						log_v_p[k] -= Math.log(v_k_a_sum[k][a] + beta_sum + i);
					}
					
				}else if(l ==2 && r == 1){
					for (int i = 0; i < cnt; i++) {
						log_v_p[k] -= Math.log(v_c_k_sum[c][k] + beta_sum + i);
					}
				}else if(l ==2 && r == 3){
					for (int i = 0; i < cnt; i++) {
						log_v_p[k] -= Math.log(v_c_k_a_sum[c][k][a] + beta_sum + i);
					}
				}	
			}
			
			p[k] = k_p;

		}

		double max_log_v_p = log_v_p[0];
		for (int k = 1; k < K; k++) {
			if (log_v_p[k] > max_log_v_p) {
				max_log_v_p = log_v_p[k];
			}
		}
		// smoothing
		for (int k = 0; k < K; k++) {
			log_v_p[k] -= max_log_v_p;
			double v_p = Math.exp(log_v_p[k]);
			p[k] *= v_p;
		}

		for (int k = 1; k < K; k++) {
			p[k] += p[k - 1];
		}

		if (p[K - 1] == 0) {
			System.out.println("Error: p[A - 1] is zero!");
			System.exit(0);
		}

		// scaled sample because of unnormalized p[]

		double u = Math.random() * p[K - 1];

		int topic = 0;
		for (int k = 0; k < K; k++) {
			if (p[k] > u) { // sample topic w.r.t distribution p
				topic = k;
				break;
			}
		}
		return topic;
	}

	private int draw_y(int d, int s, FeatureSequence sent) {
		int k = z[d][s];
		double[] p = new double[A];
		double[] log_v_p = new double[A];
		int c = doc2Collection[d];
		for (int a = 0; a < A; a++) {
			double a_p = (a_cnt[a] + alpha) / (a_sum + alpha_sum);
			int my_a_v_sum = 0;
			HashMap<Integer, Integer> my_a_v_cnt = new HashMap<Integer, Integer>();
			HashMap<Integer, String> v_l_r_map = new HashMap<Integer, String>();
			for (int n = 0; n < sent.size(); n++) {
				int level = l[d][s][n];
				int route = x[d][s][n];
				if ((level == 1 && route == 2) || (level == 1 && route == 3) 
						|| (level == 2 && route == 2) || (level == 2 && route == 3)) {
					int v = totalWordIndexMap[d][s][n];
					v_l_r_map.put(v, level+" "+route);
					if (!my_a_v_cnt.containsKey(v)) {
						my_a_v_cnt.put(v, 1);
					} else {
						int cnt = my_a_v_cnt.get(v);
						my_a_v_cnt.put(v, (++cnt));
					}
					my_a_v_sum++;
				}
			}

			log_v_p[a] = 0.0;
			Set<Integer> keys = my_a_v_cnt.keySet();
			Iterator iter = keys.iterator();
			while (iter.hasNext()) {
				int v = (Integer) iter.next();
				String l_r = v_l_r_map.get(v);
				int l = Integer.parseInt(l_r.split(" ")[0]);
				int r = Integer.parseInt(l_r.split(" ")[1]);
				int cnt = my_a_v_cnt.get(v);
				if( l ==1 && r == 2){
					for (int i = 0; i < cnt; i++) {
						log_v_p[a] += Math.log(v_a_cnt[v][a] + beta + i);
					}
				}else if(l ==1 && r == 3){
					for (int i = 0; i < cnt; i++) {
						log_v_p[a] += Math.log(v_k_a_cnt[v][k][a] + beta + i);
					}
				}else if(l ==2 && r == 2){
					for (int i = 0; i < cnt; i++) {
						log_v_p[a] += Math.log(v_c_a_cnt[v][c][a] + beta + i);
					}
				}else if(l ==2 && r == 3){
					for (int i = 0; i < cnt; i++) {
						log_v_p[a] += Math.log(v_c_k_a_cnt[v][c][k][a] + beta + i);
					}
				}
			}
			
			keys = my_a_v_cnt.keySet();
			iter = keys.iterator();
			while (iter.hasNext()) {
				int v = (Integer) iter.next();
				String l_r = v_l_r_map.get(v);
				int l = Integer.parseInt(l_r.split(" ")[0]);
				int r = Integer.parseInt(l_r.split(" ")[1]);
				int cnt = my_a_v_cnt.get(v);
				if( l ==1 && r == 2){
					for (int i = 0; i < cnt; i++) {
						log_v_p[a] -= Math.log(v_a_sum[a] + beta_sum + i);
					}
				}else if(l ==1 && r == 3){
					for (int i = 0; i < cnt; i++) {
						log_v_p[a] -= Math.log(v_k_a_sum[k][a] + beta_sum + i);
					}
				}else if(l ==2 && r == 2){
					for (int i = 0; i < cnt; i++) {
						log_v_p[a] -= Math.log(v_c_a_sum[c][a] + beta_sum + i);
					}
				}else if(l ==2 && r == 3){
					for (int i = 0; i < cnt; i++) {
						log_v_p[a] -= Math.log(v_c_k_a_sum[c][k][a] + beta_sum + i);
					}
				}
			}

			for (int i = 0; i < my_a_v_sum; i++) {
				log_v_p[a] -= Math.log(v_a_sum[a] + beta_sum + i);
			}

			p[a] = a_p;

		}

		double max_log_v_p = log_v_p[0];
		for (int a = 1; a < A; a++) {
			if (log_v_p[a] > max_log_v_p) {
				max_log_v_p = log_v_p[a];
			}
		}
		// smoothing
		for (int a = 0; a < A; a++) {
			log_v_p[a] -= max_log_v_p;
			double v_p = Math.exp(log_v_p[a]);
			p[a] *= v_p;
		}

		for (int a = 1; a < A; a++) {
			p[a] += p[a - 1];
		}

		if (p[A - 1] == 0) {
			System.out.println("Error: p[A - 1] is zero!");
			System.exit(0);
		}

		// scaled sample because of unnormalized p[]

		double u = Math.random() * p[A - 1];

		int aspect = 0;
		for (int a = 0; a < A; a++) {
			if (p[a] > u) { // sample topic w.r.t distribution p
				aspect = a;
				break;
			}
		}
		return aspect;
	}

	private void sampling_l_x(int d, int s, int n) {

		int level = l[d][s][n];
		int route = x[d][s][n];
		int topic = z[d][s];
		int aspect = y[d][s];

		int c = doc2Collection[d];
		int v = totalWordIndexMap[d][s][n];

		// decrease the counts

		if (level == 0 && route == 0) {
			v_B_cnt[v]--;
			B_sum--;
		} else if (level == 1 && route == 1) {
			v_k_cnt[v][topic]--;
			v_k_sum[topic]--;
		} else if (level == 1 && route == 2) {
			v_a_cnt[v][aspect]--;
			v_a_sum[aspect]--;
		} else if (level == 1 && route == 3) {
			v_k_a_cnt[v][topic][aspect]--;
			v_k_a_sum[topic][aspect]--;
		} else if (level == 2 && route == 1) {
			v_c_k_cnt[v][c][topic]--;
			v_c_k_sum[c][topic]--;
		} else if (level == 2 && route == 2) {
			v_c_a_cnt[v][c][aspect]--;
			v_c_a_sum[c][aspect]--;
		} else if (level == 2 && route == 3) {
			v_c_k_a_cnt[v][c][topic][aspect]--;
			v_c_k_a_sum[c][topic][aspect]--;
		}

		d_k_cnt[d][topic]--;
		d_c_k_cnt[d][c][topic]--;
		d_sum[d]--;
		d_a_cnt[d][aspect]--;
		d_c_a_cnt[d][c][aspect]--;
		d_l_cnt[d][level]--;
		d_c_l_cnt[d][c][level]--;

		l_x_a_cnt[level][route][aspect]--;
		l_a_sum[level][aspect]--;
		l_x_c_a_cnt[level][route][c][aspect]--;
		l_c_a_sum[level][c][aspect]--;

		l_x_k_cnt[level][route][topic]--;
		l_k_sum[level][topic]--;
		l_x_c_k_cnt[level][route][c][topic]--;
		l_c_k_sum[level][c][topic]--;

		l_x_B_cnt[level][route]--;
		l_B_sum[level]--;

		// sample new value for level and route
		double pTotal = 0.0;
		double[] p = new double[7];

		// l = 0, x = 0 background model
		p[0] = (d_l_cnt[d][0] + gammaL) / (d_sum[d] + gammaL_sum) * // l_p
				(l_x_B_cnt[0][0] + gammaX) / (l_B_sum[0] + gammaX_sum) * // x_p
				(v_B_cnt[v] + beta) / (B_sum + beta_sum); // v_p

		// l = 1, x = 1 general topic model
		p[1] = (d_l_cnt[d][1] + gammaL) / (d_sum[d] + gammaL_sum)
				* (l_x_k_cnt[1][1][topic] + gammaX)
				/ (l_k_sum[1][topic] + gammaX_sum) * (v_k_cnt[v][topic] + beta)
				/ (v_k_sum[topic] + beta_sum);

		// l = 1, x = 2 general aspect model
		p[2] = (d_l_cnt[d][1] + gammaL) / (d_sum[d] + gammaL_sum)
				* (l_x_a_cnt[1][2][aspect] + gammaX)
				/ (l_a_sum[1][aspect] + gammaX_sum)
				* (v_a_cnt[v][aspect] + beta) / (v_a_sum[aspect] + beta_sum);

		// l = 1, x = 3 general [topic&aspect] model
		p[3] = (d_l_cnt[d][1] + gammaL) / (d_sum[d] + gammaL_sum)
				* (l_x_k_cnt[1][3][topic] + gammaX)
				/ (l_k_sum[1][topic] + gammaX_sum)
				* (v_k_a_cnt[v][topic][aspect] + beta)
				/ (v_k_a_sum[topic][aspect] + beta_sum);

		// l = 2, x = 1 specific topic model
		p[4] = (d_c_l_cnt[d][c][2] + gammaL) / (d_sum[d] + gammaL_sum)
				* (l_x_c_k_cnt[2][1][c][topic] + gammaX)
				/ (l_c_k_sum[2][c][topic] + gammaX_sum)
				* (v_c_k_cnt[v][c][topic] + beta)
				/ (v_c_k_sum[c][topic] + beta_sum);

		// l = 2, x = 2 specific aspect model
		p[5] = (d_c_l_cnt[d][c][2] + gammaL) / (d_sum[d] + gammaL_sum)
				* (l_x_c_a_cnt[2][2][c][aspect] + gammaX)
				/ (l_c_k_sum[2][c][aspect] + gammaX_sum)
				* (v_c_a_cnt[v][c][aspect] + beta)
				/ (v_c_a_sum[c][aspect] + beta_sum);

		// l = 2, x = 3 specific [topic&aspect] model
		p[6] = (d_c_l_cnt[d][c][2] + gammaL) / (d_sum[d] + gammaL_sum)
				* (l_x_c_k_cnt[2][3][c][topic] + gammaX)
				/ (l_c_k_sum[2][c][topic] + gammaX_sum)
				* (v_c_k_a_cnt[v][c][topic][aspect] + beta)
				/ (v_c_k_a_sum[c][topic][aspect] + beta_sum);

		pTotal = p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6];
		double X = Math.random() * pTotal;
		double tmp = 0;
		for (int i = 0; i < 7; i++) {
			tmp += p[i];
			if (tmp > X) {
				if (i == 0) {
					level = 0;
					route = 0;
				} else if (i >= 1 & i < 4) {
					level = 1;
					route = i;
				} else {
					level = 2;
					route = i - 3;
				}
				break;
			}
		}

		// increase the counts

		if (level == 0 && route == 0) {
			v_B_cnt[v]++;
			B_sum++;
		} else if (level == 1 && route == 1) {
			v_k_cnt[v][topic]++;
			v_k_sum[topic]++;
		} else if (level == 1 && route == 2) {
			v_a_cnt[v][aspect]++;
			v_a_sum[aspect]++;
		} else if (level == 1 && route == 3) {
			v_k_a_cnt[v][topic][aspect]++;
			v_k_a_sum[topic][aspect]++;
		} else if (level == 2 && route == 1) {
			v_c_k_cnt[v][c][topic]++;
			v_c_k_sum[c][topic]++;
		} else if (level == 2 && route == 2) {
			v_c_a_cnt[v][c][aspect]++;
			v_c_a_sum[c][aspect]++;
		} else if (level == 2 && route == 3) {
			v_c_k_a_cnt[v][c][topic][aspect]++;
			v_c_k_a_sum[c][topic][aspect]++;
		}

		d_k_cnt[d][topic]++;
		d_c_k_cnt[d][c][topic]++;
		d_sum[d]++;
		d_a_cnt[d][aspect]++;
		d_c_a_cnt[d][c][aspect]++;
		d_l_cnt[d][level]++;
		d_c_l_cnt[d][c][level]++;

		l_x_a_cnt[level][route][aspect]++;
		l_a_sum[level][aspect]++;
		l_x_c_a_cnt[level][route][c][aspect]++;
		l_c_a_sum[level][c][aspect]++;

		l_x_k_cnt[level][route][topic]++;
		l_k_sum[level][topic]++;
		l_x_c_k_cnt[level][route][c][topic]++;
		l_c_k_sum[level][c][topic]++;

		l_x_B_cnt[level][route]++;
		l_B_sum[level]++;

		// set new assignments
		l[d][s][n] = level;
		x[d][s][n] = route;
	}

	private void updateAvgCnt() {

		for (int v = 0; v < V; v++) {
			v_B_avg_cnt[v] += v_B_cnt[v];
		}
		B_avg_sum += B_sum;

		for (int i = 0; i < V; i++) {
			for (int j = 0; j < K; j++) {
				v_k_avg_cnt[i][j] += v_k_cnt[i][j];
			}
		}

		for (int i = 0; i < K; i++) {
			v_k_avg_sum[i] += v_k_sum[i];
		}

		for (int k = 0; k < K; k++) {
			k_avg_cnt[k] += k_cnt[k];
		}
		k_avg_sum += k_sum;

		for (int i = 0; i < V; i++) {
			for (int j = 0; j < A; j++) {
				v_a_avg_cnt[i][j] += v_a_cnt[i][j];
			}
		}

		for (int i = 0; i < A; i++) {
			v_a_avg_sum[i] += v_a_sum[i];
		}

		for (int i = 0; i < A; i++) {
			a_avg_cnt[i] += a_cnt[i];
		}

		for (int i = 0; i < V; i++) {
			for (int j = 0; j < K; j++) {
				for (int m = 0; m < A; m++) {

					v_k_a_avg_cnt[i][j][m] += v_k_a_cnt[i][j][m];
				}
			}
		}

		for (int i = 0; i < K; i++) {
			for (int j = 0; j < A; j++) {
				v_k_a_avg_sum[i][j] += v_k_a_sum[i][j];
			}
		}

		for (int i = 0; i < V; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < K; m++) {
					v_c_k_avg_cnt[i][j][m] += v_c_k_cnt[i][j][m];
				}
			}
		}

		for (int j = 0; j < C; j++) {
			for (int m = 0; m < K; m++) {
				v_c_k_avg_sum[j][m] += v_c_k_sum[j][m];
			}
		}

		for (int i = 0; i < V; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < A; m++) {
					v_c_a_avg_cnt[i][j][m] += v_c_a_cnt[i][j][m];
				}
			}
		}

		for (int j = 0; j < C; j++) {
			for (int m = 0; m < A; m++) {
				v_c_a_avg_sum[j][m] += v_c_a_sum[j][m];
			}
		}

		for (int i = 0; i < V; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < K; m++) {
					for (int n = 0; n < A; n++) {

						v_c_k_a_avg_cnt[i][j][m][n] += v_c_k_a_cnt[i][j][m][n];
					}
				}
			}
		}

		for (int j = 0; j < C; j++) {
			for (int m = 0; m < K; m++) {
				for (int n = 0; n < A; n++) {
					v_c_k_a_avg_sum[j][m][n] += v_c_k_a_sum[j][m][n];
				}
			}
		}
		// ///////////////////////////////////////

		for (int i = 0; i < D; i++) {
			for (int j = 0; j < K; j++) {
				d_k_avg_cnt[i][j] += d_k_cnt[i][j];
			}
		}

		for (int i = 0; i < D; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < K; m++) {
					d_c_k_avg_cnt[i][j][m] += d_c_k_cnt[i][j][m];
				}
			}
		}

		for (int i = 0; i < D; i++) {

			d_avg_sum[i] += d_sum[i];
		}

		for (int i = 0; i < D; i++) {
			for (int j = 0; j < A; j++) {
				d_a_avg_cnt[i][j] += d_a_cnt[i][j];
			}
		}

		for (int i = 0; i < D; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < A; m++) {

					d_c_a_avg_cnt[i][j][m] += d_c_a_cnt[i][j][m];
				}
			}
		}

		for (int i = 0; i < D; i++) {
			for (int j = 0; j < 3; j++) {

				d_l_avg_cnt[i][j] += d_l_cnt[i][j];
			}
		}

		for (int i = 0; i < D; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < 3; m++) {

					d_c_l_avg_cnt[i][j][m] += d_c_l_cnt[i][j][m];
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				for (int m = 0; m < A; m++) {

					l_x_a_avg_cnt[i][j][m] += l_x_a_cnt[i][j][m];
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < A; j++) {

				l_a_avg_sum[i][j] += l_a_sum[i][j];
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				for (int m = 0; m < C; m++) {
					for (int n = 0; n < A; n++) {

						l_x_c_a_avg_cnt[i][j][m][n] += l_x_c_a_cnt[i][j][m][n];
					}
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < A; m++) {

					l_c_a_avg_sum[i][j][m] += l_c_a_sum[i][j][m];
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				for (int m = 0; m < K; m++) {

					l_x_k_avg_cnt[i][j][m] += l_x_k_cnt[i][j][m];
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < K; j++) {

				l_k_avg_sum[i][j] += l_k_sum[i][j];
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				for (int m = 0; m < C; m++) {
					for (int n = 0; n < K; n++) {

						l_x_c_k_avg_cnt[i][j][m][n] += l_x_c_k_cnt[i][j][m][n];
					}
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m > K; m++) {

					l_c_k_avg_sum[i][j][m] += l_c_k_sum[i][j][m];
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {

				l_x_B_avg_cnt[i][j] += l_x_B_cnt[i][j];
			}
		}

		for (int i = 0; i < 3; i++) {
			l_B_avg_sum[i] += l_B_sum[i];
		}

	}

	private void averageCnt() {

		for (int v = 0; v < V; v++) {
			v_B_avg_cnt[v] /= 10.0;
		}
		B_avg_sum /= 10.0;

		for (int i = 0; i < V; i++) {
			for (int j = 0; j < K; j++) {
				v_k_avg_cnt[i][j] /= 10.0;
			}

		}

		for (int i = 0; i < K; i++) {
			v_k_avg_sum[i] /= 10.0;
		}

		for (int k = 0; k < K; k++) {
			k_avg_cnt[k] /= 10.0;
		}
		k_avg_sum /= 10.0;

		for (int i = 0; i < V; i++) {
			for (int j = 0; j < A; j++) {
				v_a_avg_cnt[i][j] /= 10.0;
			}
		}

		for (int i = 0; i < A; i++) {
			a_avg_cnt[i] /= 10.0;
		}

		for (int i = 0; i < V; i++) {
			for (int j = 0; j < K; j++) {
				for (int m = 0; m < A; m++) {

					v_k_a_avg_cnt[i][j][m] /= 10.0;
				}
			}
		}

		for (int i = 0; i < K; i++) {
			for (int j = 0; j < A; j++) {
				v_k_a_avg_sum[i][j] /= 10.0;
			}
		}

		for (int i = 0; i < V; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < K; m++) {
					v_c_k_avg_cnt[i][j][m] /= 10.0;
				}
			}
		}

		for (int j = 0; j < C; j++) {
			for (int m = 0; m < K; m++) {
				v_c_k_avg_sum[j][m] /= 10.0;
			}
		}

		for (int i = 0; i < V; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < A; m++) {
					v_c_a_avg_cnt[i][j][m] /= 10.0;
				}
			}
		}

		for (int j = 0; j < C; j++) {
			for (int m = 0; m < A; m++) {
				v_c_a_avg_sum[j][m] /= 10.0;
			}
		}

		for (int i = 0; i < V; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < K; m++) {
					for (int n = 0; n < A; n++) {

						v_c_k_a_avg_cnt[i][j][m][n] /= 10.0;
					}
				}
			}
		}

		for (int j = 0; j < C; j++) {
			for (int m = 0; m < K; m++) {
				for (int n = 0; n < A; n++) {
					v_c_k_a_avg_sum[j][m][n] /= 10.0;
				}
			}
		}

		// ///////////
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < K; j++) {
				d_k_avg_cnt[i][j] /= 10.0;
			}
		}

		for (int i = 0; i < D; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < K; m++) {
					d_c_k_avg_cnt[i][j][m] /= 10.0;
				}
			}
		}

		for (int i = 0; i < D; i++) {

			d_avg_sum[i] /= 10.0;
		}

		for (int i = 0; i < D; i++) {
			for (int j = 0; j < A; j++) {
				d_a_avg_cnt[i][j] /= 10.0;
			}
		}

		for (int i = 0; i < D; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < A; m++) {

					d_c_a_avg_cnt[i][j][m] /= 10.0;
				}
			}
		}

		for (int i = 0; i < D; i++) {
			for (int j = 0; j < 3; j++) {

				d_l_avg_cnt[i][j] /= 10.0;
			}
		}

		for (int i = 0; i < D; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < 3; m++) {

					d_c_l_avg_cnt[i][j][m] /= 10.0;
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				for (int m = 0; m < A; m++) {

					l_x_a_avg_cnt[i][j][m] /= 10.0;
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < A; j++) {

				l_a_avg_sum[i][j] /= 10.0;
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				for (int m = 0; m < C; m++) {
					for (int n = 0; n < A; n++) {

						l_x_c_a_avg_cnt[i][j][m][n] /= 10.0;
					}
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < A; m++) {

					l_c_a_avg_sum[i][j][m] /= 10.0;
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				for (int m = 0; m < K; m++) {

					l_x_k_avg_cnt[i][j][m] /= 10.0;
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < K; j++) {

				l_k_avg_sum[i][j] /= 10.0;
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				for (int m = 0; m < C; m++) {
					for (int n = 0; n < K; n++) {

						l_x_c_k_avg_cnt[i][j][m][n] /= 10.0;
					}
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m > K; m++) {

					l_c_k_avg_sum[i][j][m] /= 10.0;
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {

				l_x_B_avg_cnt[i][j] /= 10.0;
			}
		}

		for (int i = 0; i < 3; i++) {
			l_B_avg_sum[i] /= 10.0;
		}

	}

	private void updateProbabilities() {

		for (int l = 0; l < 3; l++) {
			for (int x = 0; x < 4; x++) {
				if (l == 0 && x == 0) {
					for (int d = 0; d < D; d++) {
						Instance doc = (Instance) instances.get(d);
						InstanceList sents = (InstanceList) doc.getData();
						for (int s = 0; s < sents.size(); s++) {
							FeatureSequence sent = (FeatureSequence) sents.get(
									s).getData();
							for (int n = 0; n < sent.size(); n++) {
								int v = totalWordIndexMap[d][s][n];
								int topic = z[d][s];
								L_X_prob[l][x] += (d_l_avg_cnt[d][0] + gammaL)
										/ (d_avg_sum[d] + gammaL_sum)
										* // l_p
										(l_x_B_avg_cnt[0][0] + gammaX)
										/ (l_B_avg_sum[0] + gammaX_sum)
										* // x_p
										(v_B_avg_cnt[v] + beta)
										/ (B_avg_sum + beta_sum); // v_p
							}
						}

					}

				} else if (l == 1 && x == 1) {
					// l = 1, x = 1 general topic model
					for (int d = 0; d < D; d++) {
						Instance doc = (Instance) instances.get(d);
						InstanceList sents = (InstanceList) doc.getData();
						for (int s = 0; s < sents.size(); s++) {
							FeatureSequence sent = (FeatureSequence) sents.get(
									s).getData();
							for (int n = 0; n < sent.size(); n++) {
								int v = totalWordIndexMap[d][s][n];
								int topic = z[d][s];
								L_X_prob[l][x] += (d_l_avg_cnt[d][1] + gammaL)
										/ (d_avg_sum[d] + gammaL_sum)
										* (l_x_k_avg_cnt[1][1][topic] + gammaX)
										/ (l_k_avg_sum[1][topic] + gammaX_sum)
										* (v_k_avg_cnt[v][topic] + beta)
										/ (v_k_avg_sum[topic] + beta_sum);
							}
						}
					}

				} else if (l == 1 && x == 2) {
					// l = 1, x = 2 general aspect model
					for (int d = 0; d < D; d++) {
						Instance doc = (Instance) instances.get(d);
						InstanceList sents = (InstanceList) doc.getData();
						for (int s = 0; s < sents.size(); s++) {
							FeatureSequence sent = (FeatureSequence) sents.get(
									s).getData();
							for (int n = 0; n < sent.size(); n++) {
								int v = totalWordIndexMap[d][s][n];
								int aspect = y[d][s];
								L_X_prob[l][x] = (d_l_avg_cnt[d][1] + gammaL)
										/ (d_avg_sum[d] + gammaL_sum)
										* (l_x_a_avg_cnt[1][2][aspect] + gammaX)
										/ (l_a_avg_sum[1][aspect] + gammaX_sum)
										* (v_a_avg_cnt[v][aspect] + beta)
										/ (v_a_avg_sum[aspect] + beta_sum);
							}
						}

					}

				} else if (l == 1 && x == 3) {
					// l = 1, x = 3 general [topic&aspect] model
					for (int d = 0; d < D; d++) {
						Instance doc = (Instance) instances.get(d);
						InstanceList sents = (InstanceList) doc.getData();
						for (int s = 0; s < sents.size(); s++) {
							FeatureSequence sent = (FeatureSequence) sents.get(
									s).getData();
							for (int n = 0; n < sent.size(); n++) {
								int v = totalWordIndexMap[d][s][n];
								int topic = z[d][s];
								int aspect = y[d][s];
								L_X_prob[l][x] = (d_l_avg_cnt[d][1] + gammaL)
										/ (d_avg_sum[d] + gammaL_sum)
										* (l_x_k_avg_cnt[1][3][topic] + gammaX)
										/ (l_k_avg_sum[1][topic] + gammaX_sum)
										* (v_k_a_avg_cnt[v][topic][aspect] + beta)
										/ (v_k_a_avg_sum[topic][aspect] + beta_sum);
							}
						}
					}

				} else if (l == 2 && x == 1) {
					// l = 2, x = 1 specific topic model
					for (int d = 0; d < D; d++) {
						int c = doc2Collection[d];
						Instance doc = (Instance) instances.get(d);
						InstanceList sents = (InstanceList) doc.getData();
						for (int s = 0; s < sents.size(); s++) {
							FeatureSequence sent = (FeatureSequence) sents.get(
									s).getData();
							for (int n = 0; n < sent.size(); n++) {
								int v = totalWordIndexMap[d][s][n];
								int topic = z[d][s];
								int aspect = y[d][s];
								L_X_prob[l][x] = (d_c_l_avg_cnt[d][c][2] + gammaL)
										/ (d_avg_sum[d] + gammaL_sum)
										* (l_x_c_k_avg_cnt[2][1][c][topic] + gammaX)
										/ (l_c_k_avg_sum[2][c][topic] + gammaX_sum)
										* (v_c_k_avg_cnt[v][c][topic] + beta)
										/ (v_c_k_avg_sum[c][topic] + beta_sum);
							}
						}
					}

				} else if (l == 2 && x == 2) {
					// l = 2, x = 2 specific aspect model
					for (int d = 0; d < D; d++) {
						int c = doc2Collection[d];
						Instance doc = (Instance) instances.get(d);
						InstanceList sents = (InstanceList) doc.getData();
						for (int s = 0; s < sents.size(); s++) {
							FeatureSequence sent = (FeatureSequence) sents.get(
									s).getData();
							for (int n = 0; n < sent.size(); n++) {
								int v = totalWordIndexMap[d][s][n];
								int topic = z[d][s];
								int aspect = y[d][s];
								L_X_prob[l][x] = (d_c_l_avg_cnt[d][c][2] + gammaL)
										/ (d_avg_sum[d] + gammaL_sum)
										* (l_x_c_a_avg_cnt[2][2][c][aspect] + gammaX)
										/ (l_c_k_avg_sum[2][c][aspect] + gammaX_sum)
										* (v_c_a_avg_cnt[v][c][aspect] + beta)
										/ (v_c_a_avg_sum[c][aspect] + beta_sum);
							}
						}
					}

				} else if (l == 2 && x == 3) {
					// l = 2, x = 3 specific [topic&aspect] model
					for (int d = 0; d < D; d++) {
						int c = doc2Collection[d];
						Instance doc = (Instance) instances.get(d);
						InstanceList sents = (InstanceList) doc.getData();
						for (int s = 0; s < sents.size(); s++) {
							FeatureSequence sent = (FeatureSequence) sents.get(
									s).getData();
							for (int n = 0; n < sent.size(); n++) {
								int v = totalWordIndexMap[d][s][n];
								int topic = z[d][s];
								int aspect = y[d][s];
								L_X_prob[l][x] = (d_c_l_avg_cnt[d][c][2] + gammaL)
										/ (d_avg_sum[d] + gammaL_sum)
										* (l_x_c_k_avg_cnt[2][3][c][topic] + gammaX)
										/ (l_c_k_avg_sum[2][c][topic] + gammaX_sum)
										* (v_c_k_a_avg_cnt[v][c][topic][aspect] + beta)
										/ (v_c_k_a_avg_sum[c][topic][aspect] + beta_sum);
							}
						}
					}

				}
				L_X_prob[l][x] /= V;
			}
		}

		for (int v = 0; v < V; v++) { // phi_{v}^{B}
			B_v_prob[v] = (v_B_avg_cnt[v] + beta) / (B_avg_sum + beta_sum);
		}

		for (int k = 0; k < K; k++) {
			for (int v = 0; v < V; v++) {
				k_v_prob[k][v] = (v_k_avg_cnt[v][k] + beta)
						/ (v_k_avg_sum[k] + beta_sum);
			}
		}

		for (int a = 0; a < A; a++) {
			for (int v = 0; v < V; v++) {
				a_v_prob[a][v] = (v_a_avg_cnt[v][a] + beta)
						/ (v_a_avg_sum[a] + beta_sum);
			}
		}

		for (int k = 0; k < K; k++) {
			for (int a = 0; a < A; a++) {
				for (int v = 0; v < V; v++) {
					k_a_v_prob[k][a][v] = (v_k_a_avg_cnt[v][k][a] + beta)
							/ (v_k_a_avg_sum[k][a] + beta_sum);
				}
			}
		}

		for (int k = 0; k < K; k++) {
			for (int v = 0; v < V; v++) {
				for (int c = 0; c < C; c++) {
					k_v_c_prob[k][v][c] = (v_c_k_avg_cnt[v][c][k] + beta)
							/ (v_c_k_avg_sum[c][k] + beta_sum);
				}
			}
		}

		for (int a = 0; a < A; a++) {
			for (int v = 0; v < V; v++) {
				for (int c = 0; c < C; c++) {
					a_v_c_prob[a][v][c] = (v_c_a_avg_cnt[v][c][a] + beta)
							/ (v_c_a_avg_sum[c][a] + beta_sum);
				}
			}
		}

		for (int k = 0; k < K; k++) {
			for (int a = 0; a < A; a++) {
				for (int v = 0; v < V; v++) {
					for (int c = 0; c < C; c++) {
						k_a_v_c_prob[k][a][v][c] = (v_c_k_a_avg_cnt[v][c][k][a] + beta)
								/ (v_c_k_a_avg_sum[c][k][a] + beta_sum);
					}
				}
			}
		}

	}

	public void predict_labels() {
		for (int d = 0; d < D; d++) {
			Instance doc = (Instance) instances.get(d);
			InstanceList sents = (InstanceList) doc.getData();
			for (int s = 0; s < sents.size(); s++) {
				Instance inst = sents.get(s);
				FeatureSequence sent = (FeatureSequence) sents.get(s).getData();
				int topicLabel = assign_z(d, s, sent);
				inst.setTarget(topicLabel);
				// assign_y(d, s, sent);
			}
		}
	}

	private int assign_z(int d, int s, FeatureSequence sent) {
		int N = sent.size();
		double max_prob, prob = 0.0;
		double max_log_sent_prob = 0.0, log_sent_prob;
		int best_z = -1;
		int aspect = y[d][s];
		int c = doc2Collection[d];
		for (int k = 0; k < K; k++) {
			log_sent_prob = 0.0;
			for (int n = 0; n < N; n++) {
				int v = totalWordIndexMap[d][s][n];
				max_prob = 0.0;
				for (int t = 0; t < 7; t++) {
					if (t == 0) {
						prob = L_X_prob[0][0] * B_v_prob[v];
					} else if (t == 1) {
						prob = L_X_prob[1][1] * k_v_prob[k][v];
					} else if (t == 2) {
						prob = L_X_prob[1][2] * a_v_prob[aspect][v];
					} else if (t == 3) {
						prob = L_X_prob[1][3] * k_a_v_prob[k][aspect][v];
					} else if (t == 4) {
						prob = L_X_prob[2][1] * k_v_c_prob[k][v][c];
					} else if (t == 5) {
						prob = L_X_prob[2][2] * a_v_c_prob[aspect][v][c];
					} else if (t == 6) {
						prob = L_X_prob[2][3] * k_a_v_c_prob[k][aspect][v][c];
					}
					if (prob > max_prob) {
						max_prob = prob;
					}
				}
				
				log_sent_prob += Math.log(max_prob);
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

		}

		z[d][s] = best_z;
		if (best_z == -1) {
			System.out.println("Error: best_z is -1");
			System.exit(0);
		}
		return best_z;

	}

	private void assign_y(int d, int s, FeatureSequence sent) {
		int N = sent.size();
		double max_prob, prob = 0.0;
		double max_log_sent_prob = 0.0, log_sent_prob;
		int best_y = -1;
		int k  = z[d][s];
		int c = doc2Collection[d];
		for (int a = 0; a < A; a++) {
			log_sent_prob = 0.0;
			for (int n = 0; n < N; n++) {
				int v = totalWordIndexMap[d][s][n];
				max_prob = 0.0;
				for (int t = 0; t < 7; t++) {
					if (t == 0) {
						prob = L_X_prob[0][0] * B_v_prob[v];
					} else if (t == 1) {
						prob = L_X_prob[1][1] * k_v_prob[k][v];
					} else if (t == 2) {
						prob = L_X_prob[1][2] * a_v_prob[a][v];
					} else if (t == 3) {
						prob = L_X_prob[1][3] * k_a_v_prob[k][a][v];
					} else if (t == 4) {
						prob = L_X_prob[2][1] * k_v_c_prob[k][v][c];
					} else if (t == 5) {
						prob = L_X_prob[2][2] * a_v_c_prob[a][v][c];
					} else if (t == 6) {
						prob = L_X_prob[2][3] * k_a_v_c_prob[k][a][v][c];
					}
					if (prob > max_prob) {
						max_prob = prob;
					}
				}
				
				log_sent_prob += Math.log(max_prob);
			} // for(n = 0; n < N; n++)

			if (a > 0) {
				if (log_sent_prob > max_log_sent_prob) {
					max_log_sent_prob = log_sent_prob;
					best_y = a;
				}
			} else {
				max_log_sent_prob = log_sent_prob;
				best_y = a;
			}

		}
		
		if (best_y == -1) {
			System.out.println("Error: best_y is -1");
			System.exit(0);
		}
		y[d][s] = best_y;

	}

	public double[][][][] getSpecificTopicSentenceDistribution() {
		
		for(int c=0; c<C; c++){
			for (int d = 0; d < D; d++) {
				Instance doc = (Instance) instances.get(d);
				InstanceList sents = (InstanceList) doc.getData();
				for (int s = 0; s < sents.size(); s++) {
					FeatureSequence sent = (FeatureSequence) sents.get(s)
							.getData();
					double[][] p = new double[K][C];
					double[][] log_v_p = new double[K][C];
					for (int k = 0; k < K; k++) {
						double k_p = (k_avg_cnt[k] + alpha) / (k_avg_sum + alpha_sum);
						HashMap<Integer, Integer> my_k_v_cnt = new HashMap<Integer, Integer>();
						HashMap<Integer, String> v_l_r_map = new HashMap<Integer, String>();
						for (int n = 0; n < sent.size(); n++) {
							int level = l[d][s][n];
							int route = x[d][s][n];
							if ((level == 1 && route == 1) || (level == 1 && route == 3) 
									|| (level == 2 && route == 1) || (level == 2 && route == 3)) {
								int v = totalWordIndexMap[d][s][n];
								v_l_r_map.put(v, level+" "+route);
								if (!my_k_v_cnt.containsKey(v)) {
									my_k_v_cnt.put(v, 1);
								} else {
									int cnt = my_k_v_cnt.get(v);
									my_k_v_cnt.put(v, (++cnt));
								}
							}
						}

						log_v_p[k][c] = 0.0;
						Set<Integer> keys = my_k_v_cnt.keySet();
						Iterator iter = keys.iterator();
						int a = y[d][s];
						while (iter.hasNext()) {
							int v = (Integer) iter.next();
							String l_r = v_l_r_map.get(v);
							int l = Integer.parseInt(l_r.split(" ")[0]);
							int r = Integer.parseInt(l_r.split(" ")[1]);
							int cnt = my_k_v_cnt.get(v);
							if( l ==1 && r == 1){
								for (int i = 0; i < cnt; i++) {
									log_v_p[k][c] += Math.log(v_k_avg_cnt[v][k] + beta + i);
								}
							}else if(l ==1 && r == 3){
								for (int i = 0; i < cnt; i++) {
									log_v_p[k][c] += Math.log(v_k_a_avg_cnt[v][k][a] + beta + i);
								}
								
							}else if(l ==2 && r == 1){
								for (int i = 0; i < cnt; i++) {
									log_v_p[k][c] += Math.log(v_c_k_avg_cnt[v][c][k] + beta + i);
								}
							}else if(l ==2 && r == 3){
								for (int i = 0; i < cnt; i++) {
									log_v_p[k][c] += Math.log(v_c_k_a_avg_cnt[v][c][k][a] + beta + i);
								}
							}	
						}

						keys = my_k_v_cnt.keySet();
						iter = keys.iterator();
						while (iter.hasNext()) {
							int v = (Integer) iter.next();
							String l_r = v_l_r_map.get(v);
							int l = Integer.parseInt(l_r.split(" ")[0]);
							int r = Integer.parseInt(l_r.split(" ")[1]);
							int cnt = my_k_v_cnt.get(v);
							if( l ==1 && r == 1){
								for (int i = 0; i < cnt; i++) {
									log_v_p[k][c] -= Math.log(v_k_avg_sum[k] + beta_sum + i);
								}
							}else if(l ==1 && r == 3){
								for (int i = 0; i < cnt; i++) {
									log_v_p[k][c] -= Math.log(v_k_a_avg_sum[k][a] + beta_sum + i);
								}
								
							}else if(l ==2 && r == 1){
								for (int i = 0; i < cnt; i++) {
									log_v_p[k][c] -= Math.log(v_c_k_avg_sum[c][k] + beta_sum + i);
								}
							}else if(l ==2 && r == 3){
								for (int i = 0; i < cnt; i++) {
									log_v_p[k][c] -= Math.log(v_c_k_a_avg_sum[c][k][a] + beta_sum + i);
								}
							}	
						}

						p[k][c] = k_p;

					}
					
					double max_log_v_p = log_v_p[0][c];
					for (int k = 1; k < K; k++) {
						if (log_v_p[k][c] > max_log_v_p) {
							max_log_v_p = log_v_p[k][c];
						}
					}
					// smoothing
					for (int k = 0; k < K; k++) {
						log_v_p[k][c] -= max_log_v_p;
						double v_p = Math.exp(log_v_p[k][c]);
						p[k][c] *= v_p;
					}
					
				
					for (int k = 0; k < K; k++) {
						d_s_k_c_prob[d][s][k][c] = p[k][c];
					}
					
				}
			}
		}
		return d_s_k_c_prob;
	}
	
	public double[][] getSpecificTopicSentenceDistribution(int d, int s, int N) {
        double[][] p_k_c =  new double[K][C];
		double max_prob, prob = 0.0;
		double max_log_sent_prob = 0.0, log_sent_prob;

		int aspect = y[d][s];
        for(int c=0; c<C; c++){
    		for (int k = 0; k < K; k++) {
    			log_sent_prob = 0.0;
    			for (int n = 0; n < N; n++) {
    				int v = totalWordIndexMap[d][s][n];
    				max_prob = 0.0;
    				for (int t = 0; t < 7; t++) {
    					if (t == 0) {
    						prob = L_X_prob[0][0] * B_v_prob[v];
    					//	prob = B_v_prob[v];
    					} else if (t == 1) {
    						prob = L_X_prob[1][1] * k_v_prob[k][v];
    					//	prob = k_v_prob[k][v];
    					} else if (t == 2) {
    						prob = L_X_prob[1][2] * a_v_prob[aspect][v];
    					//	prob = a_v_prob[aspect][v];
    					} else if (t == 3) {
    					//	prob = L_X_prob[1][3] * k_a_v_prob[k][aspect][v];
    						prob =  k_a_v_prob[k][aspect][v];
    					} else if (t == 4) {
    						prob = L_X_prob[2][1] * k_v_c_prob[k][v][c];
    					//	prob = k_v_c_prob[k][v][c];
    					} else if (t == 5) {
    					//	prob = L_X_prob[2][2] * a_v_c_prob[aspect][v][c];
    						prob = a_v_c_prob[aspect][v][c];
    					} else if (t == 6) {
    						prob = L_X_prob[2][3] * k_a_v_c_prob[k][aspect][v][c];
    					//	prob =  k_a_v_c_prob[k][aspect][v][c];
    					}
    					if (prob > max_prob) {
    						max_prob = prob;
    					}
    				}
    				
    				log_sent_prob += Math.log(max_prob);
    			} // for(n = 0; n < N; n++)
    			
    			p_k_c[k][c] = Math.exp(log_sent_prob);

    		}
        }


		return p_k_c;

	}

	public InstanceList getInstanceList() {
		return instances;
	}

	public int[][] getTopicAssignment() {
		return z;
	}

	public int getNumberOfTopics() {
		return K;
	}

	public int getNumberOfAspects() {
		return A;
	}

}
