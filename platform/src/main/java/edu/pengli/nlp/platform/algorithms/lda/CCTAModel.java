package edu.pengli.nlp.platform.algorithms.lda;

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

// cross collection topic aspect model
public class CCTAModel {

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

	private int[][] z; // topic assignment for word of document z_{d,n}
	private int[][] y; // aspect assignment for word of document y_{d,n}
	private int[][] l; // level assignment for word of document l_{d,n}
	private int[][] x; // route assignment for word of document x_{d,n}

	private double[][] k_v_prob;
	private double[][][] k_v_c_prob;
	private double[][] a_v_prob;
	private double[][][] a_v_c_prob;

	private double[][][] k_a_v_prob;
	private double[][][][] k_a_v_c_prob;

	private double[] B_v_prob;
	private double[][] L_X_prob;

	private int[] doc2Collection;
	private int[][] totalWordIndexMap;

	// //////////////////////////////////////////////////////////////////////////////////
	// below variables for compute v_p
	// l=0 & x=0: the word come from the background model
	private int[] v_B_cnt; // the number of times word v has been assigned to
							// Background topic
	private int B_sum; // the total number of words assigned to Background topic

	private double[] v_B_avg_cnt;
	private double B_avg_sum;

	// l=1 & x=1 : the word come from the general topic model
	private int[][] v_k_cnt; // the number of times word v has been
								// assigned topic k
	double[][] v_k_avg_cnt;

	private int[] v_k_sum; // the total number of words assigned to
							// topic k
	private double[] v_k_avg_sum;

	// l=1 & x=2 : the word come from the general aspect model
	private int[][] v_a_cnt; // the number of times word v has been
								// assigned to
	private double[][] v_a_avg_cnt;
	// aspect a

	private int[] v_a_sum; // the total number of words assigned to
							// aspect a
	private double[] v_a_avg_sum;

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
	private int[][][] d_c_k_cnt; // the number of words from document i assigned
									// to specifc topic k
	private int[] d_sum; // the total number of words from document i

	private double[] d_avg_sum;

	// below variables for compute a_p
	private int[][] d_a_cnt; // the number of words from document i has been
								// assigned to general aspect a
	private int[][][] d_c_a_cnt; // the number of words from document i has been
									// assigned to specific aspect a

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

	public CCTAModel(int numTopics, int numAspects, double alpha, double beta,
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
		B_sum = 0;
		v_B_avg_cnt = new double[V];
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
		for (int i = 0; i < K; i++) {
			v_k_sum[i] = 0;
			v_k_avg_sum[i] = 0;
		}

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
		for (int i = 0; i < A; i++) {
			v_a_sum[i] = 0;
			v_a_avg_sum[i] = 0;
		}

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
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < K; j++) {
				d_k_cnt[i][j] = 0;
			}
		}

		d_c_k_cnt = new int[D][C][K];
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < K; m++) {
					d_c_k_cnt[i][j][m] = 0;
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
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < A; j++) {
				d_a_cnt[i][j] = 0;
			}
		}

		d_c_a_cnt = new int[D][C][A];
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < A; m++) {
					d_c_a_cnt[i][j][m] = 0;
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
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < A; j++) {
				l_a_sum[i][j] = 0;
			}
		}

		l_x_c_a_cnt = new int[3][4][C][A];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				for (int m = 0; m < C; m++) {
					for (int n = 0; n < A; n++) {
						l_x_c_a_cnt[i][j][m][n] = 0;
					}
				}
			}
		}

		l_c_a_sum = new int[3][C][A];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m < A; m++) {
					l_c_a_sum[i][j][m] = 0;
				}
			}
		}

		l_x_k_cnt = new int[3][4][K];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				for (int m = 0; m < K; m++) {
					l_x_k_cnt[i][j][m] = 0;
				}
			}
		}

		l_k_sum = new int[3][K];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < K; j++) {
				l_k_sum[i][j] = 0;
			}
		}

		l_x_c_k_cnt = new int[3][4][C][K];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				for (int m = 0; m < C; m++) {
					for (int n = 0; n < K; n++) {
						l_x_c_k_cnt[i][j][m][n] = 0;
					}
				}
			}
		}

		l_c_k_sum = new int[3][C][K];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < C; j++) {
				for (int m = 0; m > K; m++) {
					l_c_k_sum[i][j][m] = 0;
				}
			}
		}

		l_x_B_cnt = new int[3][4];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				l_x_B_cnt[i][j] = 0;
			}
		}

		l_B_sum = new int[3];
		for (int i = 0; i < 3; i++) {
			l_B_sum[i] = 0;
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
		z = new int[D][];
		l = new int[D][];
		y = new int[D][];
		x = new int[D][];
		totalWordIndexMap = new int[D][];
		for (int d = 0; d < D; d++) {
			Instance doc = (Instance) instances.get(d);
			FeatureSequence totalfs = getDocAllFeatures(doc);

			// doc.setData(totalfs);

			z[d] = new int[totalfs.size()];
			l[d] = new int[totalfs.size()];
			y[d] = new int[totalfs.size()];
			x[d] = new int[totalfs.size()];
			totalWordIndexMap[d] = new int[totalfs.size()];
			for (int n = 0; n < totalfs.size(); n++) {
				Object entry = totalfs.get(n);
				int v = totalDict.lookupIndex(entry);
				totalWordIndexMap[d][n] = v;
			}
		}

		for (int d = 0; d < D; d++) {
			Instance doc = (Instance) instances.get(d);
			int c = docCollectionMap.get(doc);
			FeatureSequence fs = getDocAllFeatures(doc);
			for (int n = 0; n < fs.size(); n++) {
				int topic = (int) Math.floor(Math.random() * K);
				int aspect = (int) Math.floor(Math.random() * A);
				z[d][n] = topic;
				y[d][n] = aspect;
				int level = r.nextInt(3);
				l[d][n] = level;
				int route = r.nextInt(4);
				x[d][n] = route;

				int v = totalWordIndexMap[d][n];

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
			FeatureSequence fs = getDocAllFeatures(doc);
			for (int n = 0; n < fs.size(); n++) {
				sampling_z_y_l_x(d, n);
			}

		}
	}

	private void sampling_z_y_l_x(int d, int n) {

		int topic = z[d][n];
		int aspect = y[d][n];

		int level = l[d][n];
		int route = x[d][n];

		int c = doc2Collection[d];
		int v = totalWordIndexMap[d][n];

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

		// sample new value for topic
		pTotal = 0.0;
		p = new double[K];
		double v_p = 0;
		double k_p = 0;

		if (level == 0 && route == 0) {
			for (int k = 0; k < K; k++) {
				k_p = (d_k_cnt[d][k] + alpha) / (d_sum[d] + alpha_sum);
				p[k] = k_p;

			}

		} else if (level == 1 && route == 1) {
			for (int k = 0; k < K; k++) {
				k_p = (d_k_cnt[d][k] + alpha) / (d_sum[d] + alpha_sum);
				v_p = (v_k_cnt[v][k] + beta) / (v_k_sum[k] + beta_sum);
				p[k] = k_p * v_p;
			}

		} else if (level == 1 && route == 2) {
			for (int k = 0; k < K; k++) {
				k_p = (d_k_cnt[d][k] + alpha) / (d_sum[d] + alpha_sum);
				p[k] = k_p;

			}
		} else if (level == 1 && route == 3) {
			for (int k = 0; k < K; k++) {
				k_p = (d_k_cnt[d][k] + alpha) / (d_sum[d] + alpha_sum);
				v_p = (v_k_a_cnt[v][k][aspect] + beta)
						/ (v_k_a_sum[k][aspect] + beta_sum);
				p[k] = k_p * v_p;

			}

		} else if (level == 2 && route == 1) {
			for (int k = 0; k < K; k++) {
				k_p = (d_c_k_cnt[d][c][k] + alpha) / (d_sum[d] + alpha_sum);
				v_p = (v_c_k_cnt[v][c][k] + beta)
						/ (v_c_k_sum[c][k] + beta_sum);
				p[k] = k_p * v_p;
			}
		} else if (level == 2 && route == 2) {
			for (int k = 0; k < K; k++) {
				k_p = (d_c_k_cnt[d][c][k] + alpha) / (d_sum[d] + alpha_sum);
				p[k] = k_p * v_p;
			}

		} else if (level == 2 && route == 3) {
			for (int k = 0; k < K; k++) {
				k_p = (d_c_k_cnt[d][c][k] + alpha) / (d_sum[d] + alpha_sum);
				v_p = (v_c_k_a_cnt[v][c][k][aspect] + beta)
						/ (v_c_k_a_sum[c][k][aspect] + beta_sum);
				p[k] = k_p * v_p;
			}
		}

		tmp = 0;
		X = Math.random() * pTotal;
		for (int i = 0; i < K; i++) {
			tmp += p[i];
			if (tmp > X) {
				topic = i;
				break;
			}
		}

		// sample new value for aspect
		pTotal = 0.0;
		p = new double[A];
		double a_p = 0;
		v_p = 0;

		if (level == 0 && route == 0) {
			for (int a = 0; a < A; a++) {
				a_p = (d_a_cnt[d][a] + beta) / (d_sum[d] + beta_sum);
				p[a] = a_p;
			}

		} else if (level == 1 && route == 1) {
			for (int a = 0; a < A; a++) {
				a_p = (d_a_cnt[d][a] + beta) / (d_sum[d] + beta_sum);
				p[a] = a_p;
			}

		} else if (level == 1 && route == 2) {
			for (int a = 0; a < A; a++) {
				a_p = (d_a_cnt[d][a] + beta) / (d_sum[d] + beta_sum);
				v_p = (v_a_cnt[v][a] + beta) / (v_k_sum[a] + beta_sum);
				p[a] = a_p;
			}
		} else if (level == 1 && route == 3) {
			for (int a = 0; a < A; a++) {
				a_p = (d_a_cnt[d][a] + beta) / (d_sum[d] + beta_sum);
				v_p = (v_k_a_cnt[v][topic][a] + beta)
						/ (v_k_a_sum[topic][a] + beta_sum);
				p[a] = a_p;
			}

		} else if (level == 2 && route == 1) {
			for (int a = 0; a < A; a++) {
				a_p = (d_c_a_cnt[d][c][a] + beta) / (d_sum[d] + beta_sum);
				p[a] = a_p;

			}
		} else if (level == 2 && route == 2) {
			for (int a = 0; a < A; a++) {
				a_p = (d_c_a_cnt[d][c][a] + beta) / (d_sum[d] + beta_sum);
				v_p = (v_c_a_cnt[v][c][a] + beta)
						/ (v_c_a_sum[c][a] + beta_sum);
				p[a] = a_p;
			}

		} else if (level == 2 && route == 3) {
			for (int a = 0; a < A; a++) {
				a_p = (d_c_a_cnt[d][c][a] + beta) / (d_sum[d] + beta_sum);
				v_p = (v_c_k_a_cnt[v][c][topic][a] + beta)
						/ (v_c_k_a_sum[c][topic][a] + beta_sum);
				p[a] = a_p;
			}
		}

		tmp = 0;
		X = Math.random() * pTotal;
		for (int i = 0; i < A; i++) {
			tmp += p[i];
			if (tmp > X) {
				aspect = i;
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
		z[d][n] = topic;
		l[d][n] = level;
		y[d][n] = aspect;
		x[d][n] = route;
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

		for (int i = 0; i < V; i++) {
			for (int j = 0; j < A; j++) {
				v_a_avg_cnt[i][j] += v_a_cnt[i][j];
			}
		}

		for (int i = 0; i < A; i++) {
			v_a_avg_sum[i] += v_a_sum[i];
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

		// /////////////

		for (int i = 0; i < D; i++) {

			d_avg_sum[i] += d_sum[i];
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

	}

	private void averageCnt() {

		for (int i = 0; i < V; i++) {
			for (int j = 0; j < K; j++) {
				v_k_avg_cnt[i][j] /= 10.0;
			}
		}

		for (int i = 0; i < K; i++) {

			v_k_avg_sum[i] /= 10.0;
		}

		for (int i = 0; i < V; i++) {
			for (int j = 0; j < A; j++) {
				v_a_avg_cnt[i][j] /= 10.0;
			}
		}

		for (int i = 0; i < A; i++) {
			v_a_avg_sum[i] /= 10.0;
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

	}

	private void updateProbabilities() {

		for (int l = 0; l < 3; l++) {
			for (int x = 0; x < 4; x++) {
				if (l == 0 && x == 0) {
					for (int d = 0; d < D; d++) {
						Instance doc = (Instance) instances.get(d);
						FeatureSequence allFeatures = getDocAllFeatures(doc);
						for (int n = 0; n < allFeatures.size(); n++) {
							int v = totalWordIndexMap[d][n];
							int topic = z[d][n];
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

				} else if (l == 1 && x == 1) {
					// l = 1, x = 1 general topic model
					for (int d = 0; d < D; d++) {
						Instance doc = (Instance) instances.get(d);
						InstanceList sents = (InstanceList) doc.getData();
						for (int s = 0; s < sents.size(); s++) {
							FeatureSequence sent = (FeatureSequence) sents.get(
									s).getData();
							for (int n = 0; n < sent.size(); n++) {
								int v = totalWordIndexMap[d][n];
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
								int v = totalWordIndexMap[d][n];
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
								int v = totalWordIndexMap[d][n];
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
								int v = totalWordIndexMap[d][n];
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
								int v = totalWordIndexMap[d][n];
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
								int v = totalWordIndexMap[d][n];
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

		for (int k = 0; k < K; k++) {
			for (int v = 0; v < V; v++) {
				k_v_prob[k][v] = (v_k_avg_cnt[v][k] + beta)
						/ (v_k_avg_sum[k] + beta_sum);
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
				a_v_prob[a][v] = (v_a_avg_cnt[v][a] + beta)
						/ (v_a_avg_sum[a] + beta_sum);
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
					k_a_v_prob[k][a][v] = (v_k_a_avg_cnt[v][k][a] + beta)
							/ (v_k_a_avg_sum[k][a] + beta_sum);
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
				int topicLabel = assign_z_sent(d, sent);
				int aspectLabel = assign_y_sent(d, sent);
				inst.setTarget(topicLabel+"_"+aspectLabel);
			}
		}
	}

	private int assign_z_sent(int d, FeatureSequence sent) {
		
		int N = sent.size();
		double max_prob, prob = 0.0;
		double max_log_sent_prob = 0.0, log_sent_prob;
		int best_z = -1;
		
		int c = doc2Collection[d];
		for (int k = 0; k < K; k++) {
			log_sent_prob = 0.0;
			for (int n = 0; n < N; n++) {
				int aspect = y[d][n];
				int v = totalWordIndexMap[d][n];
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
		
		if (best_z == -1) {
			System.out.println("Error: best_z is -1");
			System.exit(0);
		}
		return best_z;
		
	}

	private int assign_y_sent(int d, FeatureSequence sent) {
		int N = sent.size();
		double max_prob, prob = 0.0;
		double max_log_sent_prob = 0.0, log_sent_prob;
		int best_y = -1;
	
		int c = doc2Collection[d];
		for (int a = 0; a < A; a++) {
			log_sent_prob = 0.0;
			for (int n = 0; n < N; n++) {
				int k = z[d][n];
				int v = totalWordIndexMap[d][n];
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
		return best_y;

	}

	public int[][] getTopicAssignment() {
		return z;
	}

	public int[][] getAspectAssignment() {
		return y;
	}

	public InstanceList getInstanceList() {
		return instances;
	}

	public Alphabet getDictionary() {
		return totalDict;
	}

	public int[] getDoc2Collection() {
		return doc2Collection;
	}

	public double[][] getGeneralTopicWordDistribution() {
		return this.k_v_prob;
	}

	public double[][][] getCollectionSpecificTopicWordDistribution() {
		return this.k_v_c_prob;
	}

	public double[][] getGeneralAspectWordDistribution() {
		return this.a_v_prob;
	}

	public double[][][] getCollectionSpecificAspectWordDistribution() {
		return this.a_v_c_prob;
	}

	public double[][][] getGeneralMixWordDistribution() {
		return this.k_a_v_prob;
	}

	public double[][][][] getCollectionSpecificMixWordDistribution() {
		return this.k_a_v_c_prob;
	}

	public int[][] getTotalWordIndexMap() {
		return totalWordIndexMap;
	}

	public int getNumberOfTopics() {
		return K;
	}

	public int getNumberOfAspects() {
		return A;
	}

	private FeatureSequence getDocAllFeatures(Instance doc) {
		Alphabet dictionary = new Alphabet();
		InstanceList sents = (InstanceList) doc.getData();
		int length = 0;
		for (Instance sent : sents) {
			FeatureSequence fs = (FeatureSequence) sent.getData();
			length += fs.size();
		}
		FeatureSequence totalfs = new FeatureSequence(dictionary, length);
		for (Instance sent : sents) {
			FeatureSequence fs = (FeatureSequence) sent.getData();
			for (int i = 0; i < fs.size(); i++) {
				Object obj = fs.get(i);
				totalfs.add(obj);
			}
		}

		return totalfs;
	}

	public void output_model() {
		int topK = 10;

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
		for (int a = 0; a < A; a++) {
			System.out.println("Common Aspect " + a + "th:");
			HashMap<Integer, Double> map = new HashMap<Integer, Double>();
			for (int v = 0; v < V; v++) {
				double pro = a_v_prob[a][v];
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

		for (int c = 0; c < C; c++) {
			for (int k = 0; k < K; k++) {
				System.out.println("Collection " + c + " " + "Specific Topic "
						+ k + "th:");
				HashMap<Integer, Double> map = new HashMap<Integer, Double>();
				for (int v = 0; v < V; v++) {
					double pro = k_v_c_prob[k][v][c];
					map.put(v, pro);
				}
				LinkedHashMap rankedMap = RankMap.sortHashMapByValues(map,
						false);
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
			for (int a = 0; a < A; a++) {
				System.out.println("Collection " + c + " " + "Specific aspect "
						+ a + "th:");
				HashMap<Integer, Double> map = new HashMap<Integer, Double>();
				for (int v = 0; v < V; v++) {
					double pro = a_v_c_prob[a][v][c];
					map.put(v, pro);
				}
				LinkedHashMap rankedMap = RankMap.sortHashMapByValues(map,
						false);
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

}
